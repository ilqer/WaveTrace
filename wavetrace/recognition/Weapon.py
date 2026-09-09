"""Phase 7p-b/c: WeaponHead backends.

Input by backend:
* 'variance', 'mlp', 'svm': X_intercarrier (built from RAW magnitudes). Do NOT use gain-locked features.
* 'cnn': X_image.

Backends:
* 'variance': Baseline threshold on mean variance. Physics: metal lowers variance. Fit O(n log n), predict O(1).
* 'mlp'/'svm': Shared sklearn pipeline.
* 'cnn': 2D-CNN on CSI image. Torch is lazy imported.
"""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.domain.contracts import SCHEMA_VERSION, PipelineContract, derive_pipeline_contract
from wavetrace.recognition.Model import sklearnPipeline

# Column of the 27-block holding the WINDOW MEAN of the per-packet σ²[p] series (µ|σ²|CV, 9 stats each, stat 0 = mean).
VARIANCE_FEATURE = 9


def _torch():
    """Lazy torch import."""
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover - exercised only without the [cnn] extra
        raise ImportError(
            "WeaponHead backend 'cnn' requires PyTorch: pip install 'wavetrace[cnn]'"
        ) from e


def _buildNet(torch, hidden: int, num_classes: int, in_channels: int = 1):
    """Small 2D-CNN. AdaptiveAvgPool makes it (K, window)-agnostic."""
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(), nn.Linear(16 * 16, hidden), nn.ReLU(), nn.Linear(hidden, num_classes),
    )


class WeaponHead:
    """Stage-E weapon head (binary weapon / no-weapon; same API as PresenceHead)."""

    def __init__(self, config: ModelConfig, *, variance_feature: int = VARIANCE_FEATURE):
        self.config = config
        self._vf = int(variance_feature)
        # how the serving layer (Cli.run) must assemble x: "ic27" | "fusion" | "cnn"; None for directly-constructed heads.
        self.feature_mode: str | None = None
        self._pipe = sklearnPipeline(config) if config.backend in ("mlp", "svm") else None
        self._classes: np.ndarray | None = None
        # variance-backend state
        self._thr = self._scale = None
        self._positive_below = True
        # cnn-backend state
        self._net = None
        self._norm = None          # (mean, std) train normalization
        self._image_shape = None   # (K, window) for reshaping flattened windows
        self.contract = derive_pipeline_contract(config)  # what this head trains/serves against

    @property
    def classes_(self) -> np.ndarray:
        self._requireFitted()
        return self._pipe.classes_ if self._pipe is not None else self._classes

    # ----- fit ------------------------------------------------------------------------------------

    def fit(self, X, y, *, epochs: int = 30, lr: float = 1e-3, batch_size: int = 32,
            report=None) -> "WeaponHead":
        """Fit on inter-carrier blocks or images. Returns self."""
        y = np.asarray(y, dtype=np.int64)
        classes = np.unique(y)
        if classes.size < 2:
            # 1-class data -> a model that only ever predicts that class (silent failure); refuse.
            raise ValueError(
                f"WeaponHead.fit: training data has a single class {classes.tolist()}; need both "
                "weapon and no-weapon windows (check weapon label spans / --weapon-depth)"
            )
        if self.config.backend == "variance":
            self._fitVariance(np.asarray(X, dtype=np.float32), y)
        elif self.config.backend == "cnn":
            self._fitCnn(X, y, epochs=epochs, lr=lr, batch_size=batch_size, report=report)
        else:
            self._pipe.fit(np.asarray(X, dtype=np.float32), y)
        return self

    def _fitVariance(self, X, y) -> None:
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"variance backend is binary, got classes {classes.tolist()}")
        x = X[:, self._vf]
        order = np.argsort(x, kind="stable")
        xs = x[order]
        isPos = (y[order] == classes[1]).astype(np.int64)
        P = int(isPos.sum())
        N = int(isPos.size - P)
        if P == 0 or N == 0:
            raise ValueError("variance backend needs both classes in the training data")
        posBelow = np.cumsum(isPos)              # positives among xs[:i+1]
        nBelow = np.arange(1, xs.size + 1)
        # balanced accuracy of "positive when x <= thr" at every split point; one O(n) pass over the sorted feature
        tprLow = posBelow / P
        tnrLow = (N - (nBelow - posBelow)) / N
        balLow = (tprLow + tnrLow) / 2.0
        balHigh = 1.0 - balLow                   # flipping the direction flips both rates
        valid = np.empty(xs.size, dtype=bool)      # no threshold between equal feature values
        valid[:-1] = xs[:-1] < xs[1:]
        valid[-1] = False
        if not valid.any():
            raise ValueError("variance backend: feature is constant, nothing to threshold")
        iLow = int(np.flatnonzero(valid)[np.argmax(balLow[valid])])
        iHigh = int(np.flatnonzero(valid)[np.argmax(balHigh[valid])])
        self._positive_below = bool(balLow[iLow] >= balHigh[iHigh])
        i = iLow if self._positive_below else iHigh
        self._thr = float((xs[i] + xs[i + 1]) / 2.0)
        mad = float(np.median(np.abs(x - np.median(x))))
        self._scale = 1.4826 * mad if mad > 0 else (float(x.std()) or 1.0)  # robust σ for the logistic
        self._classes = classes

    def _fitCnn(self, X, y, *, epochs, lr, batch_size, report=None) -> None:
        torch = _torch()
        imgs = self._asImages(np.asarray(X, dtype=np.float32))  # (n, C, K, W) — 4-D
        self._image_shape = imgs.shape[1:]                        # (C, K, W) — always 3-tuple
        in_channels = imgs.shape[1]
        self._classes = np.unique(y)
        yIdx = np.searchsorted(self._classes, y)
        mean, std = float(imgs.mean()), float(imgs.std()) or 1.0
        self._norm = (mean, std)
        torch.manual_seed(self.config.seed)
        net = _buildNet(torch, self.config.hidden, int(self._classes.size), in_channels=in_channels)
        xt = torch.from_numpy((imgs - mean) / std)  # (n, C, K, W) — no unsqueeze
        yt = torch.from_numpy(yIdx.astype(np.int64))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        lossFn = torch.nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.config.seed)
        net.train()
        n = xt.shape[0]
        for ep in range(epochs):
            batchLosses = []
            for idx in torch.randperm(n, generator=gen).split(batch_size):
                opt.zero_grad()
                loss = lossFn(net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                batchLosses.append(loss.item())
            epLossAvg = float(np.mean(batchLosses)) if batchLosses else 0.0
            print(f"      cnn ep {ep+1:3d}/{epochs}  loss={epLossAvg:.4f}", end="\r", flush=True)
            if report is not None:
                # batch-loss spread = confidence band on the live training curve; accuracy from a cheap eval-mode pass
                epLossStd = float(np.std(batchLosses)) if len(batchLosses) > 1 else 0.0
                net.eval()
                with torch.no_grad():
                    acc = float((net(xt).argmax(1) == yt).float().mean())
                net.train()
                report(ep + 1, {"loss": epLossAvg, "loss_std": epLossStd, "acc": acc})
        print()
        net.eval()
        self._net = net

    # ----- predict --------------------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Predict class ids."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        self._requireFitted()
        X = np.asarray(X, dtype=np.float32)
        if self.config.backend == "variance":
            # logistic in the threshold margin; sign flips with the learned direction
            m = (self._thr - X[:, self._vf]) / self._scale
            pPos = 1.0 / (1.0 + np.exp(-(m if self._positive_below else -m)))
            return np.stack([1.0 - pPos, pPos], axis=1)
        if self.config.backend == "cnn":
            torch = _torch()
            imgs = (self._asImages(X) - self._norm[0]) / self._norm[1]  # 4-D (n,C,K,W)
            with torch.no_grad():
                logits = self._net(torch.from_numpy(imgs))  # no unsqueeze — already 4-D
                return torch.softmax(logits, dim=1).numpy()
        return self._pipe.predict_proba(X)

    def _asImages(self, X) -> np.ndarray:
        """Format to 4-D (n,C,K,W) for cnn fit and predict_proba."""
        if X.ndim == 4:
            return np.ascontiguousarray(X)
        if X.ndim == 3:
            return np.ascontiguousarray(X[:, np.newaxis, :, :])  # (n, 1, K, W)
        shape = self._image_shape or (1, self.config.k, self.config.window)
        return np.ascontiguousarray(X.reshape(X.shape[0], *shape))

    # ----- persist --------------------------------------------------------------------------------

    def save(self, path) -> Path:
        self._requireFitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {"config": asdict(self.config), "variance_feature": self._vf,
                "feature_mode": self.feature_mode,
                "contract": self.contract.to_dict(),
                "schema_version": SCHEMA_VERSION}
        if self._pipe is not None:
            blob["pipeline"] = self._pipe
        elif self.config.backend == "variance":
            blob.update(thr=self._thr, scale=self._scale, positive_below=self._positive_below,
                        classes=self._classes)
        else:  # cnn: state as numpy arrays (torch needed only to LOAD, not to open the file)
            # image_shape is always a 3-tuple (C, K, W) from P10; pre-P10 blobs may have 2-tuple
            blob.update(state={k: v.cpu().numpy() for k, v in self._net.state_dict().items()},
                        norm=self._norm, image_shape=tuple(self._image_shape), classes=self._classes)
        joblib.dump(blob, p)
        return p

    @classmethod
    def load(cls, path) -> "WeaponHead":
        blob = joblib.load(path)
        head = cls(ModelConfig(**blob["config"]), variance_feature=blob["variance_feature"])
        head.feature_mode = blob.get("feature_mode")  # absent in pre-Phase-8 models -> None
        # absent in artifacts saved before PipelineContract existed -> fall back to the same
        # derivation `save` uses, so every existing model keeps loading unchanged.
        head.contract = (PipelineContract.from_dict(blob["contract"]) if "contract" in blob
                          else derive_pipeline_contract(head.config))
        if "pipeline" in blob:
            head._pipe = blob["pipeline"]
        elif head.config.backend == "variance":
            head._thr, head._scale = blob["thr"], blob["scale"]
            head._positive_below = blob["positive_below"]
            head._classes = blob["classes"]
        else:
            torch = _torch()
            head._classes = blob["classes"]
            head._norm = blob["norm"]
            rawShape = tuple(blob["image_shape"])
            # pre-P10 blobs store a 2-tuple (K, W); prepend C=1 to get the canonical 3-tuple
            head._image_shape = rawShape if len(rawShape) == 3 else (1,) + rawShape
            in_channels = head._image_shape[0]
            net = _buildNet(torch, head.config.hidden, int(head._classes.size),
                             in_channels=in_channels)
            net.load_state_dict({k: torch.from_numpy(v) for k, v in blob["state"].items()})
            net.eval()
            head._net = net
        return head

    def _requireFitted(self) -> None:
        fitted = (self._pipe is not None and hasattr(self._pipe, "classes_")) \
            or self._thr is not None or self._net is not None
        if not fitted:
            raise ValueError("WeaponHead: not fitted (call fit() or load())")
