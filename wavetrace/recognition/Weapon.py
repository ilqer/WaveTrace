"""Phase 7p-b/c — WeaponHead: Stage-E backends behind the same wrapper API as PresenceHead.

Input contract BY BACKEND (the σ²[p] paths and the CNN consume different x_t):
  * "variance" / "mlp" / "svm" → `X_intercarrier` (n, 27) — the InterCarrierExtractor block
    (µ|σ²|CV × 9 stats) built from RAW magnitudes: `build_dataset(gain_lock=None, intercarrier=True)`.
    NEVER the gain-locked presence features (the lock erases the metal signature — P6 gotcha).
  * "cnn" → `X_image` (n, K, window); a flattened (n, K·window) row is also accepted so
    `InferenceSession.predict_window` works unchanged (it reshapes internally).

Backends:
  * "variance" — the Yousaf BASELINE-FIRST head (plan §5 7a): one threshold on the window's mean
    σ²[p] (column 9 = the sigma2-series mean). Physics says metal → LOWER inter-carrier variance,
    but fit() learns threshold AND direction from data (orientation-robust). Fit O(n log n)
    (sorted prefix scan maximizing balanced accuracy); predict O(1)/window. predict_proba = a
    logistic in the threshold margin (scale = robust σ of the feature) — calibration-free but
    monotone, which is all the soft vote needs.
  * "mlp" / "svm" — the shared sklearn pipeline (`Model.sklearn_pipeline`).
  * "cnn" — torch 2D-CNN on the CSI image (small LUMS-style net: 2×conv → adaptive pool → 2 dense;
    ported architecture, NOT their random-KFold eval — rev-7 #1). torch is imported LAZILY
    (optional dep: `pip install wavetrace[cnn]`); every other backend works without it.
    Deterministic via torch.manual_seed(config.seed). Trains/infers on Pi/laptop, never the ESP32.
"""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.recognition.Model import sklearn_pipeline

# Column of the 27-block holding the WINDOW MEAN of the per-packet σ²[p] series
# (series order µ|σ²|CV, 9 stats each, stat 0 = mean).
VARIANCE_FEATURE = 9


def _torch():
    """Lazy torch import — only the 'cnn' backend needs it."""
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover - exercised only without the [cnn] extra
        raise ImportError(
            "WeaponHead backend 'cnn' requires PyTorch: pip install 'wavetrace[cnn]'"
        ) from e


def _build_net(torch, hidden: int, num_classes: int):
    """LUMS-style small 2D-CNN; AdaptiveAvgPool makes it (K, window)-agnostic."""
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(), nn.Linear(16 * 16, hidden), nn.ReLU(), nn.Linear(hidden, num_classes),
    )


class WeaponHead:
    """Stage-E weapon head (binary weapon / no-weapon; same API as PresenceHead)."""

    def __init__(self, config: ModelConfig, *, variance_feature: int = VARIANCE_FEATURE):
        self.config = config
        self._vf = int(variance_feature)
        # how the serving layer (Cli.run) must assemble x: "ic27" | "fusion" | "cnn" (set by
        # train_weapon, persisted in save). None for directly-constructed heads.
        self.feature_mode: str | None = None
        self._pipe = sklearn_pipeline(config) if config.backend in ("mlp", "svm") else None
        self._classes: np.ndarray | None = None
        # variance-backend state
        self._thr = self._scale = None
        self._positive_below = True
        # cnn-backend state
        self._net = None
        self._norm = None          # (mean, std) train normalization
        self._image_shape = None   # (K, window) for reshaping flattened windows

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return self._pipe.classes_ if self._pipe is not None else self._classes

    # ----- fit ------------------------------------------------------------------------------------

    def fit(self, X, y, *, epochs: int = 30, lr: float = 1e-3, batch_size: int = 32) -> "WeaponHead":
        """Fit on (n, 27) inter-carrier blocks (variance/mlp/svm) or (n, K, window) images (cnn).
        epochs/lr/batch_size apply to the cnn backend only. Offline. Returns self."""
        y = np.asarray(y, dtype=np.int64)
        classes = np.unique(y)
        if classes.size < 2:
            # 1-class data -> a model that only ever predicts that class (silent failure); refuse.
            # (The synthetic weapon signature is off unless --weapon-depth > 0 — see Cli.py warning.)
            raise ValueError(
                f"WeaponHead.fit: training data has a single class {classes.tolist()}; need both "
                "weapon and no-weapon windows (check weapon label spans / --weapon-depth)"
            )
        if self.config.backend == "variance":
            self._fit_variance(np.asarray(X, dtype=np.float32), y)
        elif self.config.backend == "cnn":
            self._fit_cnn(X, y, epochs=epochs, lr=lr, batch_size=batch_size)
        else:
            self._pipe.fit(np.asarray(X, dtype=np.float32), y)
        return self

    def _fit_variance(self, X, y) -> None:
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError(f"variance backend is binary, got classes {classes.tolist()}")
        x = X[:, self._vf]
        order = np.argsort(x, kind="stable")
        xs = x[order]
        is_pos = (y[order] == classes[1]).astype(np.int64)
        P = int(is_pos.sum())
        N = int(is_pos.size - P)
        if P == 0 or N == 0:
            raise ValueError("variance backend needs both classes in the training data")
        pos_below = np.cumsum(is_pos)              # positives among xs[:i+1]
        n_below = np.arange(1, xs.size + 1)
        # balanced accuracy of "positive when x <= thr" at every split point (and its complement
        # for the opposite direction); one O(n) pass over the sorted feature
        tpr_low = pos_below / P
        tnr_low = (N - (n_below - pos_below)) / N
        bal_low = (tpr_low + tnr_low) / 2.0
        bal_high = 1.0 - bal_low                   # flipping the direction flips both rates
        valid = np.empty(xs.size, dtype=bool)      # no threshold between equal feature values
        valid[:-1] = xs[:-1] < xs[1:]
        valid[-1] = False
        if not valid.any():
            raise ValueError("variance backend: feature is constant, nothing to threshold")
        i_low = int(np.flatnonzero(valid)[np.argmax(bal_low[valid])])
        i_high = int(np.flatnonzero(valid)[np.argmax(bal_high[valid])])
        self._positive_below = bool(bal_low[i_low] >= bal_high[i_high])
        i = i_low if self._positive_below else i_high
        self._thr = float((xs[i] + xs[i + 1]) / 2.0)
        mad = float(np.median(np.abs(x - np.median(x))))
        self._scale = 1.4826 * mad if mad > 0 else (float(x.std()) or 1.0)  # robust σ for the logistic
        self._classes = classes

    def _fit_cnn(self, X, y, *, epochs, lr, batch_size) -> None:
        torch = _torch()
        imgs = self._as_images(np.asarray(X, dtype=np.float32))
        self._image_shape = imgs.shape[1:]
        self._classes = np.unique(y)
        y_idx = np.searchsorted(self._classes, y)  # logits column i <-> sorted class i
        mean, std = float(imgs.mean()), float(imgs.std()) or 1.0
        self._norm = (mean, std)
        torch.manual_seed(self.config.seed)
        net = _build_net(torch, self.config.hidden, int(self._classes.size))
        xt = torch.from_numpy((imgs - mean) / std).unsqueeze(1)  # (n, 1, K, window)
        yt = torch.from_numpy(y_idx.astype(np.int64))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.config.seed)
        net.train()
        for _ in range(epochs):
            for idx in torch.randperm(xt.shape[0], generator=gen).split(batch_size):
                opt.zero_grad()
                loss = loss_fn(net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        net.eval()
        self._net = net

    # ----- predict --------------------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """(n, d)|(n, K, window) -> (n,) class ids. O(1) per row (CNN: O(K·window))."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """-> (n, C) probabilities, columns ordered by classes_."""
        self._require_fitted()
        X = np.asarray(X, dtype=np.float32)
        if self.config.backend == "variance":
            # logistic in the threshold margin; sign flips with the learned direction
            m = (self._thr - X[:, self._vf]) / self._scale
            p_pos = 1.0 / (1.0 + np.exp(-(m if self._positive_below else -m)))
            return np.stack([1.0 - p_pos, p_pos], axis=1)
        if self.config.backend == "cnn":
            torch = _torch()
            imgs = (self._as_images(X) - self._norm[0]) / self._norm[1]
            with torch.no_grad():
                logits = self._net(torch.from_numpy(imgs).unsqueeze(1))
                return torch.softmax(logits, dim=1).numpy()
        return self._pipe.predict_proba(X)

    def _as_images(self, X) -> np.ndarray:
        """Accept (n, K, window) or flattened (n, K·window) (the predict_window seam)."""
        if X.ndim == 3:
            return np.ascontiguousarray(X)
        shape = self._image_shape or (self.config.k, self.config.window)
        return np.ascontiguousarray(X.reshape(X.shape[0], *shape))

    # ----- persist --------------------------------------------------------------------------------

    def save(self, path) -> Path:
        self._require_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {"config": asdict(self.config), "variance_feature": self._vf,
                "feature_mode": self.feature_mode}
        if self._pipe is not None:
            blob["pipeline"] = self._pipe
        elif self.config.backend == "variance":
            blob.update(thr=self._thr, scale=self._scale, positive_below=self._positive_below,
                        classes=self._classes)
        else:  # cnn: state as numpy arrays (torch needed only to LOAD, not to open the file)
            blob.update(state={k: v.cpu().numpy() for k, v in self._net.state_dict().items()},
                        norm=self._norm, image_shape=self._image_shape, classes=self._classes)
        joblib.dump(blob, p)
        return p

    @classmethod
    def load(cls, path) -> "WeaponHead":
        blob = joblib.load(path)
        head = cls(ModelConfig(**blob["config"]), variance_feature=blob["variance_feature"])
        head.feature_mode = blob.get("feature_mode")  # absent in pre-Phase-8 models -> None
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
            head._image_shape = tuple(blob["image_shape"])
            net = _build_net(torch, head.config.hidden, int(head._classes.size))
            net.load_state_dict({k: torch.from_numpy(v) for k, v in blob["state"].items()})
            net.eval()
            head._net = net
        return head

    def _require_fitted(self) -> None:
        fitted = (self._pipe is not None and hasattr(self._pipe, "classes_")) \
            or self._thr is not None or self._net is not None
        if not fitted:
            raise ValueError("WeaponHead: not fitted (call fit() or load())")
