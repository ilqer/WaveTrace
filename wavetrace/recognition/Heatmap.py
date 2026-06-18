"""Camera-supervised G×G occupancy heatmap head.

Same wrapper shape as WeaponHead (fit/predict/save/load) so the serving + UI paths reuse it.
Works for HUMAN or WEAPON depending only on which mask the SegmentationLabeler supervised on.

Input  : X_image (n, C, K, window) — C = nodes (multi-node images), same tensor WeaponHead eats.
Target : Y (n, G, G) occupancy in [0,1] from CameraLabeler (Label.mask reshaped to G×G).
Output : (n, G, G) per-cell probability. 'exists' = heatmap.max() > threshold.
Loss   : BCE + soft-Dice (Dice handles the mostly-empty grid; BCE keeps calibration).
Architecture: small encoder->decoder that maps (C, K, W) CSI to (1, G, G); AdaptiveAvgPool
makes it (K, W)-agnostic (band/size-agnostic) so training and serving can differ in window size.

torch is a LAZY optional dep (pip install wavetrace[cnn]); everything else imports without it.
"""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from wavetrace.Config import ModelConfig


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("HeatmapHead needs PyTorch: pip install 'wavetrace[cnn]'") from e


def _build_unet(torch, in_channels: int, grid: int):
    """Small CSI->heatmap CNN. AdaptiveAvgPool resolves any (K, W) -> (G, G). Pi-friendly."""
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((grid, grid)),
        nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 1, 1),    # per-cell logit
    )


class HeatmapHead:
    """Camera-supervised CSI occupancy heatmap regressor (binary occupancy per cell)."""

    def __init__(self, config: ModelConfig, *, grid: int = 16):
        self.config = config
        self.grid = int(grid)
        self.feature_mode = "heatmap"
        self._net = None
        self._norm: tuple[float, float] | None = None
        self._image_shape: tuple | None = None     # (C, K, W)

    def fit(self, X, Y, *, epochs=40, lr=1e-3, batch_size=16,
            report=None) -> "HeatmapHead":
        """X: (n, C, K, W) images. Y: (n, G, G) or (n, G*G) occupancy in [0,1].
        report: optional callback(epoch, {"loss": float}) for live UI metrics."""
        torch = _torch()
        X = self._as_images(np.asarray(X, dtype=np.float32))
        self._image_shape = X.shape[1:]
        Y = np.asarray(Y, dtype=np.float32).reshape(X.shape[0], self.grid, self.grid)
        mean, std = float(X.mean()), float(X.std()) or 1.0
        self._norm = (mean, std)
        torch.manual_seed(self.config.seed)
        net = _build_unet(torch, X.shape[1], self.grid)
        xt = torch.from_numpy((X - mean) / std)
        yt = torch.from_numpy(Y).unsqueeze(1)       # (n, 1, G, G)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        bce = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.config.seed)
        net.train()
        for ep in range(epochs):
            tot = 0.0; nb = 0
            for idx in torch.randperm(xt.shape[0], generator=gen).split(batch_size):
                opt.zero_grad()
                logits = net(xt[idx])
                p = torch.sigmoid(logits)
                # soft Dice handles the mostly-empty grid; BCE keeps per-cell calibration
                inter = (p * yt[idx]).sum(dim=(1, 2, 3))
                dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + yt[idx].sum(dim=(1, 2, 3)) + 1)
                loss = bce(logits, yt[idx]) + dice.mean()
                loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if report is not None:
                report(ep + 1, {"loss": tot / max(nb, 1)})
        net.eval(); self._net = net
        return self

    def predict_heatmap(self, X) -> np.ndarray:
        """(n, C, K, W) or one (C, K, W) -> (n, G, G) per-cell probability."""
        torch = _torch()
        X = self._as_images(np.asarray(X, dtype=np.float32))
        Xn = (X - self._norm[0]) / self._norm[1]
        with torch.no_grad():
            p = torch.sigmoid(self._net(torch.from_numpy(Xn)))  # (n, 1, G, G)
        return p.squeeze(1).numpy()

    def predict_exists(self, X, thr=0.5) -> np.ndarray:
        """Binary 'is target present' from the heatmap (max cell > thr)."""
        hm = self.predict_heatmap(X)
        return (hm.reshape(hm.shape[0], -1).max(axis=1) > thr).astype(int)

    def _as_images(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 4:
            return np.ascontiguousarray(X)
        if X.ndim == 3:
            return np.ascontiguousarray(X[:, np.newaxis, :, :])
        shape = self._image_shape or (1, self.config.k, self.config.window)
        return np.ascontiguousarray(X.reshape(X.shape[0], *shape))

    def save(self, path) -> Path:
        torch = _torch()
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "config": asdict(self.config),
            "grid": self.grid,
            "state": {k: v.cpu().numpy() for k, v in self._net.state_dict().items()},
            "norm": self._norm,
            "image_shape": tuple(self._image_shape),
        }, p)
        return p

    @classmethod
    def load(cls, path) -> "HeatmapHead":
        torch = _torch()
        blob = joblib.load(path)
        head = cls(ModelConfig(**blob["config"]), grid=blob["grid"])
        head._norm = blob["norm"]
        head._image_shape = tuple(blob["image_shape"])
        net = _build_unet(torch, head._image_shape[0], head.grid)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in blob["state"].items()})
        net.eval(); head._net = net
        return head
