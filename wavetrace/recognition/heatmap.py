"""Camera-supervised G×G occupancy heatmap head.

Input  : X_image (n, C, K, window), C = nodes.
Target : Y (n, G, G) occupancy in [0,1] from Label.mask.
Output : (n, G, G) per-cell probability.
Loss   : BCE + soft Dice (Dice handles the mostly-empty grid; BCE keeps per-cell calibration).
AdaptiveAvgPool makes the net (K, W)-agnostic, so training and serving window sizes may differ.
"""

import math
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from wavetrace.adapters.recognition.heads import check_schema_version
from wavetrace.adapters.store.model_runs import find_model_dir, resolve_model_path
from wavetrace.domain.contracts import SCHEMA_VERSION
from wavetrace.domain.model_config import ModelConfig


def _torch():
    try:
        import torch
        return torch
    except ImportError as error:
        raise ImportError("HeatmapHead needs PyTorch: pip install 'wavetrace[cnn]'") from error


def _build_heatmap_net(torch, in_channels: int, grid: int):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((grid, grid)),
        nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 1, 1),
    )


class HeatmapHead:
    def __init__(self, config: ModelConfig, *, grid: int = 16):
        self.config = config
        self.grid = int(grid)
        self.feature_mode = "heatmap"
        self._net = None
        self._norm: tuple[float, float] | None = None
        self._image_shape: tuple | None = None     # (C, K, W)

    def fit(self, X, Y, *, epochs=40, learning_rate=1e-3, batch_size=16,
            report=None) -> "HeatmapHead":
        """X: (n, C, K, W) images. Y: (n, G, G) or (n, G*G) occupancy in [0,1].
        report: optional callback(epoch, {"loss": float})."""
        torch = _torch()
        X = self._as_images(np.asarray(X, dtype=np.float32))
        self._image_shape = X.shape[1:]
        Y = np.asarray(Y, dtype=np.float32).reshape(X.shape[0], self.grid, self.grid)
        mean, standard_deviation = float(X.mean()), float(X.std()) or 1.0
        self._norm = (mean, standard_deviation)
        torch.manual_seed(self.config.seed)
        network = _build_heatmap_net(torch, X.shape[1], self.grid)
        x_tensor = torch.from_numpy((X - mean) / standard_deviation)
        y_tensor = torch.from_numpy(Y).unsqueeze(1)       # (n, 1, G, G)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        bce_loss = torch.nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.config.seed)
        network.train()
        for epoch in range(epochs):
            loss_sum = 0.0; batch_count = 0
            for batch_indices in torch.randperm(x_tensor.shape[0], generator=generator).split(batch_size):
                optimizer.zero_grad()
                logits = network(x_tensor[batch_indices])
                probabilities = torch.sigmoid(logits)
                intersection = (probabilities * y_tensor[batch_indices]).sum(dim=(1, 2, 3))
                dice = 1 - (2 * intersection + 1) / (
                    probabilities.sum(dim=(1, 2, 3)) + y_tensor[batch_indices].sum(dim=(1, 2, 3)) + 1)
                loss = bce_loss(logits, y_tensor[batch_indices]) + dice.mean()
                loss.backward(); optimizer.step()
                loss_sum += float(loss); batch_count += 1
            if report is not None:
                report(epoch + 1, {"loss": loss_sum / max(batch_count, 1)})
        network.eval(); self._net = network
        return self

    def predict_heatmap(self, X) -> np.ndarray:
        """(n, C, K, W) or one (C, K, W) -> (n, G, G) per-cell probability."""
        torch = _torch()
        X = self._as_images(np.asarray(X, dtype=np.float32))
        X_normalized = (X - self._norm[0]) / self._norm[1]
        with torch.no_grad():
            probabilities = torch.sigmoid(self._net(torch.from_numpy(X_normalized)))  # (n, 1, G, G)
        return probabilities.squeeze(1).numpy()

    def predict_exists(self, X, threshold=0.5) -> np.ndarray:
        heatmap = self.predict_heatmap(X)
        return (heatmap.reshape(heatmap.shape[0], -1).max(axis=1) > threshold).astype(int)

    def _as_images(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 4:
            return np.ascontiguousarray(X)
        if X.ndim == 3:
            return np.ascontiguousarray(X[:, np.newaxis, :, :])
        shape = self._image_shape or (1, self.config.k, self.config.window)
        return np.ascontiguousarray(X.reshape(X.shape[0], *shape))

    def save(self, path) -> Path:
        _torch()
        output_path = Path(path); output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "config": asdict(self.config),
            "grid": self.grid,
            "state": {name: weights.cpu().numpy() for name, weights in self._net.state_dict().items()},
            "norm": self._norm,
            "image_shape": tuple(self._image_shape),
            "schema_version": SCHEMA_VERSION,
        }, output_path)
        return output_path

    @classmethod
    def load(cls, path) -> "HeatmapHead":
        """`path` may name `<model_root>/heatmap.joblib` flat; it loads the current run's copy. Raises
        SchemaVersionMismatchError for another schema_version."""
        torch = _torch()
        path = resolve_model_path(path)
        blob = joblib.load(path)
        check_schema_version(blob, path, f"train the heatmap backend on a camera session dir "
                                         f"(<root>/cam_ds/heatmap/<session>) into {find_model_dir(path.parent)}")
        head = cls(ModelConfig(**blob["config"]), grid=blob["grid"])
        head._norm = blob["norm"]
        head._image_shape = tuple(blob["image_shape"])
        network = _build_heatmap_net(torch, head._image_shape[0], head.grid)
        network.load_state_dict({name: torch.from_numpy(weights) for name, weights in blob["state"].items()})
        network.eval(); head._net = network
        return head


def train_heatmap(dataset, *, report=None) -> tuple[HeatmapHead, int]:
    """Fits on the masked windows only, each paired with its own mask; returns the head and the
    masked-window count. O(n) selection plus training. Raises ValueError when no window has a mask."""
    masked_indices = [i for i, label in enumerate(dataset.labels) if label.mask]
    if not masked_indices:
        raise ValueError("dataset has no Label.mask; collect with SegmentationLabeler / YoloSegLabeler")
    masks = np.asarray([dataset.labels[i].mask for i in masked_indices], dtype=np.float32)
    grid = math.isqrt(masks.shape[1])
    config = ModelConfig(stage="presence", k=int(dataset.meta["K"]))
    head = HeatmapHead(config, grid=grid).fit(dataset.X_image[masked_indices], masks, report=report)
    return head, len(masked_indices)
