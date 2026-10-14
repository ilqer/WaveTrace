"""The image-based recognition backend: a small 2D-CNN over the CSI image (K subcarriers x window
frames, one channel per RX node). Torch is lazy-imported (`_torch`) so importing this module — and
anything that imports the registry — never requires PyTorch as a hard dependency.
"""

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError as error:  # pragma: no cover - exercised only without the [cnn] extra
        raise ImportError(
            "WeaponHead backend 'cnn' requires PyTorch: pip install 'wavetrace[cnn]'"
        ) from error


def _build_network(torch, hidden: int, num_classes: int, in_channels: int = 1):
    """AdaptiveAvgPool makes the network (K, window)-agnostic."""
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(), nn.Linear(16 * 16, hidden), nn.ReLU(), nn.Linear(hidden, num_classes),
    )


class CnnBackend:
    feature_mode = "cnn"

    def __init__(self, config, **backend_options):
        self.config = config
        self._net = None
        self._normalization_stats = None  # (mean, standard deviation) train normalization
        self._image_shape = None   # (C, K, window) for reshaping flattened windows
        self._classes = None

    @property
    def classes_(self) -> np.ndarray:
        return self._classes

    def _as_images(self, X) -> np.ndarray:
        """Accepts 2-D (flattened windows), 3-D (n,K,W) or 4-D input; returns (n,C,K,W)."""
        if X.ndim == 4:
            return np.ascontiguousarray(X)
        if X.ndim == 3:
            return np.ascontiguousarray(X[:, np.newaxis, :, :])  # (n, 1, K, W)
        shape = self._image_shape or (1, self.config.k, self.config.window)
        return np.ascontiguousarray(X.reshape(X.shape[0], *shape))

    def fit(self, X, y, *, epochs: int = 30, learning_rate: float = 1e-3, batch_size: int = 32,
            report=None, **_ignored) -> None:
        torch = _torch()
        images = self._as_images(np.asarray(X, dtype=np.float32))  # (n, C, K, W)
        self._image_shape = images.shape[1:]                       # (C, K, W)
        in_channels = images.shape[1]
        self._classes = np.unique(y)
        class_indices = np.searchsorted(self._classes, y)
        mean, standard_deviation = float(images.mean()), float(images.std()) or 1.0
        self._normalization_stats = (mean, standard_deviation)
        torch.manual_seed(self.config.seed)
        network = _build_network(torch, self.config.hidden, int(self._classes.size), in_channels=in_channels)
        x_tensor = torch.from_numpy((images - mean) / standard_deviation)
        y_tensor = torch.from_numpy(class_indices.astype(np.int64))
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self.config.seed)
        network.train()
        sample_count = x_tensor.shape[0]
        for epoch in range(epochs):
            batch_losses = []
            for batch_indices in torch.randperm(sample_count, generator=generator).split(batch_size):
                optimizer.zero_grad()
                loss = loss_function(network(x_tensor[batch_indices]), y_tensor[batch_indices])
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_loss_mean = float(np.mean(batch_losses)) if batch_losses else 0.0
            print(f"      cnn ep {epoch+1:3d}/{epochs}  loss={epoch_loss_mean:.4f}", end="\r", flush=True)
            if report is not None:
                # batch-loss spread = confidence band on the live training curve; accuracy from a cheap
                # eval-mode pass
                epoch_loss_standard_deviation = float(np.std(batch_losses)) if len(batch_losses) > 1 else 0.0
                network.eval()
                with torch.no_grad():
                    accuracy = float((network(x_tensor).argmax(1) == y_tensor).float().mean())
                network.train()
                report(epoch + 1, {"loss": epoch_loss_mean, "loss_std": epoch_loss_standard_deviation,
                                   "acc": accuracy})
        print()
        network.eval()
        self._net = network

    def predict_proba(self, X) -> np.ndarray:
        torch = _torch()
        X = np.asarray(X, dtype=np.float32)
        mean, standard_deviation = self._normalization_stats
        images = (self._as_images(X) - mean) / standard_deviation
        with torch.no_grad():
            logits = self._net(torch.from_numpy(images))
            return torch.softmax(logits, dim=1).numpy()

    def save(self) -> dict:
        return {
            "state": {name: weights.cpu().numpy() for name, weights in self._net.state_dict().items()},
            "norm": self._normalization_stats,
            "image_shape": tuple(self._image_shape),
            "classes": self._classes,
        }

    @classmethod
    def load(cls, blob: dict, config, **backend_options) -> "CnnBackend":
        torch = _torch()
        backend = cls(config)
        backend._classes = blob["classes"]
        backend._normalization_stats = blob["norm"]
        raw_shape = tuple(blob["image_shape"])
        # older blobs store a 2-tuple (K, W); prepend C=1 for the canonical 3-tuple
        backend._image_shape = raw_shape if len(raw_shape) == 3 else (1,) + raw_shape
        in_channels = backend._image_shape[0]
        network = _build_network(torch, config.hidden, int(backend._classes.size), in_channels=in_channels)
        network.load_state_dict({name: torch.from_numpy(weights) for name, weights in blob["state"].items()})
        network.eval()
        backend._net = network
        return backend
