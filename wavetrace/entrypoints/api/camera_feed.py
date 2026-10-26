import threading
from collections.abc import Callable


def _load_ultralytics_yolo(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


class CameraFeed:
    """The dashboard's live camera: `is_streaming` keeps the MJPEG stream running until /api/camera/stop
    clears it, and each YOLO weights file is loaded at most once. `load_model` is called on worker threads,
    so the cache is locked."""

    def __init__(self, load_model: Callable[[str], object] = _load_ultralytics_yolo):
        self.is_streaming = False
        self._load_model = load_model
        self._model_by_weights: dict[str, object | None] = {}
        self._model_lock = threading.Lock()

    def load_model(self, weights: str):
        """The model for `weights`, or None when it failed to load; a failure is cached, not retried per frame."""
        with self._model_lock:
            if weights not in self._model_by_weights:
                try:
                    self._model_by_weights[weights] = self._load_model(weights)
                except Exception as error:
                    self._model_by_weights[weights] = None
                    print(f"[YOLO] load failed: {error}")
            return self._model_by_weights[weights]
