"""The real-time path: load a head, classify a window, publish. Under 8 ms.

'presence' and 'weapon' both classify every emitted window. The forward pass is O(1) and the input
row buffer is reused between windows."""

import time

import numpy as np

from wavetrace.adapters.recognition.heads import load_presence_head, load_weapon_head
from wavetrace.domain.recognition import PresenceHead


class InferenceSession:
    """Serve a persisted head."""

    def __init__(self, model_path=None, *, head=None, loader=None):
        if head is not None:
            self._head = head
        else:
            self._head = (loader or load_presence_head)(model_path)
        self._row = None  # reused (1, d) input row

    @property
    def head(self) -> PresenceHead:
        return self._head

    def predictProbaWindow(self, feature_vector) -> np.ndarray:
        """Predict class probabilities. Reuses row buffer. O(1)."""
        flattened = np.asarray(feature_vector, dtype=np.float32).ravel()
        if self._row is None or self._row.shape[1] != flattened.size:
            self._row = np.empty((1, flattened.size), dtype=np.float32)
        self._row[0, :] = flattened
        return self._head.predict_proba(self._row)[0]

    def predictWindow(self, feature_vector) -> tuple[int, float]:
        """One emitted window's feature vector (d,) -> (class_id, probability). O(1)."""
        proba = self.predictProbaWindow(feature_vector)
        i = int(np.argmax(proba))
        return int(self._head.classes_[i]), float(proba[i])


def planInferenceInput(mode: str, head) -> tuple[bool, bool, object]:
    """Return (apply_lock, intercarrier, pick) for the serving loop.
    `pick(features, image, intercarrier) -> x` is the row fed to `InferenceSession.predictWindow`.
    A presence head always takes the plain feature vector; a weapon head is self-describing via
    `head.feature_mode`, falling back on `head.default_feature_mode` for models saved before that
    was recorded."""
    if mode == "presence":
        return True, False, (lambda features, image, intercarrier: features)
    feature_mode = getattr(head, "feature_mode", None) or head.default_feature_mode
    if feature_mode == "cnn":
        return False, False, (lambda features, image, intercarrier: image.reshape(-1))
    if feature_mode == "fusion":
        return True, True, (lambda features, image, intercarrier: np.hstack([intercarrier, features]))
    return False, True, (lambda features, image, intercarrier: intercarrier)  # ic27 / variance


def modeSession(mode: str, model_path) -> InferenceSession:
    """Mode switch: 'presence' or 'weapon'. O(1)."""
    if mode == "presence":
        loader = load_presence_head
    elif mode == "weapon":
        loader = load_weapon_head
    else:
        raise ValueError(f"mode must be 'presence' or 'weapon', got {mode!r}")
    return InferenceSession(model_path, loader=loader)


def measureLatency(session: InferenceSession, feature_vector, iters: int = 200) -> dict:
    """Measure inference latency over iters calls. Returns ms stats."""
    for _ in range(5):
        session.predictWindow(feature_vector)
    samples = np.empty(iters)
    for i in range(iters):
        start_time = time.perf_counter()
        session.predictWindow(feature_vector)
        samples[i] = time.perf_counter() - start_time
    return {
        "mean_ms": float(samples.mean() * 1e3),
        "p95_ms": float(np.percentile(samples, 95) * 1e3),
        "max_ms": float(samples.max() * 1e3),
        "iters": iters,
    }
