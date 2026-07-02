"""Real-time inference path: load head, classify window, <8ms.

Modes:
* 'presence': human detection.
* 'weapon': weapon detection. Classifies every emitted window.

Forward pass is O(1). Input row buffer is reused."""

import time

import numpy as np

from wavetrace.recognition.Model import PresenceHead
from wavetrace.recognition.Weapon import WeaponHead


class InferenceSession:
    """Serve a persisted head."""

    def __init__(self, model_path=None, *, head=None, loader=None):
        if head is not None:
            self._head = head
        else:
            self._head = (loader or PresenceHead.load)(model_path)
        self._row = None  # reused (1, d) input row

    @property
    def head(self) -> PresenceHead:
        return self._head

    def predictProbaWindow(self, feature_vector) -> np.ndarray:
        """Predict class probabilities. Reuses row buffer. O(1)."""
        v = np.asarray(feature_vector, dtype=np.float32).ravel()
        if self._row is None or self._row.shape[1] != v.size:
            self._row = np.empty((1, v.size), dtype=np.float32)
        self._row[0, :] = v
        return self._head.predict_proba(self._row)[0]

    def predictWindow(self, feature_vector) -> tuple[int, float]:
        """One emitted window's feature vector (d,) -> (class_id, probability). O(1)."""
        proba = self.predictProbaWindow(feature_vector)
        i = int(np.argmax(proba))
        return int(self._head.classes_[i]), float(proba[i])


def modeSession(mode: str, model_path) -> InferenceSession:
    """Mode switch: 'presence' or 'weapon'. O(1)."""
    if mode == "presence":
        loader = PresenceHead.load
    elif mode == "weapon":
        loader = WeaponHead.load
    else:
        raise ValueError(f"mode must be 'presence' or 'weapon', got {mode!r}")
    return InferenceSession(model_path, loader=loader)


def measureLatency(session: InferenceSession, feature_vector, iters: int = 200) -> dict:
    """Measure inference latency over iters calls. Returns ms stats."""
    for _ in range(5):
        session.predictWindow(feature_vector)
    samples = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        session.predictWindow(feature_vector)
        samples[i] = time.perf_counter() - t0
    return {
        "mean_ms": float(samples.mean() * 1e3),
        "p95_ms": float(np.percentile(samples, 95) * 1e3),
        "max_ms": float(samples.max() * 1e3),
        "iters": iters,
    }
