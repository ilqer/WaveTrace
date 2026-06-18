"""Phase 6d/7 — the real-time inference path: load a persisted head, classify one window, <8 ms.

TWO OPERATING MODES (user decision 2026-06-11 — REPLACES the earlier A→E two-stage gate; the
application runs ONE mode at a time, selected by the operator, with NO cross-gating):
  * "presence" — human detection: PresenceHead on the gain-locked 9·K feature vector.
  * "weapon"   — weapon detection: WeaponHead on its backend's input (raw inter-carrier block for
    variance/mlp/svm, CSI image for cnn) — classifies EVERY emitted window, independent of any
    presence verdict. Soft voting (Vote.SegmentVoter) still applies WITHIN this mode over a
    PresenceSegmenter-bounded active segment (that is a DSP activity gate, not the presence model).
Use `mode_session(mode, path)` to get the right serving session.

Forward pass is O(1) (fixed-length feature vector through a tiny MLP/SVM). The (1, d) input row is
reused across calls (no per-window wrapper allocation on our side; sklearn allocates internally —
acceptable Python glue, the future C++/numpy-tiny path removes it).
"""

import time

import numpy as np

from wavetrace.recognition.Model import PresenceHead
from wavetrace.recognition.Weapon import WeaponHead


class InferenceSession:
    """Serve one persisted head on the real-time path.

    Default loader = PresenceHead.load; pass loader=WeaponHead.load for a Stage-E session (P7), or
    head=<fitted head> to wrap an in-memory model. CNN weapon heads accept the flattened window
    image here (they reshape internally)."""

    def __init__(self, model_path=None, *, head=None, loader=None):
        if head is not None:
            self._head = head
        else:
            self._head = (loader or PresenceHead.load)(model_path)
        self._row = None  # reused (1, d) input row

    @property
    def head(self) -> PresenceHead:
        return self._head

    def predict_proba_window(self, feature_vector) -> np.ndarray:
        """(d,) -> (C,) class probabilities; reuses the same row buffer as predict_window. O(1)."""
        v = np.asarray(feature_vector, dtype=np.float32).ravel()
        if self._row is None or self._row.shape[1] != v.size:
            self._row = np.empty((1, v.size), dtype=np.float32)
        self._row[0, :] = v
        return self._head.predict_proba(self._row)[0]

    def predict_window(self, feature_vector) -> tuple[int, float]:
        """One emitted window's feature vector (d,) -> (class_id, probability). O(1)."""
        proba = self.predict_proba_window(feature_vector)
        i = int(np.argmax(proba))
        return int(self._head.classes_[i]), float(proba[i])


def mode_session(mode: str, model_path) -> InferenceSession:
    """The application's mode switch: 'presence' (human detection) or 'weapon' (weapon detection).
    Modes are independent — each loads its own model and consumes its own feature contract. O(1)."""
    if mode == "presence":
        loader = PresenceHead.load
    elif mode == "weapon":
        loader = WeaponHead.load
    else:
        raise ValueError(f"mode must be 'presence' or 'weapon', got {mode!r}")
    return InferenceSession(model_path, loader=loader)


def measure_latency(session: InferenceSession, feature_vector, iters: int = 200) -> dict:
    """Per-call predict_window latency over `iters` calls (after a small warmup so one-time sklearn
    setup doesn't pollute the gate). Returns mean/p95/max in ms — the Phase-6 DoD asserts max < 8."""
    for _ in range(5):
        session.predict_window(feature_vector)
    samples = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        session.predict_window(feature_vector)
        samples[i] = time.perf_counter() - t0
    return {
        "mean_ms": float(samples.mean() * 1e3),
        "p95_ms": float(np.percentile(samples, 95) * 1e3),
        "max_ms": float(samples.max() * 1e3),
        "iters": iters,
    }
