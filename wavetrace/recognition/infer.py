"""Per-window latency stats (the serving budget is 8 ms)."""

import time

import numpy as np

from wavetrace.adapters.recognition.session import InferenceSession
from wavetrace.domain.recognition.verdict import decide_class


def measure_latency(session: InferenceSession, feature_vector, iters: int = 200) -> dict:
    """Latency stats in ms over `iters` timed calls, after 5 warm-up calls."""
    classes = session.head.classes_
    for _ in range(5):
        decide_class(session.predict_window_proba(feature_vector), classes)
    samples = np.empty(iters)
    for i in range(iters):
        start_time = time.perf_counter()
        decide_class(session.predict_window_proba(feature_vector), classes)
        samples[i] = time.perf_counter() - start_time
    return {
        "mean_ms": float(samples.mean() * 1e3),
        "p95_ms": float(np.percentile(samples, 95) * 1e3),
        "max_ms": float(samples.max() * 1e3),
        "iters": iters,
    }
