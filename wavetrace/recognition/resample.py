"""Three guards against irregular frame timing, applied before a window reaches a head.

2.4 GHz congestion and queueing make the inter-frame spacing uneven:
  * resampleUniform - interpolate each series onto a uniform grid, which the features and the
    FFT-based Doppler/PSD both assume.
  * fsOk - drop a window whose live fs strays too far from nominal. fs is always measured from the
    timestamps, and resampling cannot rescue a window that is mostly gaps.
  * acceptFormat - the controlled link emits exactly one packet format, and a stray legacy frame
    (128 B against 384 B) would mis-parse silently, so any other length is rejected at ingest.

O(n) per emitted window, not per frame.
"""

import numpy as np


def resampleUniform(values, timestamps, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample irregularly-timed samples onto a uniform target_fs grid (np.interp per series).

    values (n,) or (n, k) float; timestamps (n,) strictly increasing, same clock as target grid.
    Returns (resampled (m,)|(m, k) float32, grid (m,) float64) with m = floor(span·fs) + 1,
    grid[0] = timestamps[0]. O(n·k)."""
    t = np.asarray(timestamps, dtype=np.float64)
    v = np.asarray(values, dtype=np.float32)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("resampleUniform: need >= 2 timestamps")
    if v.shape[0] != t.size:
        raise ValueError(f"resampleUniform: {v.shape[0]} values vs {t.size} timestamps")
    if np.any(np.diff(t) <= 0):
        raise ValueError("resampleUniform: timestamps must be strictly increasing")
    if target_fs <= 0:
        raise ValueError("resampleUniform: target_fs must be positive")
    m = int(np.floor((t[-1] - t[0]) * target_fs)) + 1
    grid = t[0] + np.arange(m) / target_fs
    if v.ndim == 1:
        out = np.interp(grid, t, v)
    else:
        out = np.empty((m, v.shape[1]), dtype=np.float64)
        for j in range(v.shape[1]):  # np.interp is 1-D; k is small (NBVI K ~ 12)
            out[:, j] = np.interp(grid, t, v[:, j])
    return out.astype(np.float32), grid


def fsOk(timestamps, nominal_fs: float, tol: float) -> bool:
    """True iff the live fs estimated from the window's timestamps is within ±tol (relative) of
    nominal. Live fs = (n-1)/span — the same estimator the dataset meta uses. O(1)."""
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or nominal_fs <= 0 or tol <= 0:
        return False
    span = float(t[-1] - t[0])
    if span <= 0:
        return False
    live = (t.size - 1) / span
    return abs(live - nominal_fs) / nominal_fs <= tol


def acceptFormat(frame_len: int, expected_len: int) -> bool:
    """Ingest format filter: accept only the one controlled-link packet length. O(1)."""
    return expected_len > 0 and int(frame_len) == int(expected_len)
