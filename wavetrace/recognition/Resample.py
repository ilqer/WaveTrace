"""Phase 6f — timing-jitter guards for the front-end glue (plan §5 Phase 6).

Real captures arrive with irregular inter-frame spacing (2.4 GHz congestion, queueing): three guards
make detection robust to residual capture irregularity before windows reach a head:
  * resample_uniform — linear interp of each series onto a uniform grid (the §2.9 features and the
    FFT-based Doppler/PSD assume uniform sampling).
  * fs_ok — drop a window whose LIVE estimated fs deviates from nominal beyond tolerance (fs is
    always estimated from timestamps, never assumed — REFERENCE §4); resampling can't fix a window
    that is mostly gaps.
  * accept_format — ingest format filter: the dedicated controlled link emits exactly ONE packet
    format; a stray legacy frame (e.g. 128 B vs 384 B) would silently mis-parse, so reject any other
    length at ingest (plan §2 "ingest format filter").

Per-emit O(n)/window, not per-frame. NOTE (user): Python glue for now — move to C++/a faster
library on the real-time path later.
"""

import numpy as np


def resample_uniform(values, timestamps, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample irregularly-timed samples onto a uniform target_fs grid (np.interp per series).

    values (n,) or (n, k) float; timestamps (n,) strictly increasing, same clock as target grid.
    Returns (resampled (m,)|(m, k) float32, grid (m,) float64) with m = floor(span·fs) + 1,
    grid[0] = timestamps[0]. O(n·k)."""
    t = np.asarray(timestamps, dtype=np.float64)
    v = np.asarray(values, dtype=np.float32)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("resample_uniform: need >= 2 timestamps")
    if v.shape[0] != t.size:
        raise ValueError(f"resample_uniform: {v.shape[0]} values vs {t.size} timestamps")
    if np.any(np.diff(t) <= 0):
        raise ValueError("resample_uniform: timestamps must be strictly increasing")
    if target_fs <= 0:
        raise ValueError("resample_uniform: target_fs must be positive")
    m = int(np.floor((t[-1] - t[0]) * target_fs)) + 1
    grid = t[0] + np.arange(m) / target_fs
    if v.ndim == 1:
        out = np.interp(grid, t, v)
    else:
        out = np.empty((m, v.shape[1]), dtype=np.float64)
        for j in range(v.shape[1]):  # np.interp is 1-D; k is small (NBVI K ~ 12)
            out[:, j] = np.interp(grid, t, v[:, j])
    return out.astype(np.float32), grid


def fs_ok(timestamps, nominal_fs: float, tol: float) -> bool:
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


def accept_format(frame_len: int, expected_len: int) -> bool:
    """Ingest format filter: accept only the one controlled-link packet length. O(1)."""
    return expected_len > 0 and int(frame_len) == int(expected_len)
