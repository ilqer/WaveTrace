"""Phase 5b — timestamp-align camera labels to CSI feature windows + MEASURE sync error (OFFLINE).

Each CSI window (timestamp = window END, Q5) is matched to the NEAREST label within `tolerance`
seconds; windows with no label in tolerance are dropped. Two-pointer O(W+L) over time-sorted inputs.

The matched-Δt distribution IS the sync-error measurement (Phase-5 DoD). With discrete camera frames
a constant clock offset between the two streams shows up as the systematic component of mean Δt, so
this also DETECTS/measures a clock skew (REFERENCE_DIGEST §0B: a small offset = silently wrong
labels). Δt is defined label.timestamp − window_t, so mean Δt ≈ the label stream's offset relative to
the CSI clock.
"""

from dataclasses import dataclass

import numpy as np

from wavetrace import Label


@dataclass
class AlignmentResult:
    matched: list[tuple[int, Label]]  # (window index, nearest Label) within tolerance
    dts: np.ndarray                   # matched Δt = label_ts − window_ts, shape (len(matched),)
    dropped: list[int]                # window indices with no label in tolerance
    stats: dict                       # mean_dt, max_abs_dt, p95_abs_dt, matched, dropped


def align(window_timestamps, labels, tolerance: float) -> AlignmentResult:
    """Nearest-label-within-tolerance match for each window timestamp. `labels` must be sorted by
    `.timestamp` (CameraLabeler.label_stream does this); window_timestamps ascending. O(W+L)."""
    lt = [l.timestamp for l in labels]
    n = len(lt)
    matched: list[tuple[int, Label]] = []
    dts: list[float] = []
    dropped: list[int] = []

    j = 0  # monotonically advances across the sorted windows -> O(W+L) overall
    for wi, w in enumerate(window_timestamps):
        if n == 0:
            dropped.append(wi)
            continue
        while j + 1 < n and abs(lt[j + 1] - w) <= abs(lt[j] - w):
            j += 1
        dt = lt[j] - w
        if abs(dt) <= tolerance:
            matched.append((wi, labels[j]))
            dts.append(dt)
        else:
            dropped.append(wi)

    dt_arr = np.asarray(dts, dtype=np.float64)
    stats = {
        "mean_dt": float(dt_arr.mean()) if dt_arr.size else 0.0,
        "max_abs_dt": float(np.abs(dt_arr).max()) if dt_arr.size else 0.0,
        "p95_abs_dt": float(np.percentile(np.abs(dt_arr), 95)) if dt_arr.size else 0.0,
        "matched": len(matched),
        "dropped": len(dropped),
    }
    return AlignmentResult(matched=matched, dts=dt_arr, dropped=dropped, stats=stats)


def estimate_clock_offset(truth_times, truth_classes, labels, *, max_lag=0.2, step=0.005):
    """Recover a CONSTANT clock offset of `labels` relative to a KNOWN-truth class sequence
    (truth_times/truth_classes on the CSI clock) by the lag that maximizes class agreement.

    Why this and not `align`'s Δt: nearest-timestamp matching always minimizes |Δt|, so a dense label
    stream's matched Δt stays within ½ a label period regardless of a constant offset — the offset is
    invisible in Δt and instead silently corrupts label CONTENT (REFERENCE_DIGEST §0B). The honest
    skew measurement is therefore a CONTENT cross-correlation on a staged calibration sequence: a
    label recorded at time `lt` corresponds to CSI time `lt − offset`, so testing candidate `off`
    matches truth sample `tt` against the nearest label at `tt + off`; the `off` with best agreement is
    the offset. Returns (offset_s, agreement∈[0,1]). O(n_lags·(T+L)). Apply −offset before `align`."""
    lt = [l.timestamp for l in labels]
    lc = [l.class_id for l in labels]
    n = len(lt)
    tt = list(truth_times)
    tc = list(truth_classes)
    lags = np.arange(-max_lag, max_lag + 1e-9, step)
    agrees = np.empty(lags.size)
    for li, off in enumerate(lags):
        agree = 0
        j = 0  # target = tt + off is monotonic in tt -> single forward sweep per candidate
        for ti, ci in zip(tt, tc):
            if n == 0:
                break
            target = ti + off
            while j + 1 < n and abs(lt[j + 1] - target) <= abs(lt[j] - target):
                j += 1
            if lc[j] == ci:
                agree += 1
        agrees[li] = agree / len(tt) if tt else 0.0
    best = agrees.max() if agrees.size else 0.0
    # discrete labels give a perfect-agreement PLATEAU ~one label period wide; its midpoint is the
    # offset estimate (first-argmax would bias to the plateau edge).
    plateau = lags[agrees >= best - 1e-9]
    return float(plateau.mean()), float(best)
