"""Match camera labels to CSI feature windows, and measure how well the two clocks agree.

Each window (timestamped at its END) takes the nearest label within `tolerance` seconds; a window
with no label that close is dropped. Two pointers over time-sorted inputs, O(W+L).

The spread of matched Δt is the sync-error measurement. Δt is label.timestamp − window_t, so its
mean is roughly how far the label stream sits from the CSI clock, and a small constant offset
there means silently wrong labels rather than obviously missing ones.
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
    `.timestamp` (CameraLabeler.labelStream does this); window_timestamps ascending. O(W+L)."""
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

    dtArr = np.asarray(dts, dtype=np.float64)
    stats = {
        "mean_dt": float(dtArr.mean()) if dtArr.size else 0.0,
        "max_abs_dt": float(np.abs(dtArr).max()) if dtArr.size else 0.0,
        "p95_abs_dt": float(np.percentile(np.abs(dtArr), 95)) if dtArr.size else 0.0,
        "matched": len(matched),
        "dropped": len(dropped),
    }
    return AlignmentResult(matched=matched, dts=dtArr, dropped=dropped, stats=stats)


def estimateClockOffset(truth_times, truth_classes, labels, *, max_lag=0.2, step=0.005):
    """Recover a CONSTANT clock offset of `labels` relative to a KNOWN-truth class sequence
    (truth_times/truth_classes on the CSI clock) by the lag that maximizes class agreement.

    Why this and not `align`'s Δt: nearest-timestamp matching always minimizes |Δt|, so a dense label
    stream's matched Δt stays within half a label period whatever the constant offset is. The offset
    is invisible in Δt and corrupts label CONTENT instead. Measuring it honestly therefore needs a
    content cross-correlation over a staged calibration sequence: a
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
