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
    `.timestamp` (CameraLabeler.label_stream does this); window_timestamps ascending. O(W+L)."""
    label_timestamps = [label.timestamp for label in labels]
    num_labels = len(label_timestamps)
    matched: list[tuple[int, Label]] = []
    collected_time_deltas_s: list[float] = []
    dropped: list[int] = []

    j = 0  # monotonically advances across the sorted windows -> O(W+L) overall
    for window_index, window_timestamp in enumerate(window_timestamps):
        if num_labels == 0:
            dropped.append(window_index)
            continue
        while (j + 1 < num_labels
               and abs(label_timestamps[j + 1] - window_timestamp)
               <= abs(label_timestamps[j] - window_timestamp)):
            j += 1
        time_delta_s = label_timestamps[j] - window_timestamp
        if abs(time_delta_s) <= tolerance:
            matched.append((window_index, labels[j]))
            collected_time_deltas_s.append(time_delta_s)
        else:
            dropped.append(window_index)

    time_deltas_s = np.asarray(collected_time_deltas_s, dtype=np.float64)
    stats = {
        "mean_dt": float(time_deltas_s.mean()) if time_deltas_s.size else 0.0,
        "max_abs_dt": float(np.abs(time_deltas_s).max()) if time_deltas_s.size else 0.0,
        "p95_abs_dt": float(np.percentile(np.abs(time_deltas_s), 95)) if time_deltas_s.size else 0.0,
        "matched": len(matched),
        "dropped": len(dropped),
    }
    return AlignmentResult(matched=matched, dts=time_deltas_s, dropped=dropped, stats=stats)


def estimate_clock_offset(truth_times, truth_classes, labels, *, max_lag=0.2, step=0.005):
    """Recover a CONSTANT clock offset of `labels` relative to a KNOWN-truth class sequence
    (truth_times/truth_classes on the CSI clock) by the lag that maximizes class agreement.

    Why this and not `align`'s Δt: nearest-timestamp matching always minimizes |Δt|, so a dense label
    stream's matched Δt stays within half a label period whatever the constant offset is. The offset
    is invisible in Δt and corrupts label CONTENT instead. Measuring it honestly therefore needs a
    content cross-correlation over a staged calibration sequence: a label recorded at
    `label.timestamp` corresponds to CSI time `label.timestamp − offset`, so testing candidate `lag_s`
    matches truth sample `truth_timestamp` against the nearest label at `truth_timestamp + lag_s`; the
    `lag_s` with best agreement is the offset. Returns (offset_s, agreement∈[0,1]). O(n_lags·(T+L)).
    Apply −offset before `align`."""
    label_timestamps = [label.timestamp for label in labels]
    label_class_ids = [label.class_id for label in labels]
    num_labels = len(label_timestamps)
    truth_timestamps = list(truth_times)
    truth_class_ids = list(truth_classes)
    lags = np.arange(-max_lag, max_lag + 1e-9, step)
    agrees = np.empty(lags.size)
    for lag_index, lag_s in enumerate(lags):
        agreement_count = 0
        # target = truth_timestamp + lag_s is monotonic in truth_timestamp -> single forward sweep
        # per candidate
        j = 0
        for truth_timestamp, truth_class_id in zip(truth_timestamps, truth_class_ids):
            if num_labels == 0:
                break
            target = truth_timestamp + lag_s
            while (j + 1 < num_labels
                   and abs(label_timestamps[j + 1] - target) <= abs(label_timestamps[j] - target)):
                j += 1
            if label_class_ids[j] == truth_class_id:
                agreement_count += 1
        agrees[lag_index] = agreement_count / len(truth_timestamps) if truth_timestamps else 0.0
    best = agrees.max() if agrees.size else 0.0
    # discrete labels give a perfect-agreement PLATEAU ~one label period wide; its midpoint is the
    # offset estimate (first-argmax would bias to the plateau edge).
    plateau = lags[agrees >= best - 1e-9]
    return float(plateau.mean()), float(best)
