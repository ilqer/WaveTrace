"""Per-node link health + dataset diagnostics. All numpy, all off the hot path.

A `NodeHealthMeter` ingests the same CsiFrame stream the front-end sees (tee it through), tracks a
rolling window per node_id, and emits a health dict per node on demand. The cluster_sync / drift /
separation helpers are one-shot computations for the UI's calibration & training panels.

node_id < 100 => 2.4 GHz (ESP32), node_id >= 100 => 5 GHz (nexmon Pi).
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from wavetrace.domain.contracts import DEFAULT_TARGET_SAMPLE_RATE_HZ


@dataclass(frozen=True, slots=True)
class HealthMeterOptions:
    """Tuning knobs for `NodeHealthMeter`."""

    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ
    min_sample_rate_fraction: float = 0.8
    rolling_window_frames: int = 200

    def __post_init__(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        if not 0.0 < self.min_sample_rate_fraction <= 1.0:
            raise ValueError("min_sample_rate_fraction must be in (0, 1]")
        if self.rolling_window_frames <= 0:
            raise ValueError("rolling_window_frames must be positive")


class NodeHealthMeter:
    """Rolling per-node link health. Feed every CsiFrame via `observe`; read `snapshot()` whenever
    the UI wants an update (e.g. once per second). O(1) per frame, O(nodes·rolling_window_frames)
    per snapshot."""

    def __init__(self, options: HealthMeterOptions = HealthMeterOptions(), *, gain_ref_by_node=None):
        self.target_sample_rate_hz = options.target_sample_rate_hz
        self.min_sample_rate_hz = options.target_sample_rate_hz * options.min_sample_rate_fraction
        self.rolling_window_frames = options.rolling_window_frames
        self._wall_receive_times_s = defaultdict(lambda: deque(maxlen=self.rolling_window_frames))
        self._mean_amplitudes = defaultdict(lambda: deque(maxlen=self.rolling_window_frames))
        # cross-subcarrier coefficient of variation (std/mean) per frame
        self._variation_coefficients = defaultdict(lambda: deque(maxlen=self.rolling_window_frames))
        self._frame_counts = defaultdict(int)
        self._last_timestamps_s = {}
        self._reference_amplitude_by_node = dict(gain_ref_by_node or {})
        self._subcarriers_by_node = {}

    def set_reference(self, node_id, ref_scale=None, subcarriers=None):
        """Record a node's calibration reference amplitude + NBVI subcarriers for display."""
        if ref_scale is not None:
            self._reference_amplitude_by_node[int(node_id)] = float(ref_scale)
        if subcarriers is not None:
            self._subcarriers_by_node[int(node_id)] = list(subcarriers)

    def observe(self, frame) -> None:
        """One CsiFrame. O(A·S)."""
        node_id = int(frame.node_id)
        now_s = time.time()
        csi = np.asarray(frame.grid)               # (A, S) complex
        amplitude = np.abs(csi)                    # (A, S)
        mean_amplitude = float(amplitude.mean())
        # cross-subcarrier std/mean over the antenna-averaged subcarriers = turbulence/motion
        magnitudes = amplitude.mean(axis=0)        # (S,) antenna-averaged
        variation_coefficient = float(magnitudes.std() / magnitudes.mean()) if magnitudes.mean() > 0 else 0.0
        self._wall_receive_times_s[node_id].append(now_s)
        self._mean_amplitudes[node_id].append(mean_amplitude)
        self._variation_coefficients[node_id].append(variation_coefficient)
        self._frame_counts[node_id] += 1
        self._last_timestamps_s[node_id] = float(frame.timestamp)

    def _sample_rate_hz(self, node_id) -> float:
        receive_times_s = self._wall_receive_times_s[node_id]
        if len(receive_times_s) < 2:
            return 0.0
        span_s = receive_times_s[-1] - receive_times_s[0]
        return (len(receive_times_s) - 1) / span_s if span_s > 0 else 0.0

    def snapshot(self) -> list[dict]:
        """Per-node health dict list. Call on UI cadence (~1 Hz), not per frame."""
        node_health = []
        for node_id in sorted(self._wall_receive_times_s.keys()):
            mean_amplitudes = np.asarray(self._mean_amplitudes[node_id], dtype=np.float64)
            mean_amplitude = float(mean_amplitudes.mean()) if mean_amplitudes.size else 0.0
            noise_floor = (float(np.percentile(mean_amplitudes, 5)) if mean_amplitudes.size >= 20
                           else mean_amplitude * 0.1)
            snr_db = float(20.0 * np.log10(mean_amplitude / noise_floor)) if noise_floor > 1e-12 else 0.0
            sample_rate_hz = self._sample_rate_hz(node_id)
            reference_amplitude = self._reference_amplitude_by_node.get(node_id)
            gain_drift = (round(mean_amplitude / reference_amplitude, 3)
                          if (reference_amplitude and reference_amplitude > 0) else None)
            node_health.append({
                "node_id": node_id,
                "band": "5GHz" if node_id >= 100 else "2.4GHz",
                "hz": round(sample_rate_hz, 1),
                "hz_ok": sample_rate_hz >= self.min_sample_rate_hz,
                "frames": self._frame_counts[node_id],
                "mean_amp": round(mean_amplitude, 4),
                "snr_db": round(snr_db, 1),
                "cv": round(
                    float(np.mean(list(self._variation_coefficients[node_id])))
                    if self._variation_coefficients[node_id] else 0.0,
                    4,
                ),
                "gain_drift": gain_drift,
                "loss_pct": round(
                    max(0.0, (self.target_sample_rate_hz - sample_rate_hz) / self.target_sample_rate_hz
                        * 100.0),
                    1,
                ),
                "subcarriers": self._subcarriers_by_node.get(node_id, []),
                "last_ts": round(self._last_timestamps_s.get(node_id, 0.0), 4),
            })
        return node_health


def cluster_sync(meter: NodeHealthMeter) -> dict:
    """Time-alignment health: spread of the latest per-node frame timestamps.
    Large spread => nodes are de-synced => stacking will drop windows. O(nodes)."""
    last_timestamp_s_by_node = {node_id: meter._last_timestamps_s.get(node_id, 0.0)
                                for node_id in meter._wall_receive_times_s.keys()}
    if len(last_timestamp_s_by_node) < 2:
        return {"spread_s": 0.0, "ok": True, "per_node_last": last_timestamp_s_by_node}
    last_timestamps_s = np.asarray(list(last_timestamp_s_by_node.values()))
    spread_s = float(last_timestamps_s.max() - last_timestamps_s.min())
    return {"spread_s": round(spread_s, 4), "ok": spread_s <= 0.05,
            "per_node_last": last_timestamp_s_by_node}


def baseline_drift(calibration_result, recent_frames) -> dict:
    """How far the current quiet room has drifted from the calibrated baseline.
    >~20% mean ratio suggests recalibration. O(F·S)."""
    if not recent_frames:
        return {"mean_ratio": None, "max_ratio": None, "recalibrate": False}
    amplitudes = np.stack([np.abs(np.asarray(frame.grid)).mean(axis=0) for frame in recent_frames])  # (F, S)
    current_magnitudes = amplitudes.mean(axis=0)
    baseline_magnitudes = np.asarray(calibration_result.baseline_mag, dtype=np.float64)
    subcarrier_count = min(current_magnitudes.size, baseline_magnitudes.size)
    ratio = current_magnitudes[:subcarrier_count] / np.where(
        baseline_magnitudes[:subcarrier_count] > 1e-9, baseline_magnitudes[:subcarrier_count], 1e-9)
    mean_ratio = float(np.mean(ratio))
    max_ratio = float(np.max(np.abs(ratio - 1.0)) + 1.0)
    return {"mean_ratio": round(mean_ratio, 3), "max_ratio": round(max_ratio, 3),
            "recalibrate": abs(mean_ratio - 1.0) > 0.2}


def feature_separation(x_intercarrier, y, *, variance_col=9) -> dict:
    """Is the σ²[p] weapon signature separable? Returns two histograms (weapon vs none) of the
    inter-carrier variance feature + a rank-AUC separability score. O(n log n).

    auc = 0.5 means no separation, 1.0 means perfect. Threshold 0.65 is a rough 'worth training' gate."""
    X = np.asarray(x_intercarrier, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] != y.size:
        return {"error": "shape"}
    variance_feature = X[:, variance_col]
    positives, negatives = variance_feature[y == 1], variance_feature[y == 0]
    if positives.size == 0 or negatives.size == 0:
        return {"error": "single_class"}
    # rank-AUC (Mann-Whitney U / n*m): P(random positive > random negative)
    order = np.argsort(np.concatenate([positives, negatives]), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    auc = ((ranks[:positives.size].sum() - positives.size * (positives.size + 1) / 2)
           / (positives.size * negatives.size))
    auc = float(max(auc, 1 - auc))  # fold: 0.5=random, 1.0=perfect
    bin_edges = np.linspace(float(variance_feature.min()), float(variance_feature.max()), 33)
    positive_histogram, _ = np.histogram(positives, bins=bin_edges)
    negative_histogram, _ = np.histogram(negatives, bins=bin_edges)
    return {
        "auc": round(auc, 3),
        "edges": bin_edges.tolist(),
        "weapon_hist": positive_histogram.tolist(),
        "none_hist": negative_histogram.tolist(),
        "separable": auc > 0.65,
    }


def dataset_report(dataset) -> dict:
    """4-quadrant style summary of a built dataset for the training UI panel."""
    y = np.asarray(dataset.y)
    classes, counts = np.unique(y, return_counts=True)
    return {
        "n_samples": int(y.size),
        "distribution": {str(int(class_id)): int(count) for class_id, count in zip(classes, counts)},
        "class_counts": {str(int(class_id)): int(count) for class_id, count in zip(classes, counts)},
        "balance": round(float(counts.min() / counts.max()), 3) if counts.size > 1 else 0.0,
        "sessions": sorted({str(group_id) for group_id in dataset.session_ids}),
        "subjects": sorted({str(group_id) for group_id in dataset.subject_ids}),
        "sync_error": dataset.meta.get("sync_error", {}),
        "num_nodes": dataset.meta.get("num_nodes", 1),
        "K": dataset.meta.get("K"),
        "K_img": dataset.meta.get("K_img"),
    }
