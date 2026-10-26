"""Per-node link health + dataset diagnostics, off the hot path.

node_id < 100 => 2.4 GHz (ESP32), node_id >= 100 => 5 GHz (nexmon Pi).
"""

import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from wavetrace.domain.calibration import CalibrationResult
from wavetrace.domain.contracts import (
    DEFAULT_MIN_SAMPLE_RATE_FRACTION, DEFAULT_TARGET_SAMPLE_RATE_HZ, require_valid_min_sample_rate_fraction,
)
from wavetrace.domain.rules import SampleRateMustBeWithinTolerance

_UNCALIBRATED_REFERENCE = (0.0, ())


@dataclass(frozen=True, slots=True)
class HealthMeterOptions:
    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ
    min_sample_rate_fraction: float = DEFAULT_MIN_SAMPLE_RATE_FRACTION
    rolling_window_frames: int = 200

    def __post_init__(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        require_valid_min_sample_rate_fraction(self.min_sample_rate_fraction)
        if self.rolling_window_frames <= 0:
            raise ValueError("rolling_window_frames must be positive")


class NodeHealthMeter:
    """O(1) per frame, O(nodes·rolling_window_frames) per snapshot.

    gain_drift = rolling mean |H| / the calibration's quiet-room mean |H| (1.0 = gain unchanged), None
    for a node with neither its own calibration nor `default_calibration`."""

    def __init__(self, options: HealthMeterOptions = HealthMeterOptions(), *,
                 calibration_by_node: Mapping[int, CalibrationResult] | None = None,
                 default_calibration: CalibrationResult | None = None):
        self.target_sample_rate_hz = options.target_sample_rate_hz
        self._sample_rate_rule = SampleRateMustBeWithinTolerance(options.target_sample_rate_hz,
                                                                 options.min_sample_rate_fraction)
        self.rolling_window_frames = options.rolling_window_frames
        self._wall_receive_times_s = defaultdict(lambda: deque(maxlen=self.rolling_window_frames))
        self._mean_amplitudes = defaultdict(lambda: deque(maxlen=self.rolling_window_frames))
        # cross-subcarrier coefficient of variation (std/mean) per frame
        self._variation_coefficients = defaultdict(lambda: deque(maxlen=self.rolling_window_frames))
        self._frame_counts = defaultdict(int)
        self._last_timestamps_s = {}
        self._reference_by_node = {int(node_id): _calibration_reference(result)
                                   for node_id, result in (calibration_by_node or {}).items()}
        self._default_reference = (_calibration_reference(default_calibration)
                                   if default_calibration is not None else _UNCALIBRATED_REFERENCE)

    def observe(self, frame) -> None:
        """O(A·S)."""
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
        """Call on UI cadence (~1 Hz), not per frame."""
        node_health = []
        for node_id in sorted(self._wall_receive_times_s.keys()):
            mean_amplitudes = np.asarray(self._mean_amplitudes[node_id], dtype=np.float64)
            mean_amplitude = float(mean_amplitudes.mean()) if mean_amplitudes.size else 0.0
            noise_floor = (float(np.percentile(mean_amplitudes, 5)) if mean_amplitudes.size >= 20
                           else mean_amplitude * 0.1)
            snr_db = float(20.0 * np.log10(mean_amplitude / noise_floor)) if noise_floor > 1e-12 else 0.0
            sample_rate_hz = self._sample_rate_hz(node_id)
            reference_amplitude, subcarriers = self._reference_by_node.get(node_id, self._default_reference)
            node_health.append({
                "node_id": node_id,
                "band": "5GHz" if node_id >= 100 else "2.4GHz",
                "hz": round(sample_rate_hz, 1),
                "hz_ok": self._sample_rate_rule.is_satisfied(sample_rate_hz),
                "frames": self._frame_counts[node_id],
                "mean_amp": round(mean_amplitude, 4),
                "snr_db": round(snr_db, 1),
                "cv": round(
                    float(np.mean(list(self._variation_coefficients[node_id])))
                    if self._variation_coefficients[node_id] else 0.0,
                    4,
                ),
                "gain_drift": (round(mean_amplitude / reference_amplitude, 3)
                               if reference_amplitude > 0 else None),
                "loss_pct": round(
                    max(0.0, (self.target_sample_rate_hz - sample_rate_hz) / self.target_sample_rate_hz
                        * 100.0),
                    1,
                ),
                "subcarriers": list(subcarriers),
                "last_ts": round(self._last_timestamps_s.get(node_id, 0.0), 4),
            })
        return node_health


def _calibration_reference(result: CalibrationResult) -> tuple[float, tuple[int, ...]]:
    # baseline_mag is antenna-averaged per subcarrier, so its mean is the same all-(antenna,
    # subcarrier) mean |H| that observe() records live
    return float(np.mean(result.baseline_mag)), tuple(int(subcarrier) for subcarrier in result.subcarriers)


def cluster_sync(meter: NodeHealthMeter) -> dict:
    """Large spread of the latest per-node timestamps => nodes de-synced => stacking drops windows.
    O(nodes)."""
    last_timestamp_s_by_node = {node_id: meter._last_timestamps_s.get(node_id, 0.0)
                                for node_id in meter._wall_receive_times_s.keys()}
    if len(last_timestamp_s_by_node) < 2:
        return {"spread_s": 0.0, "ok": True, "per_node_last": last_timestamp_s_by_node}
    last_timestamps_s = np.asarray(list(last_timestamp_s_by_node.values()))
    spread_s = float(last_timestamps_s.max() - last_timestamps_s.min())
    return {"spread_s": round(spread_s, 4), "ok": spread_s <= 0.05,
            "per_node_last": last_timestamp_s_by_node}


def baseline_drift(calibration_result, recent_frames) -> dict:
    """Mean ratio more than ~20% from 1 suggests recalibration. O(F·S)."""
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


def dataset_report(datasets) -> dict:
    """The class balance, sessions and subjects of every dataset together; sync_error, num_nodes, K and K_img
    are the first dataset's. Each needs only y, session_ids, subject_ids and meta. O(total windows)."""
    datasets = list(datasets)
    y = np.concatenate([np.asarray(dataset.y) for dataset in datasets])
    classes, counts = np.unique(y, return_counts=True)
    first_meta = datasets[0].meta
    return {
        "n_samples": int(y.size),
        "distribution": {str(int(class_id)): int(count) for class_id, count in zip(classes, counts)},
        "class_counts": {str(int(class_id)): int(count) for class_id, count in zip(classes, counts)},
        "balance": round(float(counts.min() / counts.max()), 3) if counts.size > 1 else 0.0,
        "sessions": sorted({str(group_id) for dataset in datasets for group_id in dataset.session_ids}),
        "subjects": sorted({str(group_id) for dataset in datasets for group_id in dataset.subject_ids}),
        "sync_error": first_meta.get("sync_error", {}),
        "num_nodes": first_meta.get("num_nodes", 1),
        "K": first_meta.get("K"),
        "K_img": first_meta.get("K_img"),
    }
