"""Per-session calibration from an empty room: amplitude scale plus NBVI subcarrier selection.

The gain lock is for amplitude and presence features only. Phase and material features need the
absolute attenuation it removes.
"""

from dataclasses import dataclass, field

import numpy as np

from wavetrace import CsiFrame, GainLock, select_subcarriers_nbvi, valid_subcarriers


@dataclass
class CalibrationResult:
    reference_scale: float       # GainLock reference amplitude level (NaN if gain lock disabled)
    subcarriers: list[int]       # NBVI-selected, non-consecutive subcarrier indices
    num_baseline: int
    baseline_mag: np.ndarray     # mean |H| per subcarrier over the quiet baseline, shape (S,)
    baseline_diff: np.ndarray    # mean CFO-free differential channel H(k)·conj(H(k-1)), complex, (S-1,)
    # ALL noise-gate-passing subcarriers, ascending — the CNN image rows
    image_subcarriers: list[int] = field(default_factory=list)


def reflection_signature(grid, result: CalibrationResult):
    """Material signature (subject vs baseline). Returns (magnitude_ratio, phase_delta).

    * magnitude_ratio[k]: Attenuation coefficient. Metal changes it from 1.
    * phase_delta[k]: Shift in CFO-free differential phase. Measures mm-level path-length changes.

    Pass a raw CSI grid: gain-locked frames have lost the attenuation this measures. Antennas are
    averaged. O(A·S)."""
    csi = np.asarray(grid)
    amplitude = np.abs(csi).mean(axis=0)
    differential_channel = (csi[:, 1:] * np.conj(csi[:, :-1])).mean(axis=0)
    magnitude_ratio = amplitude / np.where(result.baseline_mag > 1e-12, result.baseline_mag, 1e-12)
    phase_delta = np.angle(differential_channel * np.conj(result.baseline_diff))   # wrap-safe in (-pi, pi]
    return magnitude_ratio.astype(np.float32), phase_delta.astype(np.float32)


def build_image_baseline(result: "CalibrationResult", *, locked: bool) -> np.ndarray:
    """Quiet-room baseline. O(S). Rescales to gain-lock basis if locked."""
    baseline_magnitudes = np.asarray(result.baseline_mag, dtype=np.float32)
    if locked and not np.isnan(float(result.reference_scale)):
        return baseline_magnitudes * (float(result.reference_scale) / float(baseline_magnitudes.mean()))
    return baseline_magnitudes.copy()


class Calibration:
    def __init__(
        self,
        *,
        baseline_packets: int = 300,
        nbvi_max: int = 12,
        nbvi_alpha: float = 0.75,
        noise_gate_percentile: float = 0.15,
        use_gain_lock: bool = True,
    ):
        self._baseline_packets = baseline_packets
        self._gain_lock = GainLock(baseline_packets) if use_gain_lock else None
        self._nbvi_max_subcarriers = nbvi_max
        self._nbvi_alpha = nbvi_alpha
        self._noise_gate_percentile = noise_gate_percentile
        self._amplitudes: list[np.ndarray] = []
        self._differential_channels: list[np.ndarray] = []

    def observe(self, frame: CsiFrame) -> None:
        if self._gain_lock is not None:
            self._gain_lock.observe(frame)
        csi = np.asarray(frame.grid)
        self._amplitudes.append(np.abs(csi).mean(axis=0))
        self._differential_channels.append((csi[:, 1:] * np.conj(csi[:, :-1])).mean(axis=0))

    @property
    def ready(self) -> bool:
        return len(self._amplitudes) >= self._baseline_packets

    @property
    def num_baseline(self) -> int:
        return len(self._amplitudes)

    @property
    def gain_lock(self) -> GainLock:
        """The locked GainLock — call .apply(frame) on it during deployment (amplitude path only)."""
        if self._gain_lock is None:
            raise ValueError("Calibration: gain lock disabled (use_gain_lock=False)")
        return self._gain_lock

    def finalize(self) -> CalibrationResult:
        """Also locks the GainLock exposed by `gain_lock` (when enabled). Raises ValueError with no
        frames or fewer than `baseline_packets`."""
        if not self._amplitudes:
            raise ValueError("Calibration: no baseline frames observed")
        if not self.ready:
            raise ValueError(
                f"Calibration: only {len(self._amplitudes)} baseline frames observed, "
                f"need >= {self._baseline_packets} (collect more, or lower baseline_packets)"
            )
        if self._gain_lock is not None:
            self._gain_lock.finalize()
            reference_scale = self._gain_lock.reference_scale
        else:
            reference_scale = float("nan")
        amplitudes = np.stack(self._amplitudes).astype(np.float32)
        subcarriers = select_subcarriers_nbvi(
            amplitudes,
            alpha=self._nbvi_alpha,
            max_subcarriers=self._nbvi_max_subcarriers,
            noise_gate_percentile=self._noise_gate_percentile,
        )
        image_subcarriers = valid_subcarriers(amplitudes, noise_gate_percentile=self._noise_gate_percentile)
        return CalibrationResult(
            reference_scale=reference_scale,
            subcarriers=list(subcarriers),
            image_subcarriers=list(image_subcarriers),
            num_baseline=len(self._amplitudes),
            baseline_mag=amplitudes.mean(axis=0),
            baseline_diff=np.stack(self._differential_channels).mean(axis=0),
        )
