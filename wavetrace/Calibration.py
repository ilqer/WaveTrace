"""Phase 3c — per-session calibration flow.

Calibrate on a QUIET, empty space (REFERENCE_DIGEST §4): collect a baseline of still frames, lock
the AGC gain (GainLock), and select the informative subcarriers (NBVI). The result feeds the
deployment pipeline: rescale amplitudes with the locked gain and restrict features to the chosen
subcarriers. All offline (not the real-time path).
"""

from dataclasses import dataclass

import numpy as np

from wavetrace import CsiFrame, GainLock, select_subcarriers_nbvi


@dataclass
class CalibrationResult:
    reference_scale: float       # GainLock reference amplitude level
    subcarriers: list[int]       # NBVI-selected, non-consecutive subcarrier indices
    num_baseline: int            # number of baseline frames used


class Calibration:
    """Accumulate quiet-baseline frames, then produce a CalibrationResult.

    NBVI is computed on the antenna-averaged magnitude per subcarrier (the caller's geometry is
    collapsed to a per-subcarrier view, since NBVI selects subcarriers, not antennas).
    """

    def __init__(
        self,
        *,
        baseline_packets: int = 300,
        nbvi_max: int = 12,
        nbvi_alpha: float = 0.75,
        noise_gate_percentile: float = 0.15,
    ):
        self._gain = GainLock(baseline_packets)
        self._nbvi_max = nbvi_max
        self._nbvi_alpha = nbvi_alpha
        self._gate = noise_gate_percentile
        self._amps: list[np.ndarray] = []

    def observe(self, frame: CsiFrame) -> None:
        """Add one quiet-baseline frame. O(n)."""
        self._gain.observe(frame)
        self._amps.append(np.abs(frame.grid).mean(axis=0))  # antenna-averaged |.| per subcarrier

    @property
    def ready(self) -> bool:
        """True once enough baseline frames have been collected (per baseline_packets)."""
        return self._gain.ready

    @property
    def num_baseline(self) -> int:
        return len(self._amps)

    @property
    def gain_lock(self) -> GainLock:
        """The locked GainLock — call .apply(frame) on it during deployment."""
        return self._gain

    def finalize(self) -> CalibrationResult:
        """Lock the gain and run NBVI; returns the calibration result. Offline."""
        if not self._amps:
            raise ValueError("Calibration: no baseline frames observed")
        self._gain.finalize()
        amp = np.stack(self._amps).astype(np.float32)  # (F, S)
        subc = select_subcarriers_nbvi(
            amp,
            alpha=self._nbvi_alpha,
            max_subcarriers=self._nbvi_max,
            noise_gate_percentile=self._gate,
        )
        return CalibrationResult(
            reference_scale=self._gain.reference_scale,
            subcarriers=list(subc),
            num_baseline=len(self._amps),
        )
