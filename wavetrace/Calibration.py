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
    baseline_mag: np.ndarray     # mean |H| per subcarrier over the quiet baseline, shape (S,)
    baseline_diff: np.ndarray    # mean CFO-free differential channel H(k)·conj(H(k-1)), complex, (S-1,)


def reflection_signature(grid, result: CalibrationResult):
    """Material/reflection signature of a subject frame vs the empty-room baseline (REFERENCE §0B,
    in-baggage/material-ID). Returns (mag_ratio, phase_delta):
      * mag_ratio[k]   = |H_subj(k)| / |H_base(k)|   — the reflection/attenuation coefficient; a metal
                         object in the path drives it away from 1 (per-subcarrier dielectric signature).
      * phase_delta[k] = ∠( D_subj(k) · conj(D_base(k)) ), D(k)=H(k)·conj(H(k-1)) — the change in the
                         CFO-FREE differential (group-delay) phase; phase resolves mm-level path-length
                         change, and D is the complex quantity compressed sensing super-resolves in the
                         delay domain. CFO is common-mode within a frame so the differential cancels it,
                         making this comparable across captures (raw absolute phase is NOT).
    `grid` = one subject frame's complex CSI (A x S). Antennas are averaged (magnitude) / complex-fused
    (differential). Offline. O(A·S)."""
    g = np.asarray(grid)
    amp = np.abs(g).mean(axis=0)                                   # (S,) antenna-averaged |H|
    diff = (g[:, 1:] * np.conj(g[:, :-1])).mean(axis=0)            # (S-1,) CFO-free differential
    mag_ratio = amp / np.where(result.baseline_mag > 1e-12, result.baseline_mag, 1e-12)
    phase_delta = np.angle(diff * np.conj(result.baseline_diff))   # wrap-safe in (-pi, pi]
    return mag_ratio.astype(np.float32), phase_delta.astype(np.float32)


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
        self._diffs: list[np.ndarray] = []

    def observe(self, frame: CsiFrame) -> None:
        """Add one quiet-baseline frame. O(n)."""
        self._gain.observe(frame)
        g = np.asarray(frame.grid)
        self._amps.append(np.abs(g).mean(axis=0))               # antenna-averaged |.| per subcarrier
        self._diffs.append((g[:, 1:] * np.conj(g[:, :-1])).mean(axis=0))  # CFO-free differential (S-1,)

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
            baseline_mag=amp.mean(axis=0),                      # (S,) mean |H| over the baseline
            baseline_diff=np.stack(self._diffs).mean(axis=0),   # (S-1,) mean CFO-free differential
        )
