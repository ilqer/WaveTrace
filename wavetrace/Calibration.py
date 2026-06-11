"""Phase 3c — per-session calibration flow.

Calibrate on a QUIET, empty space (REFERENCE_DIGEST §4): collect a baseline of still frames,
optionally lock the AGC gain (GainLock), and select the informative subcarriers (NBVI). The result
feeds the deployment pipeline: rescale amplitudes with the locked gain and restrict features to the
chosen subcarriers. All offline (not the real-time path).

GainLock is OPTIONAL (`use_gain_lock`, gated by Config.signal.gain_lock_enabled): it serves only the
amplitude / presence feature path. The phase path is scale-invariant and the material features
(σ²[p], reflection_signature) must NOT consume gain-locked frames — see reflection_signature.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from wavetrace import CsiFrame, GainLock, select_subcarriers_nbvi


@dataclass
class CalibrationResult:
    reference_scale: float       # GainLock reference amplitude level (NaN if gain lock disabled)
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
    `grid` = one subject frame's RAW complex CSI (A x S) — do NOT pass a GainLock.apply'd frame: gain
    lock rescales every frame to a common mean, which cancels exactly the bulk attenuation mag_ratio
    measures. Antennas are averaged (magnitude) / complex-fused (differential). Offline. O(A·S)."""
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
        use_gain_lock: bool = True,
    ):
        self._baseline_packets = baseline_packets
        self._gain = GainLock(baseline_packets) if use_gain_lock else None
        self._nbvi_max = nbvi_max
        self._nbvi_alpha = nbvi_alpha
        self._gate = noise_gate_percentile
        self._amps: list[np.ndarray] = []
        self._diffs: list[np.ndarray] = []

    def observe(self, frame: CsiFrame) -> None:
        """Add one quiet-baseline frame. O(n)."""
        if self._gain is not None:
            self._gain.observe(frame)
        g = np.asarray(frame.grid)
        self._amps.append(np.abs(g).mean(axis=0))               # antenna-averaged |.| per subcarrier
        self._diffs.append((g[:, 1:] * np.conj(g[:, :-1])).mean(axis=0))  # CFO-free differential (S-1,)

    @property
    def ready(self) -> bool:
        """True once enough baseline frames have been collected (per baseline_packets). Counts
        observed frames directly so it holds whether or not the gain lock is enabled."""
        return len(self._amps) >= self._baseline_packets

    @property
    def num_baseline(self) -> int:
        return len(self._amps)

    @property
    def gain_lock(self) -> GainLock:
        """The locked GainLock — call .apply(frame) on it during deployment (amplitude path only)."""
        if self._gain is None:
            raise ValueError("Calibration: gain lock disabled (use_gain_lock=False)")
        return self._gain

    def finalize(self) -> CalibrationResult:
        """Lock the gain (if enabled) and run NBVI; returns the calibration result. Offline.

        Guards on `ready`: a too-short quiet baseline yields a weak reference scale / NBVI ranking,
        so finalize refuses unless baseline_packets frames were observed (raise, don't silently
        proceed). reference_scale is NaN when the gain lock is disabled."""
        if not self._amps:
            raise ValueError("Calibration: no baseline frames observed")
        if not self.ready:
            raise ValueError(
                f"Calibration: only {len(self._amps)} baseline frames observed, "
                f"need >= {self._baseline_packets} (collect more, or lower baseline_packets)"
            )
        if self._gain is not None:
            self._gain.finalize()
            reference_scale = self._gain.reference_scale
        else:
            reference_scale = float("nan")
        amp = np.stack(self._amps).astype(np.float32)  # (F, S)
        subc = select_subcarriers_nbvi(
            amp,
            alpha=self._nbvi_alpha,
            max_subcarriers=self._nbvi_max,
            noise_gate_percentile=self._gate,
        )
        return CalibrationResult(
            reference_scale=reference_scale,
            subcarriers=list(subc),
            num_baseline=len(self._amps),
            baseline_mag=amp.mean(axis=0),                      # (S,) mean |H| over the baseline
            baseline_diff=np.stack(self._diffs).mean(axis=0),   # (S-1,) mean CFO-free differential
        )


def save_calibration(result: CalibrationResult, out_dir) -> Path:
    """Serialize a CalibrationResult to out_dir (meta.json + .npy), mirroring save_dataset. O(S)."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "baseline_mag.npy", np.asarray(result.baseline_mag, dtype=np.float32))
    np.save(p / "baseline_diff.npy", np.asarray(result.baseline_diff, dtype=np.complex64))
    meta = {
        "reference_scale": float(result.reference_scale),  # JSON null for NaN -> handled on load
        "subcarriers": [int(s) for s in result.subcarriers],
        "num_baseline": int(result.num_baseline),
    }
    with open(p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return p


def load_calibration(out_dir) -> tuple[CalibrationResult, GainLock | None]:
    """Round-trip a saved calibration. Returns (result, gain_lock): the GainLock is rebuilt locked
    from the persisted reference_scale (via lock_to), or None when the lock was disabled (NaN). O(S)."""
    p = Path(out_dir)
    with open(p / "meta.json") as f:
        meta = json.load(f)
    ref = float(meta["reference_scale"])
    result = CalibrationResult(
        reference_scale=ref,
        subcarriers=[int(s) for s in meta["subcarriers"]],
        num_baseline=int(meta["num_baseline"]),
        baseline_mag=np.load(p / "baseline_mag.npy"),
        baseline_diff=np.load(p / "baseline_diff.npy"),
    )
    gain_lock = None
    if not np.isnan(ref):
        gain_lock = GainLock(result.num_baseline)
        gain_lock.lock_to(ref)
    return result, gain_lock
