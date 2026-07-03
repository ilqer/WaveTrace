"""Phase 3c: Per-session calibration.

Empty-room baseline calibration. Output sets amplitude scale and subcarrier selection (NBVI).

GainLock is optional. Only use it for amplitude/presence features. Do not use for phase or material features (they need absolute attenuation).
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from wavetrace import CsiFrame, GainLock, select_subcarriers_nbvi, valid_subcarriers


@dataclass
class CalibrationResult:
    reference_scale: float       # GainLock reference amplitude level (NaN if gain lock disabled)
    subcarriers: list[int]       # NBVI-selected, non-consecutive subcarrier indices
    numBaseline: int            # number of baseline frames used
    baseline_mag: np.ndarray     # mean |H| per subcarrier over the quiet baseline, shape (S,)
    baseline_diff: np.ndarray    # mean CFO-free differential channel H(k)·conj(H(k-1)), complex, (S-1,)
    image_subcarriers: list[int] = field(default_factory=list)  # ALL noise-gate-passing subcarriers, ascending — CNN image rows


def reflectionSignature(grid, result: CalibrationResult):
    """Material signature (subject vs baseline). Returns (mag_ratio, phase_delta).

    * mag_ratio[k]: Attenuation coefficient. Metal changes it from 1.
    * phase_delta[k]: Shift in CFO-free differential phase. Measures mm-level path-length changes.

    Pass raw CSI grid. Do NOT pass GainLock'd frames (destroys attenuation info). Antennas are averaged/fused. O(A·S)."""
    g = np.asarray(grid)
    amp = np.abs(g).mean(axis=0)
    diff = (g[:, 1:] * np.conj(g[:, :-1])).mean(axis=0)
    magRatio = amp / np.where(result.baseline_mag > 1e-12, result.baseline_mag, 1e-12)
    phaseDelta = np.angle(diff * np.conj(result.baseline_diff))   # wrap-safe in (-pi, pi]
    return magRatio.astype(np.float32), phaseDelta.astype(np.float32)


def imageBaseline(result: "CalibrationResult", *, locked: bool) -> np.ndarray:
    """Quiet-room baseline. O(S). Rescales to gain-lock basis if locked."""
    b = np.asarray(result.baseline_mag, dtype=np.float32)
    if locked and not np.isnan(float(result.reference_scale)):
        return b * (float(result.reference_scale) / float(b.mean()))
    return b.copy()


class Calibration:
    """Accumulate baseline frames, produce CalibrationResult."""

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
        self._amps.append(np.abs(g).mean(axis=0))
        self._diffs.append((g[:, 1:] * np.conj(g[:, :-1])).mean(axis=0))

    @property
    def ready(self) -> bool:
        """True when baseline_packets frames observed."""
        return len(self._amps) >= self._baseline_packets

    @property
    def numBaseline(self) -> int:
        return len(self._amps)

    @property
    def gainLock(self) -> GainLock:
        """The locked GainLock — call .apply(frame) on it during deployment (amplitude path only)."""
        if self._gain is None:
            raise ValueError("Calibration: gain lock disabled (use_gain_lock=False)")
        return self._gain

    def finalize(self) -> CalibrationResult:
        """Finalize calibration: lock gain, run NBVI. Returns CalibrationResult."""
        if not self._amps:
            raise ValueError("Calibration: no baseline frames observed")
        if not self.ready:
            raise ValueError(
                f"Calibration: only {len(self._amps)} baseline frames observed, "
                f"need >= {self._baseline_packets} (collect more, or lower baseline_packets)"
            )
        if self._gain is not None:
            self._gain.finalize()
            referenceScale = self._gain.reference_scale
        else:
            referenceScale = float("nan")
        amp = np.stack(self._amps).astype(np.float32)
        subc = select_subcarriers_nbvi(
            amp,
            alpha=self._nbvi_alpha,
            max_subcarriers=self._nbvi_max,
            noise_gate_percentile=self._gate,
        )
        imgSubc = valid_subcarriers(amp, noise_gate_percentile=self._gate)
        return CalibrationResult(
            reference_scale=referenceScale,
            subcarriers=list(subc),
            image_subcarriers=list(imgSubc),
            numBaseline=len(self._amps),
            baseline_mag=amp.mean(axis=0),
            baseline_diff=np.stack(self._diffs).mean(axis=0),
        )


def saveCalibration(result: CalibrationResult, out_dir) -> Path:
    """Serialize a CalibrationResult to out_dir (meta.json + .npy), mirroring saveDataset. O(S)."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "baseline_mag.npy", np.asarray(result.baseline_mag, dtype=np.float32))
    np.save(p / "baseline_diff.npy", np.asarray(result.baseline_diff, dtype=np.complex64))
    meta = {
        "reference_scale": float(result.reference_scale),  # NaN -> JSON null, handled on load
        "subcarriers": [int(s) for s in result.subcarriers],
        "image_subcarriers": [int(s) for s in result.image_subcarriers],
        "numBaseline": int(result.numBaseline),
    }
    with open(p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return p


def loadCalibration(out_dir) -> tuple[CalibrationResult, GainLock | None]:
    """Round-trip saved calibration. Returns (result, gainLock). O(S)."""
    p = Path(out_dir)
    with open(p / "meta.json") as f:
        meta = json.load(f)
    ref = float(meta["reference_scale"])
    result = CalibrationResult(
        reference_scale=ref,
        subcarriers=[int(s) for s in meta["subcarriers"]],
        image_subcarriers=[int(s) for s in meta.get("image_subcarriers", meta["subcarriers"])],
        numBaseline=int(meta.get("numBaseline", meta.get("num_baseline"))),
        baseline_mag=np.load(p / "baseline_mag.npy"),
        baseline_diff=np.load(p / "baseline_diff.npy"),
    )
    gainLock = None
    if not np.isnan(ref):
        gainLock = GainLock(result.numBaseline)
        gainLock.lock_to(ref)
    return result, gainLock
