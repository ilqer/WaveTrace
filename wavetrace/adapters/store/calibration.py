import json
from pathlib import Path

import numpy as np

from wavetrace import GainLock
from wavetrace.domain.calibration import CalibrationResult


def save_calibration(result: CalibrationResult, out_dir) -> Path:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "baseline_mag.npy", np.asarray(result.baseline_mag, dtype=np.float32))
    np.save(directory / "baseline_diff.npy", np.asarray(result.baseline_diff, dtype=np.complex64))
    meta = {
        # NaN is written as the bare token NaN (json's allow_nan default), which json.load reads back
        "reference_scale": float(result.reference_scale),
        "subcarriers": [int(subcarrier) for subcarrier in result.subcarriers],
        "image_subcarriers": [int(subcarrier) for subcarrier in result.image_subcarriers],
        "numBaseline": int(result.num_baseline),
    }
    with open(directory / "meta.json", "w") as meta_file:
        json.dump(meta, meta_file, indent=2)
    return directory


def load_calibration(calibration_dir) -> tuple[CalibrationResult, GainLock | None]:
    """gain_lock is None when the saved reference_scale is NaN (calibrated without a gain lock)."""
    directory = Path(calibration_dir)
    with open(directory / "meta.json") as meta_file:
        meta = json.load(meta_file)
    reference_scale = float(meta["reference_scale"])
    result = CalibrationResult(
        reference_scale=reference_scale,
        subcarriers=[int(subcarrier) for subcarrier in meta["subcarriers"]],
        image_subcarriers=[int(subcarrier)
                           for subcarrier in meta.get("image_subcarriers", meta["subcarriers"])],
        num_baseline=int(meta.get("numBaseline", meta.get("num_baseline"))),
        baseline_mag=np.load(directory / "baseline_mag.npy"),
        baseline_diff=np.load(directory / "baseline_diff.npy"),
    )
    gain_lock = None
    if not np.isnan(reference_scale):
        gain_lock = GainLock(result.num_baseline)
        gain_lock.lock_to(reference_scale)
    return result, gain_lock
