from wavetrace.application.ports import CsiSource
from wavetrace.domain.calibration import Calibration, CalibrationResult


def calibrate(source: CsiSource, calibration: Calibration) -> CalibrationResult:
    """Consumes `calibration`: finalize() also locks its `gain_lock` when enabled."""
    for frame in source.frames():
        calibration.observe(frame)
    return calibration.finalize()
