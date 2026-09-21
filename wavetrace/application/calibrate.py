"""The calibrate use-case: quiet-baseline calibration -> a persisted calibration dir."""

from wavetrace.Calibration import Calibration, saveCalibration


def calibrate_source(source, out_dir, *, baseline_packets=300, use_gain_lock=True, nbvi_max=12):
    """Run the calibration flow over a quiet-baseline source and persist the result. Offline."""
    calibration = Calibration(baseline_packets=baseline_packets, nbvi_max=nbvi_max,
                               use_gain_lock=use_gain_lock)
    for frame in source.frames():
        calibration.observe(frame)
    result = calibration.finalize()
    return saveCalibration(result, out_dir), result
