from wavetrace.calibration import Calibration, save_calibration


def calibrate_source(source, out_dir, *, baseline_packets=300, use_gain_lock=True, nbvi_max=12):
    calibration = Calibration(baseline_packets=baseline_packets, nbvi_max=nbvi_max,
                               use_gain_lock=use_gain_lock)
    for frame in source.frames():
        calibration.observe(frame)
    calibration_result = calibration.finalize()
    return save_calibration(calibration_result, out_dir), calibration_result
