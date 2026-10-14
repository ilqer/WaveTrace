"""Directory-based entry points: load from disk, run a `wavetrace.application` use-case, save to disk."""

from wavetrace.adapters.recognition.session import open_session
from wavetrace.adapters.store.calibration import load_calibration, save_calibration
from wavetrace.adapters.store.dataset import save_dataset
from wavetrace.adapters.store.localization import save_localization
from wavetrace.application.behaviours import (
    AlertGuardBehaviour, DriftMonitoredSource, SegmentVoteBehaviour,
)
from wavetrace.application.calibrate import calibrate
from wavetrace.application.collect import collect_dataset
from wavetrace.application.localize import localize
from wavetrace.application.serve import infer_windows
from wavetrace.domain.calibration import Calibration
from wavetrace.domain.contracts import DEFAULT_HOP_FRAMES, DEFAULT_WINDOW_FRAMES
from wavetrace.domain.guard import AlertGuard, DriftMonitor
from wavetrace.domain.localize import Localizer, Tracker
from wavetrace.domain.recognition.vote import SegmentVoter


def calibrate_source(source, out_dir, *, baseline_packets=300, use_gain_lock=True, nbvi_max=12):
    """Returns (calibration dir, CalibrationResult)."""
    calibration_result = calibrate(source, Calibration(baseline_packets=baseline_packets, nbvi_max=nbvi_max,
                                                       use_gain_lock=use_gain_lock))
    return save_calibration(calibration_result, out_dir), calibration_result


def collect_source(source, calibration_dir, out_dir, spans, *, stage="presence",
                    window=DEFAULT_WINDOW_FRAMES, hop=DEFAULT_HOP_FRAMES,
                    session_id="", subject_id="", frame_average=1, subtract_baseline=False,
                    subtract_ic_baseline=False, labeler=None, tier=""):
    """Returns (dataset dir, Dataset); the options are `collect_dataset`'s."""
    calibration_result, gain_lock = load_calibration(calibration_dir)
    dataset = collect_dataset(source, calibration_result, gain_lock, spans, stage=stage, window=window,
                              hop=hop, session_id=session_id, subject_id=subject_id,
                              frame_average=frame_average, subtract_baseline=subtract_baseline,
                              subtract_ic_baseline=subtract_ic_baseline, labeler=labeler, tier=tier)
    return save_dataset(dataset, out_dir), dataset


def run_inference(source, calibration_dir, model_path, mode, publisher, *, vote=False, guard=False):
    """mode: 'presence' or 'weapon', picks the head loader. guard adds recalibration advisories
    (O(S) extra per frame) and, in weapon mode only, debounced weapon alerts; vote appends one soft-vote verdict over the
    stream. Returns the published verdicts."""
    calibration_result, gain_lock = load_calibration(calibration_dir)
    session = open_session(mode, model_path)
    behaviours = []
    if guard:
        source = DriftMonitoredSource(source, DriftMonitor(calibration_result.baseline_mag), publisher)
        if mode == "weapon":
            behaviours.append(AlertGuardBehaviour(AlertGuard()))
    if vote:
        behaviours.append(SegmentVoteBehaviour(SegmentVoter()))
    return infer_windows(source, calibration_result, gain_lock, session, mode, publisher,
                         behaviours=behaviours)


def localize_source(source, out_dir, *, num_antennas, antenna_spacing_wavelengths=0.5,
                     method="music", num_sources=1, num_angles=181,
                     subcarrier_spacing_hz=312.5e3, max_range_m=12.0, num_ranges=64,
                     range_enabled=True, filter_track=True, publisher=None):
    """Returns (localization dir, aggregate Localization). Needs >= 2 RX antennas (2-antenna ESP32 /
    Pi NIC). filter_track=False publishes the raw per-frame fixes instead of the Kalman track."""
    localizer = Localizer(num_antennas, spacing=antenna_spacing_wavelengths, method=method,
                          num_sources=num_sources, num_angles=num_angles,
                          subcarrier_spacing_hz=subcarrier_spacing_hz, max_range_m=max_range_m,
                          num_ranges=num_ranges, range_enabled=range_enabled)
    tracker = Tracker(range_enabled=range_enabled) if filter_track else None
    room_map = localize(source, localizer, tracker, publisher)
    return save_localization(room_map, out_dir), room_map
