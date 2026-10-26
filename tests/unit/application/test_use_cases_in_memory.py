import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fixtures.synthetic_csi import calibrate_quiet_baseline, generate_quiet_baseline
from wavetrace.adapters.source import SyntheticSource
from wavetrace.application.calibrate import calibrate
from wavetrace.application.collect import collect_dataset
from wavetrace.application.localize import localize
from wavetrace.application.behaviours import AlertGuardBehaviour, SegmentVoteBehaviour
from wavetrace.application.serve import infer_windows, iter_window_verdicts
from wavetrace.domain.calibration import Calibration
from wavetrace.domain.contracts import derive_pipeline_contract
from wavetrace.domain.guard import AlertGuard, WeaponAlertEvent
from wavetrace.domain.localize import Localizer
from wavetrace.domain.model_config import ModelConfig
from wavetrace.domain.recognition.vote import SegmentVoter
from wavetrace.domain.rules import BusinessRuleViolationError
from wavetrace.domain.tracking import Tracker
from wavetrace.synthetic import generate_paired_recording, generate_stream

NUM_ANTENNAS, NUM_SUBCARRIERS, SAMPLE_RATE_HZ = 2, 32, 100.0
WINDOW_FRAMES, HOP_FRAMES = 32, 16
PRESENCE_SPAN_S = (3.0, 7.0)


class _RecordingPublisher:
    def __init__(self):
        self.results, self.events = [], []

    def publish(self, result):
        self.results.append(result)

    def publish_event(self, event):
        self.events.append(event)


class _FixedVerdictSession:
    """Always answers class 1 with probability 0.75. trained_subcarrier_width None is a legacy model."""

    def __init__(self, trained_subcarrier_width=None, frame_average=1):
        self.num_forward_passes = 0
        config = ModelConfig(stage="presence", k=4, window=WINDOW_FRAMES, hop=HOP_FRAMES,
                             frame_average=frame_average)
        self.head = SimpleNamespace(
            config=config,
            contract=replace(derive_pipeline_contract(config), subcarrier_width=trained_subcarrier_width),
            classes_=np.array([0, 1]),
            predict_proba=self._predict_proba)

    def _predict_proba(self, rows):
        self.num_forward_passes += 1
        return np.tile([0.25, 0.75], (len(rows), 1))

    def predict_window_proba(self, feature_vector):
        return self.head.predict_proba(np.asarray(feature_vector).reshape(1, -1))[0]


def _calibrated():
    return calibrate_quiet_baseline(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        num_frames=60, seed=7, baseline_packets=50)


def _recording(**signal_options):
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        duration_s=10.0, camera_frame_rate_hz=30.0, seed=300, **signal_options)
    return frames


def _collect_options(**overrides):
    options = dict(stage="presence", window=WINDOW_FRAMES, hop=HOP_FRAMES, session_id="s0",
                   subject_id="u0", frame_average=1, subtract_baseline=False,
                   subtract_ic_baseline=False, labeler=None, tier="")
    options.update(overrides)
    return options


def test_calibrate_returns_the_baseline_without_touching_disk():
    expected_result, _ = _calibrated()
    baseline = generate_quiet_baseline(num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS,
                                       sample_rate_hz=SAMPLE_RATE_HZ, num_frames=60, seed=7)
    calibration = Calibration(baseline_packets=50)
    calibration_result = calibrate(SyntheticSource(baseline), calibration)
    assert calibration_result.num_baseline == 60  # every observed frame, not just baseline_packets
    assert calibration_result.subcarriers == expected_result.subcarriers
    assert np.allclose(calibration_result.baseline_mag, expected_result.baseline_mag)
    assert calibration.gain_lock.locked


def test_collect_dataset_labels_windows_inside_the_spans_and_stamps_tier():
    calibration_result, gain_lock = _calibrated()
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    dataset = collect_dataset(SyntheticSource(frames), calibration_result, gain_lock, [PRESENCE_SPAN_S],
                              **_collect_options(tier="open"))
    assert dataset.X_intercarrier is None and dataset.meta["tier"] == "open"
    inside = (dataset.t >= PRESENCE_SPAN_S[0]) & (dataset.t <= PRESENCE_SPAN_S[1])
    assert set(dataset.y[inside]) == {1} and set(dataset.y[~inside]) == {0}


def test_collect_dataset_weapon_stage_ignores_the_gain_lock():
    calibration_result, gain_lock = _calibrated()
    frames = _recording(presence_spans=[(0.0, 10.0)], weapon_spans=[(2.5, 7.5)],
                        weapon_signature_depth=0.5)
    locked = collect_dataset(SyntheticSource(frames), calibration_result, gain_lock, [(2.5, 7.5)],
                             **_collect_options(stage="weapon"))
    unlocked = collect_dataset(SyntheticSource(frames), calibration_result, None, [(2.5, 7.5)],
                               **_collect_options(stage="weapon"))
    assert locked.X_intercarrier.shape[1] == 27
    assert np.array_equal(locked.X_intercarrier, unlocked.X_intercarrier)


def test_iter_window_verdicts_yields_each_window_with_its_front_end_output():
    calibration_result, gain_lock = _calibrated()
    signal_options = dict(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    dataset = collect_dataset(SyntheticSource(_recording(**signal_options)), calibration_result, gain_lock, [],
                              **_collect_options())
    verdicts = [(verdict.timestamp_s, verdict.class_id, verdict.confidence, list(verdict.proba),
                 verdict.window_view.features.copy(), verdict.window_view.image.copy())
                for verdict in iter_window_verdicts(SyntheticSource(_recording(**signal_options)),
                                                    calibration_result, gain_lock, _FixedVerdictSession(),
                                                    "presence")]
    assert [verdict[0] for verdict in verdicts] == list(dataset.t)
    assert {(class_id, confidence, tuple(proba)) for _, class_id, confidence, proba, _, _ in verdicts} == {
        (1, 0.75, (0.25, 0.75))}
    np.testing.assert_array_equal(np.stack([verdict[4] for verdict in verdicts]), dataset.X_features)
    np.testing.assert_array_equal(np.stack([verdict[5] for verdict in verdicts]), dataset.X_image)


VIOLATION_LOGGER = "wavetrace.application.behaviours.rule_violations"


def _verdict_timestamps(frames, session, *, calibrated_subcarrier_width=NUM_SUBCARRIERS, **serve_options):
    calibration_result, gain_lock = _calibrated()
    calibration_result = replace(calibration_result, subcarrier_width=calibrated_subcarrier_width)
    return [verdict.timestamp_s for verdict in iter_window_verdicts(SyntheticSource(frames), calibration_result,
                                                                    gain_lock, session, "presence",
                                                                    **serve_options)
            if verdict is not None]


def _with_wider_frames(frames, every):
    wider_frames, _ = generate_stream(num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS + 16,
                                      sample_rate_hz=SAMPLE_RATE_HZ, num_frames=len(frames) // every + 1,
                                      perturbation_hz=1.0, perturbation_depth=0.5, cfo_hz=0.0, noise_std=0.01,
                                      seed=5)
    mixed = []
    for i, frame in enumerate(frames):
        mixed.append(frame)
        if i % every == 0:
            mixed.append(wider_frames[i // every])
    return mixed


def test_a_model_scores_only_frames_of_its_trained_capture_width(caplog):
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    clean = _verdict_timestamps(frames, _FixedVerdictSession(trained_subcarrier_width=NUM_SUBCARRIERS))
    with caplog.at_level(logging.WARNING, logger=VIOLATION_LOGGER):
        mixed = _verdict_timestamps(_with_wider_frames(frames, every=50),
                                    _FixedVerdictSession(trained_subcarrier_width=NUM_SUBCARRIERS))
    assert mixed == clean and clean
    warnings = [record.getMessage() for record in caplog.records if record.name == VIOLATION_LOGGER]
    assert warnings and warnings[0] == ("SubcarrierWidthMustMatchContract violated on the served stream: "
                                        "48 subcarriers, the model was trained on 32 "
                                        "(1 since the last report, 1 in total)")


def test_a_model_of_another_capture_width_than_a_legacy_calibration_scores_nothing():
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    session = _FixedVerdictSession(trained_subcarrier_width=64)
    assert _verdict_timestamps(frames, session, calibrated_subcarrier_width=None) == []


def test_a_calibration_made_from_another_width_than_the_model_is_refused():
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    with pytest.raises(BusinessRuleViolationError,
                       match="SubcarrierWidthMustMatchContract.*32 subcarriers, the model was trained on 64"):
        _verdict_timestamps(frames, _FixedVerdictSession(trained_subcarrier_width=64))


class _LoadTrackingSource(SyntheticSource):
    def __init__(self, frames):
        super().__init__(frames)
        self.num_loads = 0

    def frames(self):
        self.num_loads += 1
        return super().frames()


@pytest.mark.parametrize("trained_subcarrier_width", [4, 64])
def test_a_refused_calibration_never_loads_the_recording(trained_subcarrier_width):
    source = _LoadTrackingSource(_recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20))
    calibration_result, gain_lock = _calibrated()
    session = _FixedVerdictSession(trained_subcarrier_width=trained_subcarrier_width)
    with pytest.raises(BusinessRuleViolationError):
        list(iter_window_verdicts(source, calibration_result, gain_lock, session, "presence"))
    assert source.num_loads == 0


def test_a_legacy_model_scores_only_frames_of_the_width_its_calibration_was_made_from(caplog):
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    clean = _verdict_timestamps(frames, _FixedVerdictSession())
    with caplog.at_level(logging.WARNING, logger=VIOLATION_LOGGER):
        mixed = _verdict_timestamps(_with_wider_frames(frames, every=50), _FixedVerdictSession())
    assert mixed == clean and clean
    warnings = [record.getMessage() for record in caplog.records if record.name == VIOLATION_LOGGER]
    assert warnings[0] == ("SubcarrierWidthMustMatchCalibration violated on the served stream: "
                           "48 subcarriers, the calibration was made from 32 "
                           "(1 since the last report, 1 in total)")


def test_a_calibration_wider_than_the_model_capture_is_refused_at_session_start():
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    with pytest.raises(BusinessRuleViolationError, match="SubcarrierWidthMustCoverCalibration"):
        _verdict_timestamps(frames, _FixedVerdictSession(trained_subcarrier_width=4))


def test_a_legacy_model_and_calibration_score_the_stream_unfiltered(caplog):
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    with caplog.at_level(logging.WARNING, logger=VIOLATION_LOGGER):
        legacy = _verdict_timestamps(frames, _FixedVerdictSession(), calibrated_subcarrier_width=None)
    assert legacy == _verdict_timestamps(frames, _FixedVerdictSession(trained_subcarrier_width=NUM_SUBCARRIERS))
    assert not [record for record in caplog.records if record.name == VIOLATION_LOGGER]


def _slow_down_from(frames, first_slow_index, factor):
    """Stretches the timestamps from `first_slow_index` on by `factor`, so the rate there is 1/factor."""
    boundary_s = frames[first_slow_index].timestamp
    for frame in frames[first_slow_index:]:
        frame.timestamp = boundary_s + factor * (frame.timestamp - boundary_s)
    return frames


def _presence_frames():
    return _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)


def test_a_window_delivered_at_the_target_rate_is_scored(caplog):
    with caplog.at_level(logging.WARNING, logger=VIOLATION_LOGGER):
        timestamps = _verdict_timestamps(_presence_frames(), _FixedVerdictSession())
    assert len(timestamps) == 61
    assert not [record for record in caplog.records if record.name == VIOLATION_LOGGER]


def test_a_window_below_the_minimum_rate_is_skipped_counted_and_logged(caplog):
    first_slow_index = 500
    session = _FixedVerdictSession()
    with caplog.at_level(logging.WARNING, logger=VIOLATION_LOGGER):
        timestamps = _verdict_timestamps(_slow_down_from(_presence_frames(), first_slow_index, 2.0), session)
    clean = _verdict_timestamps(_presence_frames(), _FixedVerdictSession())
    num_fast_windows = 30  # windows 0..29 end before the slow stretch; 30 and 31 straddle it, 32.. are slow
    assert timestamps[:num_fast_windows] == clean[:num_fast_windows]
    assert num_fast_windows <= len(timestamps) < num_fast_windows + 2
    assert session.num_forward_passes == len(timestamps)
    warnings = [record.getMessage() for record in caplog.records if record.name == VIOLATION_LOGGER]
    assert len(warnings) == 1
    assert warnings[0].startswith("SampleRateMustBeWithinTolerance violated on the served stream: ")
    assert warnings[0].endswith("Hz delivered, at least 80.0 Hz needed (1 since the last report, 1 in total)")


def test_a_stream_at_half_the_target_rate_scores_nothing_until_the_fraction_allows_it():
    frames = _slow_down_from(_presence_frames(), 0, 2.0)
    assert _verdict_timestamps(frames, _FixedVerdictSession()) == []
    assert len(_verdict_timestamps(_slow_down_from(_presence_frames(), 0, 2.0), _FixedVerdictSession(),
                                   min_sample_rate_fraction=0.4)) == 61


@pytest.mark.parametrize("frame_average, num_windows", [(1, 61), (2, 30)])
def test_the_window_rate_is_measured_over_every_frame_the_window_averages(frame_average, num_windows):
    session = _FixedVerdictSession(frame_average=frame_average)
    assert len(_verdict_timestamps(_presence_frames(), session)) == num_windows
    slow_frames = _slow_down_from(_presence_frames(), 0, 2.0)
    assert _verdict_timestamps(slow_frames, _FixedVerdictSession(frame_average=frame_average)) == []


def test_a_window_is_scored_when_only_its_last_raw_frames_are_slow_but_its_whole_span_averages_fast():
    last_fast_window = 10
    first_slow_index = last_fast_window * 2 * HOP_FRAMES + WINDOW_FRAMES
    # window 10 spans raw frames 320..383: its first 32 run at 100 Hz, its last 32 at 71 Hz, 83 Hz over all 64
    slow_frames = _slow_down_from(_presence_frames(), first_slow_index, 1.4)
    timestamps = _verdict_timestamps(slow_frames, _FixedVerdictSession(frame_average=2))
    assert len(timestamps) == last_fast_window + 1


def test_a_skipped_window_yields_none_so_the_caller_can_check_for_a_stop():
    calibration_result, gain_lock = _calibrated()
    slow_frames = _slow_down_from(_presence_frames(), 0, 2.0)
    yielded = list(iter_window_verdicts(SyntheticSource(slow_frames), calibration_result, gain_lock,
                                        _FixedVerdictSession(), "presence"))
    assert yielded and all(item is None for item in yielded)


def test_infer_windows_publishes_nothing_for_skipped_windows():
    calibration_result, gain_lock = _calibrated()
    publisher = _RecordingPublisher()
    slow_frames = _slow_down_from(_presence_frames(), 0, 2.0)
    assert infer_windows(SyntheticSource(slow_frames), calibration_result, gain_lock, _FixedVerdictSession(),
                         "presence", publisher) == []
    assert publisher.results == []


@pytest.mark.parametrize("fraction", [0.0, 1.1])
def test_a_minimum_rate_fraction_outside_zero_to_one_is_refused(fraction):
    with pytest.raises(ValueError, match="min_sample_rate_fraction must be in"):
        _verdict_timestamps(_presence_frames(), _FixedVerdictSession(), min_sample_rate_fraction=fraction)


@pytest.mark.parametrize("vote, extra_rows", [(False, 0), (True, 1)])
def test_infer_windows_publishes_one_verdict_per_window_plus_the_vote(vote, extra_rows):
    calibration_result, gain_lock = _calibrated()
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    num_windows = collect_dataset(SyntheticSource(frames), calibration_result, gain_lock, [],
                                  **_collect_options()).y.size
    publisher = _RecordingPublisher()
    session = _FixedVerdictSession()
    behaviours = [SegmentVoteBehaviour(SegmentVoter())] if vote else []
    results = infer_windows(SyntheticSource(frames), calibration_result, gain_lock,
                            session, "presence", publisher, behaviours=behaviours)
    assert len(results) == len(publisher.results) == num_windows + extra_rows
    assert session.num_forward_passes == num_windows
    assert all(result.class_id == 1 for result in results)
    if vote:
        assert results[-1].confidence == pytest.approx(0.75)


def test_infer_windows_with_the_guard_fires_an_alert_and_keeps_the_verdicts():
    calibration_result, gain_lock = _calibrated()
    frames = _recording(presence_spans=[PRESENCE_SPAN_S], presence_turbulence_std=0.20)
    unguarded = infer_windows(SyntheticSource(frames), calibration_result, gain_lock,
                              _FixedVerdictSession(), "presence", _RecordingPublisher())
    publisher = _RecordingPublisher()
    guarded = infer_windows(SyntheticSource(frames), calibration_result, gain_lock,
                            _FixedVerdictSession(), "presence", publisher,
                            behaviours=[AlertGuardBehaviour(AlertGuard())])
    assert len(guarded) == len(unguarded)
    assert any(isinstance(event, WeaponAlertEvent) for event in publisher.events)


@pytest.mark.parametrize("use_tracker", [False, True])
def test_localize_publishes_one_fix_per_frame_and_returns_the_room_map(use_tracker):
    frames, _ = generate_stream(num_antennas=2, num_subcarriers=32, sample_rate_hz=100.0, num_frames=20,
                                perturbation_hz=1.0, perturbation_depth=0.3, cfo_hz=10.0,
                                noise_std=0.01, seed=3)
    publisher = _RecordingPublisher()
    room_map = localize(SyntheticSource(frames), Localizer(2), Tracker() if use_tracker else None, publisher)
    fixes = Localizer(2).locate_stream(frames)
    if use_tracker:
        tracker = Tracker()
        states = [tracker.update(fix) for fix in fixes]
        expected = [(state.confidence, state.measured, state.x_m, state.y_m, state.angle_deg) for state in states]
    else:
        expected = [(fix.confidence, fix.confidence >= 0.5, fix.x_m, fix.y_m, fix.peak_angle_deg) for fix in fixes]
    assert len(publisher.results) == len(expected)
    for result, (confidence, is_located, x_m, y_m, angle_deg) in zip(publisher.results, expected):
        assert (result.class_id == 1) == is_located
        assert result.confidence == pytest.approx(confidence, abs=1e-4)
        assert list(result.bbox[:2]) == pytest.approx([x_m, y_m], abs=1e-4)
        assert result.keypoints[0] == pytest.approx(angle_deg, abs=1e-4)
    assert all(len(result.bbox) == 4 and len(result.keypoints) == 2 for result in publisher.results)
    assert room_map.heatmap.shape == (room_map.ranges_m.size, room_map.angles_deg.size)


def test_localize_without_a_publisher_still_returns_the_room_map():
    frames, _ = generate_stream(num_antennas=2, num_subcarriers=32, sample_rate_hz=100.0, num_frames=20,
                                perturbation_hz=1.0, perturbation_depth=0.3, cfo_hz=10.0,
                                noise_std=0.01, seed=3)
    room_map = localize(SyntheticSource(frames), Localizer(2), Tracker(), None)
    assert room_map.angles_deg.size == 181
