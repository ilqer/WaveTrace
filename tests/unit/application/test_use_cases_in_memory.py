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
from wavetrace.domain.guard import AlertGuard, WeaponAlertEvent
from wavetrace.domain.localize import Localizer
from wavetrace.domain.model_config import ModelConfig
from wavetrace.domain.recognition.vote import SegmentVoter
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
    """Always answers class 1 with probability 0.75."""

    def __init__(self):
        self.num_forward_passes = 0
        self.head = SimpleNamespace(
            config=ModelConfig(stage="presence", k=4, window=WINDOW_FRAMES, hop=HOP_FRAMES),
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
