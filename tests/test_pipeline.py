"""End-to-end: capture/recording -> calibrate -> collect-data -> train -> run -> publish.

Parity: the serving pipeline (iter_windows) must produce byte-identical features to
training (build_dataset).
"""

import io
import json

import numpy as np
import pytest

from fixtures.synthetic_csi import generate_quiet_baseline
from wavetrace.synthetic import generate_paired_recording
from wavetrace import CsiFrame, Label, RecognitionResult
from wavetrace.calibration import Calibration, load_calibration, save_calibration
from wavetrace.application.calibrate import calibrate_source
from wavetrace.application.collect import collect_source
from wavetrace.application.serve import run_inference
from wavetrace.frontend import iter_windows
from wavetrace.adapters.source import RecordingSource, SyntheticSource, load_recording, save_recording
from wavetrace.groundtruth import build_dataset
from wavetrace.groundtruth.camera_labeler import ScriptedLabeler
from wavetrace.output import JsonlPublisher, result_to_dict
from wavetrace.recognition import (
    InferenceSession,
    measure_latency,
    plan_input,
    train_presence,
    train_weapon,
)

NUM_ANTENNAS, NUM_SUBCARRIERS, SAMPLE_RATE_HZ = 2, 32, 100.0


def _baseline():
    return generate_quiet_baseline(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        num_frames=60, seed=7)


def _weapon_recording(seed=200, duration_s=10.0):
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        duration_s=duration_s, camera_frame_rate_hz=30.0, presence_spans=[(0.0, duration_s)],
        presence_turbulence_std=0.10,
        weapon_spans=[(2.5, 7.5)], weapon_signature_depth=0.5, seed=seed)
    return frames


def _presence_recording(seed=300, duration=10.0):
    # Turbulence only inside the presence span. Windows outside are 'absent'.
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        duration_s=duration, camera_frame_rate_hz=30.0, presence_spans=[(3.0, 7.0)],
        presence_turbulence_std=0.20,
        amplitude_hz=2.0, amplitude_depth=0.45, seed=seed)
    return frames


def test_recording_roundtrip(tmp_path):
    frames = _weapon_recording(duration_s=2.0)
    save_recording(frames, tmp_path / "rec")
    loaded_frames = list(load_recording(tmp_path / "rec"))
    assert len(loaded_frames) == len(frames)
    assert all(np.allclose(np.asarray(original.grid), np.asarray(loaded.grid))
               for original, loaded in zip(frames, loaded_frames))
    assert all(original.timestamp == loaded.timestamp for original, loaded in zip(frames, loaded_frames))
    assert sum(1 for _ in RecordingSource(tmp_path / "rec").frames()) == len(frames)


def test_calibration_roundtrip_rebuilds_lock(tmp_path):
    calibration = Calibration(baseline_packets=50)
    for frame in _baseline():
        calibration.observe(frame)
    calibration_result = calibration.finalize()
    save_calibration(calibration_result, tmp_path / "cal")
    reloaded_result, reloaded_gain_lock = load_calibration(tmp_path / "cal")
    assert calibration_result.subcarriers == reloaded_result.subcarriers
    assert np.allclose(calibration_result.baseline_mag, reloaded_result.baseline_mag)
    assert reloaded_gain_lock is not None and reloaded_gain_lock.locked and (
        reloaded_gain_lock.reference_scale == calibration_result.reference_scale)
    # Rebuilt gain lock applies identically to the original.
    original_frame = _weapon_recording(duration_s=0.2)[0]
    copied_frame = CsiFrame(NUM_ANTENNAS, NUM_SUBCARRIERS)
    copied_frame.timestamp = original_frame.timestamp
    copied_frame.grid[:, :] = np.asarray(original_frame.grid)
    calibration.gain_lock.apply(original_frame)
    reloaded_gain_lock.apply(copied_frame)
    assert np.allclose(np.asarray(original_frame.grid), np.asarray(copied_frame.grid))


def test_calibration_disabled_lock_roundtrips_to_none(tmp_path):
    calibration = Calibration(baseline_packets=50, use_gain_lock=False)
    for frame in _baseline():
        calibration.observe(frame)
    save_calibration(calibration.finalize(), tmp_path / "cal")
    _, gain_lock = load_calibration(tmp_path / "cal")
    assert gain_lock is None


def test_run_features_match_build_dataset(tmp_path):
    """iter_windows (serving) must yield the SAME features build_dataset (training) stores."""
    frames = _weapon_recording(duration_s=4.0)
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    result, gain_lock = load_calibration(tmp_path / "cal")
    # Training arrays: using raw mags, gain_lock=None.
    dataset = build_dataset(frames, result, None, ScriptedLabeler([(2.5, 7.5, True)]),
                            window=32, hop=16, intercarrier=True)
    window_features, window_intercarrier_features = [], []
    for _, features, _image, intercarrier_features in iter_windows(
            frames, result.subcarriers, None, window=32, hop=16, intercarrier=True):
        window_features.append(features.copy())
        window_intercarrier_features.append(intercarrier_features.copy())
    assert np.allclose(np.stack(window_features), dataset.X_features)
    assert np.allclose(np.stack(window_intercarrier_features), dataset.X_intercarrier)


class _FakeHead:
    def __init__(self, backend, feature_mode=None):
        from wavetrace.config import ModelConfig
        self.config = ModelConfig(stage="weapon", k=12, backend=backend)
        self.feature_mode = feature_mode


def test_serving_plan_table():
    features = np.arange(3.0)
    image = np.zeros((2, 2))
    intercarrier_features = np.arange(5.0)
    should_apply_lock, needs_intercarrier, pick_row = plan_input("presence", _FakeHead("mlp"))
    assert (should_apply_lock, needs_intercarrier) == (True, False) and np.array_equal(
        pick_row(features, image, intercarrier_features), features)
    should_apply_lock, needs_intercarrier, pick_row = plan_input("weapon", _FakeHead("variance", "ic27"))
    assert (should_apply_lock, needs_intercarrier) == (False, True) and np.array_equal(
        pick_row(features, image, intercarrier_features), intercarrier_features)
    should_apply_lock, needs_intercarrier, pick_row = plan_input("weapon", _FakeHead("mlp", "fusion"))
    assert (should_apply_lock, needs_intercarrier) == (True, True) and np.array_equal(
        pick_row(features, image, intercarrier_features), np.hstack([intercarrier_features, features]))
    should_apply_lock, needs_intercarrier, pick_row = plan_input("weapon", _FakeHead("cnn", "cnn"))
    assert (should_apply_lock, needs_intercarrier) == (False, False) and np.array_equal(
        pick_row(features, image, intercarrier_features), image.reshape(-1))


def test_end_to_end_weapon(tmp_path):
    frames = _weapon_recording()
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    _, dataset = collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds",
                                [(2.5, 7.5)], stage="weapon", window=32, hop=16,
                                session_id="s0", subject_id="u0")
    assert dataset.X_intercarrier.shape[1] == 27
    train_weapon([tmp_path / "ds"], out_dir=tmp_path / "m", feature_mode="ic27")

    buffer = io.StringIO()
    with JsonlPublisher(buffer, mode="weapon") as pub:
        results = run_inference(SyntheticSource(frames), tmp_path / "cal",
                                tmp_path / "m" / "model.joblib", "weapon", pub)
    lines = [json.loads(line) for line in buffer.getvalue().strip().split("\n")]
    # N windows generates N valid lines. Serving window count must equal dataset sample count.
    assert len(lines) == len(results) == dataset.y.size
    assert all(set(line) == {"t", "class", "conf", "mode", "bbox", "keypoints"} for line in lines)
    # 2.5..7.5 s is the recording's weapon span; windows outside it are the clear room
    inside = [line["class"] for line in lines if 2.5 <= line["t"] < 7.5]
    outside = [line["class"] for line in lines if line["t"] < 2.5 or line["t"] >= 7.5]
    assert np.mean(inside) > 0.6 and np.mean(outside) < 0.25


def test_collect_with_camera_labeler_persists_mask_and_tier(tmp_path):
    frames = _weapon_recording(duration_s=4.0)
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    grid = 4

    def camera_label(timestamp_s):
        present = 2.5 <= timestamp_s <= 7.5
        label = Label()
        label.class_id = 1 if present else 0
        label.name = "weapon" if present else "background"
        label.timestamp = float(timestamp_s)
        label.mask = [1.0 if present else 0.0] * (grid * grid)
        label.mask_grid = grid
        return label

    _, dataset = collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds", [],
                                stage="weapon", window=32, hop=16, labeler=camera_label, tier="concealed")
    from wavetrace.groundtruth import load_dataset
    reloaded = load_dataset(tmp_path / "ds")
    assert reloaded.meta["tier"] == "concealed"
    assert reloaded.y.size > 0 and all(label.mask_grid == grid for label in reloaded.labels)
    assert all(len(label.mask) == grid * grid for label in reloaded.labels)


def test_end_to_end_presence(tmp_path):
    frames = _presence_recording()
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    _, dataset = collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds",
                                [(3.0, 7.0)], stage="presence", window=32, hop=16,
                                session_id="s0", subject_id="u0")
    assert dataset.X_intercarrier is None and set(np.unique(dataset.y)) == {0, 1}
    train_presence([tmp_path / "ds"], out_dir=tmp_path / "m")
    buffer = io.StringIO()
    with JsonlPublisher(buffer, mode="presence") as pub:
        results = run_inference(SyntheticSource(frames), tmp_path / "cal",
                                tmp_path / "m" / "model.joblib", "presence", pub)
    assert len(results) == dataset.y.size
    lines = [json.loads(line) for line in buffer.getvalue().strip().split("\n")]
    assert all(line["mode"] == "presence" for line in lines)


def test_run_vote_appends_segment_verdict(tmp_path):
    frames = _weapon_recording()
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds", [(2.5, 7.5)],
                   stage="weapon", window=32, hop=16, session_id="s0", subject_id="u0")
    train_weapon([tmp_path / "ds"], out_dir=tmp_path / "m", feature_mode="ic27")
    plain = run_inference(SyntheticSource(frames), tmp_path / "cal", tmp_path / "m" / "model.joblib",
                          "weapon", JsonlPublisher(io.StringIO()))
    voted = run_inference(SyntheticSource(frames), tmp_path / "cal", tmp_path / "m" / "model.joblib",
                          "weapon", JsonlPublisher(io.StringIO()), vote=True)
    assert len(voted) == len(plain) + 1  # one extra soft-vote verdict at the end


def test_inference_latency_under_8ms(tmp_path):
    frames = _weapon_recording(duration_s=4.0)
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds", [(2.5, 7.5)],
                   stage="weapon", window=32, hop=16)
    head, _ = train_weapon([tmp_path / "ds"], out_dir=tmp_path / "m", feature_mode="ic27")
    session = InferenceSession(head=head)
    stats = measure_latency(session, np.zeros(27, dtype=np.float32))
    assert stats["max_ms"] < 8.0


def test_result_to_dict_schema():
    recognition_result = RecognitionResult()
    recognition_result.class_id = 1
    recognition_result.confidence = 0.9
    recognition_result.timestamp = 1.2
    result_dict = result_to_dict(recognition_result, mode="weapon")
    assert result_dict == {"t": 1.2, "class": 1, "conf": pytest.approx(0.9), "mode": "weapon",
                           "bbox": None, "keypoints": []}
