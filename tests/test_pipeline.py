"""End-to-end: capture/recording -> calibrate -> collect-data -> train -> run -> publish.

Parity: the serving pipeline (iter_windows) must produce byte-identical features to
training (build_dataset).
"""

import io
import json

import numpy as np
import pytest

from wavetrace.synthetic import generate_stream
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

NUM_ANT, NUM_SUB, FS = 2, 32, 100.0


def _baseline():
    frames, _ = generate_stream(num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS,
                               num_frames=60, perturbation_hz=0.0, perturbation_depth=0.0, cfo_hz=0.0,
                               noise_std=0.005, seed=7)
    return frames


def _weapon_recording(seed=200, duration=10.0):
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=duration,
        camera_fps=30.0, presence_spans=[(0.0, duration)], presence_turbulence_std=0.10,
        weapon_spans=[(2.5, 7.5)], weapon_signature_depth=0.5, seed=seed)
    return frames


def _presence_recording(seed=300, duration=10.0):
    # Turbulence only inside the presence span. Windows outside are 'absent'.
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=duration,
        camera_fps=30.0, presence_spans=[(3.0, 7.0)], presence_turbulence_std=0.20,
        amplitude_hz=2.0, amplitude_depth=0.45, seed=seed)
    return frames


def test_recording_roundtrip(tmp_path):
    frames = _weapon_recording(duration=2.0)
    save_recording(frames, tmp_path / "rec")
    rec = list(load_recording(tmp_path / "rec"))
    assert len(rec) == len(frames)
    assert all(np.allclose(np.asarray(a.grid), np.asarray(b.grid)) for a, b in zip(frames, rec))
    assert all(a.timestamp == b.timestamp for a, b in zip(frames, rec))
    assert sum(1 for _ in RecordingSource(tmp_path / "rec").frames()) == len(frames)


def test_calibration_roundtrip_rebuilds_lock(tmp_path):
    cal = Calibration(baseline_packets=50)
    for fr in _baseline():
        cal.observe(fr)
    res = cal.finalize()
    save_calibration(res, tmp_path / "cal")
    res2, gl2 = load_calibration(tmp_path / "cal")
    assert res.subcarriers == res2.subcarriers
    assert np.allclose(res.baseline_mag, res2.baseline_mag)
    assert gl2 is not None and gl2.locked and gl2.reference_scale == res.reference_scale
    # Rebuilt gain lock applies identically to the original.
    a = _weapon_recording(duration=0.2)[0]
    b = CsiFrame(NUM_ANT, NUM_SUB); b.timestamp = a.timestamp; b.grid[:, :] = np.asarray(a.grid)
    cal.gain_lock.apply(a); gl2.apply(b)
    assert np.allclose(np.asarray(a.grid), np.asarray(b.grid))


def test_calibration_disabled_lock_roundtrips_to_none(tmp_path):
    cal = Calibration(baseline_packets=50, use_gain_lock=False)
    for fr in _baseline():
        cal.observe(fr)
    save_calibration(cal.finalize(), tmp_path / "cal")
    _, gl = load_calibration(tmp_path / "cal")
    assert gl is None


def test_run_features_match_build_dataset(tmp_path):
    """iter_windows (serving) must yield the SAME features build_dataset (training) stores."""
    frames = _weapon_recording(duration=4.0)
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    result, gain_lock = load_calibration(tmp_path / "cal")
    # Training arrays: using raw mags, gain_lock=None.
    ds = build_dataset(frames, result, None, ScriptedLabeler([(2.5, 7.5, True)]),
                       window=32, hop=16, intercarrier=True)
    feats, ics = [], []
    for _, f, _img, ic in iter_windows(frames, result.subcarriers, None, window=32, hop=16,
                                       intercarrier=True):
        feats.append(f.copy()); ics.append(ic.copy())
    assert np.allclose(np.stack(feats), ds.X_features)
    assert np.allclose(np.stack(ics), ds.X_intercarrier)


class _FakeHead:
    def __init__(self, backend, feature_mode=None):
        from wavetrace.config import ModelConfig
        self.config = ModelConfig(stage="weapon", k=12, backend=backend)
        self.feature_mode = feature_mode


def test_serving_plan_table():
    f = np.arange(3.0); i = np.zeros((2, 2)); ic = np.arange(5.0)
    lock, inter, pick = plan_input("presence", _FakeHead("mlp"))
    assert (lock, inter) == (True, False) and np.array_equal(pick(f, i, ic), f)
    lock, inter, pick = plan_input("weapon", _FakeHead("variance", "ic27"))
    assert (lock, inter) == (False, True) and np.array_equal(pick(f, i, ic), ic)
    lock, inter, pick = plan_input("weapon", _FakeHead("mlp", "fusion"))
    assert (lock, inter) == (True, True) and np.array_equal(pick(f, i, ic), np.hstack([ic, f]))
    lock, inter, pick = plan_input("weapon", _FakeHead("cnn", "cnn"))
    assert (lock, inter) == (False, False) and np.array_equal(pick(f, i, ic), i.reshape(-1))


def test_end_to_end_weapon(tmp_path):
    frames = _weapon_recording()
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    _, ds = collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds",
                           [(2.5, 7.5)], stage="weapon", window=32, hop=16,
                           session_id="s0", subject_id="u0")
    assert ds.X_intercarrier.shape[1] == 27
    train_weapon([tmp_path / "ds"], out_dir=tmp_path / "m", feature_mode="ic27")

    buf = io.StringIO()
    with JsonlPublisher(buf, mode="weapon") as pub:
        results = run_inference(SyntheticSource(frames), tmp_path / "cal",
                                tmp_path / "m" / "model.joblib", "weapon", pub)
    lines = [json.loads(l) for l in buf.getvalue().strip().split("\n")]
    # N windows generates N valid lines. Serving window count must equal dataset sample count.
    assert len(lines) == len(results) == ds.y.size
    assert all(set(l) == {"t", "class", "conf", "mode", "bbox", "keypoints"} for l in lines)
    # Class verdicts must separate weapon span from empty room windows.
    inside = [l["class"] for l in lines if 2.5 <= l["t"] < 7.5]
    outside = [l["class"] for l in lines if l["t"] < 2.5 or l["t"] >= 7.5]
    assert np.mean(inside) > 0.6 and np.mean(outside) < 0.25


def test_collect_with_camera_labeler_persists_mask_and_tier(tmp_path):
    frames = _weapon_recording(duration=4.0)
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    grid = 4

    def camera_label(t):
        present = 2.5 <= t <= 7.5
        lab = Label()
        lab.class_id = 1 if present else 0
        lab.name = "weapon" if present else "background"
        lab.timestamp = float(t)
        lab.mask = [1.0 if present else 0.0] * (grid * grid)
        lab.mask_grid = grid
        return lab

    _, ds = collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds", [],
                           stage="weapon", window=32, hop=16, labeler=camera_label, tier="concealed")
    from wavetrace.groundtruth import load_dataset
    reloaded = load_dataset(tmp_path / "ds")
    assert reloaded.meta["tier"] == "concealed"
    assert reloaded.y.size > 0 and all(l.mask_grid == grid for l in reloaded.labels)
    assert all(len(l.mask) == grid * grid for l in reloaded.labels)


def test_end_to_end_presence(tmp_path):
    frames = _presence_recording()
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    _, ds = collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds",
                           [(3.0, 7.0)], stage="presence", window=32, hop=16,
                           session_id="s0", subject_id="u0")
    assert ds.X_intercarrier is None and set(np.unique(ds.y)) == {0, 1}
    train_presence([tmp_path / "ds"], out_dir=tmp_path / "m")
    buf = io.StringIO()
    with JsonlPublisher(buf, mode="presence") as pub:
        results = run_inference(SyntheticSource(frames), tmp_path / "cal",
                                tmp_path / "m" / "model.joblib", "presence", pub)
    assert len(results) == ds.y.size
    lines = [json.loads(l) for l in buf.getvalue().strip().split("\n")]
    assert all(l["mode"] == "presence" for l in lines)


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
    frames = _weapon_recording(duration=4.0)
    calibrate_source(SyntheticSource(_baseline()), tmp_path / "cal", baseline_packets=50)
    collect_source(SyntheticSource(frames), tmp_path / "cal", tmp_path / "ds", [(2.5, 7.5)],
                   stage="weapon", window=32, hop=16)
    head, _ = train_weapon([tmp_path / "ds"], out_dir=tmp_path / "m", feature_mode="ic27")
    session = InferenceSession(head=head)
    stats = measure_latency(session, np.zeros(27, dtype=np.float32))
    assert stats["max_ms"] < 8.0


def test_result_to_dict_schema():
    r = RecognitionResult(); r.class_id = 1; r.confidence = 0.9; r.timestamp = 1.2
    d = result_to_dict(r, mode="weapon")
    assert d == {"t": 1.2, "class": 1, "conf": pytest.approx(0.9), "mode": "weapon",
                 "bbox": None, "keypoints": []}
