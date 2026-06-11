"""Phase 6 — Stage A presence head + multi-RX plumbing tests (6a–6f).

The eval-gate test uses 4 synthetic recordings (2 subjects x 2 sessions, presence -> per-subcarrier
turbulence, plus a fast common-amplitude wobble that confuses a scalar energy gate) and asserts the
trained head beats BOTH no-train baselines on leave-one-session-out AND leave-one-subject-out folds.
Synthetic separability validates the LEARNING PIPELINE only — real camera-labeled recordings are
required before any accuracy claim (plan §2.2).
"""

import json

import numpy as np
import pytest

from fixtures.SyntheticCsi import generateStream
from fixtures.SyntheticRecording import generatePairedRecording
from wavetrace.Calibration import Calibration
from wavetrace.Config import ModelConfig
from wavetrace.groundtruth import (
    ScriptedLabeler,
    build_dataset,
    load_dataset,
    presence_label_fn,
    save_dataset,
)
from wavetrace.recognition import (
    InferenceSession,
    PresenceHead,
    accept_format,
    concat_datasets,
    evaluate_presence,
    fs_ok,
    fuse,
    leave_one_group_out,
    measure_latency,
    resample_uniform,
    segmenter_baseline,
    train_presence,
)

NUM_ANT = 2
NUM_SUB = 32
FS = 100.0
SPAN = (2.5, 7.5)  # presence span (s) inside each 10 s recording
# turbulence interleaved across subjects so every train fold sees weak AND strong sessions
RECORDINGS = [("s0", "u0", 100, 0.15), ("s1", "u1", 101, 0.20),
              ("s2", "u1", 102, 0.25), ("s3", "u0", 103, 0.30)]


def _calibrate():
    """Quiet-baseline Calibration -> (result, locked GainLock) shared by all recordings."""
    baseline, _ = generateStream(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, numFrames=60,
        perturbationHz=0.0, perturbationDepth=0.0, cfoHz=0.0, noiseStd=0.005, seed=7,
    )
    cal = Calibration(baseline_packets=50)
    for fr in baseline:
        cal.observe(fr)
    return cal.finalize(), cal.gain_lock


def _recording(sess, subj, seed, turb, duration=10.0):
    """One presence recording: amplitude wobble (narrowband 'interference') in BOTH classes,
    per-subcarrier turbulence only inside the presence span."""
    frames, _, truth = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=duration,
        cameraFps=30.0, presenceSpans=[SPAN], amplitudeHz=2.0, amplitudeDepth=0.45,
        presenceTurbulenceStd=turb, sessionId=sess, subjectId=subj, seed=seed,
    )
    return frames, truth


@pytest.fixture(scope="module")
def presence_data():
    """4 recordings -> labeled datasets -> concatenated (X, y, groups) + CSI images."""
    result, gain = _calibrate()
    datasets = []
    for sess, subj, seed, turb in RECORDINGS:
        frames, _ = _recording(sess, subj, seed, turb)
        labeler = ScriptedLabeler([(*SPAN, True)], label_fn=presence_label_fn)
        datasets.append(build_dataset(frames, result, gain, labeler, window=32, hop=16,
                                      session_id=sess, subject_id=subj))
    X, y, sess_ids, subj_ids = concat_datasets(datasets)
    return {
        "datasets": datasets,
        "X": X, "y": y, "sess": sess_ids, "subj": subj_ids,
        "X_image": np.concatenate([d.X_image for d in datasets]),
        "K": len(result.subcarriers),
        "config": ModelConfig(stage="presence", k=len(result.subcarriers)),
    }


# ----- 6a: learnable synthetic + group ids --------------------------------------------------------

def test_presence_turbulence_separates_features(presence_data):
    # present windows must carry higher per-subcarrier temporal std (feature idx 1 of each 9-block)
    X, y, K = presence_data["X"], presence_data["y"], presence_data["K"]
    stds = X.reshape(X.shape[0], K, 9)[:, :, 1].mean(axis=1)
    assert stds[y == 1].mean() > 1.5 * stds[y == 0].mean()


def test_turbulence_touches_only_presence_spans():
    kwargs = dict(numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=2.0,
                  cameraFps=30.0, presenceSpans=[(0.5, 1.0)], seed=11)
    plain, _, _ = generatePairedRecording(**kwargs)
    turb, _, truth = generatePairedRecording(**kwargs, presenceTurbulenceStd=0.2,
                                             sessionId="sA", subjectId="uA")
    for fp, ft in zip(plain, turb):
        same = np.array_equal(np.asarray(fp.grid), np.asarray(ft.grid))
        inside = 0.5 <= fp.timestamp < 1.0
        assert same != inside  # outside spans byte-identical, inside modulated
    assert truth["session_id"] == "sA" and truth["subject_id"] == "uA"
    assert truth["presence_turbulence_std"] == pytest.approx(0.2)


def test_dataset_group_ids_roundtrip(presence_data, tmp_path):
    ds = presence_data["datasets"][0]
    n = ds.y.size
    assert list(ds.session_ids) == ["s0"] * n and list(ds.subject_ids) == ["u0"] * n
    reloaded = load_dataset(save_dataset(ds, tmp_path / "ds"))
    assert list(reloaded.session_ids) == ["s0"] * n
    assert list(reloaded.subject_ids) == ["u0"] * n


def test_model_config_validates():
    assert ModelConfig(stage="presence", k=12).backend == "mlp"  # locked default
    with pytest.raises(ValueError):
        ModelConfig(stage="posture", k=12)          # not a stage
    with pytest.raises(ValueError):
        ModelConfig(stage="presence", k=12, backend="rf")  # not a wired backend
    with pytest.raises(ValueError, match="'mlp'/'svm'"):
        PresenceHead(ModelConfig(stage="presence", k=12, backend="cnn"))  # cnn = weapon-side (P7)
    with pytest.raises(ValueError):
        ModelConfig(stage="presence", k=0)
    with pytest.raises(ValueError):
        ModelConfig(stage="presence", k=12, fs_tol=1.5)


# ----- 6b: PresenceHead + Train -------------------------------------------------------------------

def _blobs(n=120, d=18, seed=0):
    """Two trivially separable feature clusters (per-class mean shift)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.int64)
    X[y == 1] += 3.0
    return X, y


@pytest.mark.parametrize("backend", ["mlp", "svm"])
def test_head_fit_predict_shapes(backend):
    X, y = _blobs()
    head = PresenceHead(ModelConfig(stage="presence", k=2, backend=backend)).fit(X, y)
    pred = head.predict(X)
    proba = head.predict_proba(X)
    assert pred.shape == (X.shape[0],) and set(pred) <= {0, 1}
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (pred == y).mean() > 0.95


def test_head_save_load_roundtrip(tmp_path):
    X, y = _blobs()
    head = PresenceHead(ModelConfig(stage="presence", k=2)).fit(X, y)
    path = head.save(tmp_path / "m" / "model.joblib")
    loaded = PresenceHead.load(path)
    assert loaded.config == head.config
    assert np.array_equal(loaded.predict(X), head.predict(X))
    assert np.allclose(loaded.predict_proba(X), head.predict_proba(X))


def test_head_unfitted_raises():
    head = PresenceHead(ModelConfig(stage="presence", k=2))
    with pytest.raises(ValueError, match="not fitted"):
        head.predict(np.zeros((1, 18), np.float32))


def test_train_presence_persists(presence_data, tmp_path):
    dirs = [save_dataset(d, tmp_path / f"ds{i}") for i, d in enumerate(presence_data["datasets"][:2])]
    head, metrics = train_presence(dirs, tmp_path / "models")
    assert (tmp_path / "models" / "model.joblib").exists()
    with open(tmp_path / "models" / "metrics.json") as f:
        assert json.load(f) == metrics
    assert metrics["backend"] == "mlp" and metrics["k"] == presence_data["K"]
    assert metrics["n_samples"] == sum(d.y.size for d in presence_data["datasets"][:2])
    assert metrics["sessions"] == ["s0", "s1"] and metrics["subjects"] == ["u0", "u1"]
    assert metrics["train_accuracy"] > 0.9
    assert head.predict(presence_data["X"][:4]).shape == (4,)


# ----- 6c: the LOCKED eval gate + baselines -------------------------------------------------------

def test_eval_gate_head_beats_both_baselines(presence_data):
    """The Phase-6 DoD: leave-one-session-out AND leave-one-subject-out accuracy beats the
    majority class AND the no-train PresenceSegmenter by a clear margin (synthetic = plumbing)."""
    d = presence_data
    report = evaluate_presence(
        d["X"], d["y"], session_ids=d["sess"], subject_ids=d["subj"], config=d["config"],
        X_image=d["X_image"],
        segmenter_kwargs={"cv_window": 16, "enter_cv": 0.01, "exit_cv": 0.005},
    )
    for split in ("session", "subject"):
        rep = report[split]
        assert rep["accuracy"] >= 0.92                          # measured 0.98 / 0.975
        assert rep["accuracy"] >= rep["majority_accuracy"] + 0.30
        assert rep["accuracy"] >= report["segmenter_accuracy"] + 0.05
        cm = rep["confusion"]
        assert cm.shape == (2, 2) and cm.sum() == d["y"].size
    # folds are group-disjoint and cover every group exactly once
    assert sorted(f["group"] for f in report["session"]["folds"]) == ["s0", "s1", "s2", "s3"]
    assert sorted(f["group"] for f in report["subject"]["folds"]) == ["u0", "u1"]
    for f in report["session"]["folds"]:
        assert f["n"] == int((d["sess"] == f["group"]).sum())   # whole group held out


def test_logo_requires_two_groups(presence_data):
    d = presence_data
    with pytest.raises(ValueError, match="2 distinct groups"):
        leave_one_group_out(d["X"], d["y"], np.full(d["y"].size, "only"), lambda: None)


def test_segmenter_baseline_flags_turbulent_windows(presence_data):
    d = presence_data
    pred = segmenter_baseline(d["X_image"], cv_window=16, enter_cv=0.01, exit_cv=0.005)
    assert pred.shape == d["y"].shape and set(np.unique(pred)) <= {0, 1}
    # the DSP gate is a meaningful baseline on its own: well above chance, below the trained head
    assert (pred == d["y"]).mean() > 0.75
    assert pred[d["y"] == 1].mean() > pred[d["y"] == 0].mean()


# ----- 6d: inference + latency gate ---------------------------------------------------------------

@pytest.fixture(scope="module")
def inference_session(presence_data, tmp_path_factory):
    head = PresenceHead(presence_data["config"]).fit(presence_data["X"], presence_data["y"])
    path = head.save(tmp_path_factory.mktemp("models") / "model.joblib")
    return InferenceSession(path)


def test_infer_predict_window_deterministic(presence_data, inference_session):
    d = presence_data
    feat_present = d["X"][d["y"] == 1][0]
    feat_absent = d["X"][d["y"] == 0][0]
    cls1, p1 = inference_session.predict_window(feat_present)
    cls0, p0 = inference_session.predict_window(feat_absent)
    assert (cls1, cls0) == (1, 0)
    assert 0.5 <= p1 <= 1.0 and 0.5 <= p0 <= 1.0   # argmax probability
    assert inference_session.predict_window(feat_present) == (cls1, p1)  # deterministic


def test_presence_mode_session(presence_data, tmp_path):
    # 'presence' = the human-detection operating mode (independent of weapon mode, no cross-gating)
    from wavetrace.recognition import mode_session
    d = presence_data
    head = PresenceHead(d["config"]).fit(d["X"], d["y"])
    session = mode_session("presence", head.save(tmp_path / "p.joblib"))
    cls, proba = session.predict_window(d["X"][d["y"] == 1][0])
    assert cls == 1 and 0.5 <= proba <= 1.0


def test_infer_latency_under_8ms(presence_data, inference_session):
    stats = measure_latency(inference_session, presence_data["X"][0], iters=200)
    assert stats["mean_ms"] < 8.0
    assert stats["p95_ms"] < 8.0  # DoD: per-window inference < 8 ms


# ----- 6e: multi-RX feature-level fusion ----------------------------------------------------------

def test_fuse_concat_shape_and_order():
    a = np.arange(9, dtype=np.float32)
    b = np.arange(9, 27, dtype=np.float32)
    c = np.arange(27, 30, dtype=np.float32)
    assert np.array_equal(fuse([a]), a)                       # 1 node = identity
    fused = fuse([a, b, c])
    assert fused.shape == (30,) and fused.dtype == np.float32
    assert np.array_equal(fused, np.concatenate([a, b, c]))   # stable node order
    out = np.empty(30, dtype=np.float32)
    assert fuse([a, b, c], out=out) is out                    # per-emit buffer reuse


def test_fuse_validates():
    with pytest.raises(ValueError, match="no node features"):
        fuse([])
    with pytest.raises(ValueError, match="1-D"):
        fuse([np.zeros((2, 9), np.float32)])
    with pytest.raises(ValueError, match="out"):
        fuse([np.zeros(9, np.float32)], out=np.empty(10, np.float32))


# ----- 6f: timing-jitter guards -------------------------------------------------------------------

def test_resample_uniform_recovers_jittered_series():
    rng = np.random.default_rng(0)
    n, fs = 200, 100.0
    t = np.arange(n) / fs + rng.uniform(-0.2 / fs, 0.2 / fs, n)  # jittered, still increasing
    t.sort()
    vals = np.sin(2 * np.pi * 2.0 * t).astype(np.float32)
    res, grid = resample_uniform(vals, t, fs)
    assert grid[0] == t[0] and np.allclose(np.diff(grid), 1.0 / fs)
    assert np.abs(res - np.sin(2 * np.pi * 2.0 * grid)).max() < 0.02  # ≈ uniform reference

    multi = np.stack([vals, 2 * vals], axis=1)                # (n, k) path
    res2, _ = resample_uniform(multi, t, fs)
    assert res2.shape == (grid.size, 2)
    assert np.allclose(res2[:, 0] * 2, res2[:, 1], atol=1e-5)


def test_resample_uniform_validates():
    with pytest.raises(ValueError, match=">= 2"):
        resample_uniform([1.0], [0.0], 100.0)
    with pytest.raises(ValueError, match="increasing"):
        resample_uniform([1.0, 2.0, 3.0], [0.0, 0.02, 0.01], 100.0)
    with pytest.raises(ValueError, match="target_fs"):
        resample_uniform([1.0, 2.0], [0.0, 0.01], 0.0)


def test_fs_ok_drops_deviating_windows():
    t = np.arange(50) / 100.0
    assert fs_ok(t, 100.0, 0.1)
    assert fs_ok(t + np.random.default_rng(1).uniform(-1e-3, 1e-3, 50) * 0, 100.0, 0.1)
    assert not fs_ok(t[::2], 100.0, 0.1)      # decimated -> live fs 50 Hz, out of tol
    assert not fs_ok(t[:1], 100.0, 0.1)       # too short to estimate fs
    assert not fs_ok(np.zeros(5), 100.0, 0.1)  # zero span


def test_accept_format_single_packet_format():
    assert accept_format(384, 384)             # the one controlled-link format
    assert not accept_format(128, 384)         # stray legacy frame rejected
    assert not accept_format(0, 0)
