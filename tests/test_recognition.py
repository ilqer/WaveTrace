"""Stage A presence head and multi-RX plumbing, on 4 synthetic recordings.

Accuracy is measured leave-one-session-out and leave-one-subject-out.
"""

import json

import numpy as np
import pytest

from wavetrace.synthetic import generate_paired_recording
from wavetrace.adapters.recognition.heads import (
    build_presence_head,
    build_weapon_head,
    load_presence_head,
)
from wavetrace.config import ModelConfig
from wavetrace.groundtruth import (
    ScriptedLabeler,
    build_dataset,
    load_dataset,
    presence_label_function,
    save_dataset,
)
from wavetrace.recognition import (
    InferenceSession,
    concat_datasets,
    evaluate_presence,
    fuse,
    leave_one_group_out,
    measure_latency,
    segmenter_baseline,
    train_presence,
)
from fixtures.synthetic_csi import calibrate_quiet_baseline

NUM_ANTENNAS = 2
NUM_SUBCARRIERS = 32
SAMPLE_RATE_HZ = 100.0
SPAN = (2.5, 7.5)  # presence span (s) inside each 10 s recording
# turbulence interleaved across subjects so every train fold sees weak AND strong sessions
RECORDINGS = [("s0", "u0", 100, 0.15), ("s1", "u1", 101, 0.20),
              ("s2", "u1", 102, 0.25), ("s3", "u0", 103, 0.30)]


def _recording(session_id, subject_id, seed, turbulence_std, duration_s=10.0):
    """Presence recording: wobble in both classes. Per-subcarrier turbulence only during presence."""
    frames, _, truth = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        duration_s=duration_s, camera_frame_rate_hz=30.0, presence_spans=[SPAN],
        amplitude_hz=2.0, amplitude_depth=0.45,
        presence_turbulence_std=turbulence_std, session_id=session_id, subject_id=subject_id, seed=seed,
    )
    return frames, truth


@pytest.fixture(scope="module")
def presence_data():
    """4 recordings -> labeled datasets -> concatenated (X, y, groups) + CSI images."""
    result, gain = calibrate_quiet_baseline(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        num_frames=60, seed=7, baseline_packets=50)
    datasets = []
    for session_id, subject_id, seed, turbulence_std in RECORDINGS:
        frames, _ = _recording(session_id, subject_id, seed, turbulence_std)
        labeler = ScriptedLabeler([(*SPAN, True)], label_function=presence_label_function)
        datasets.append(build_dataset(frames, result, gain, labeler, window=32, hop=16,
                                      session_id=session_id, subject_id=subject_id))
    X, y, session_ids, subject_ids = concat_datasets(datasets)
    return {
        "datasets": datasets,
        "X": X, "y": y, "sess": session_ids, "subj": subject_ids,
        "X_image": np.concatenate([dataset.X_image for dataset in datasets]),
        "K": len(result.subcarriers),
        "config": ModelConfig(stage="presence", k=len(result.subcarriers)),
    }


def test_presence_turbulence_separates_features(presence_data):
    # Present windows must have higher per-subcarrier temporal std (idx 1).
    X, y, subcarrier_count = presence_data["X"], presence_data["y"], presence_data["K"]
    subcarrier_temporal_std = X.reshape(X.shape[0], subcarrier_count, 9)[:, :, 1].mean(axis=1)
    assert subcarrier_temporal_std[y == 1].mean() > 1.5 * subcarrier_temporal_std[y == 0].mean()


def test_turbulence_touches_only_presence_spans():
    kwargs = dict(num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS,
                  sample_rate_hz=SAMPLE_RATE_HZ, duration_s=2.0, camera_frame_rate_hz=30.0,
                  presence_spans=[(0.5, 1.0)], seed=11)
    plain, _, _ = generate_paired_recording(**kwargs)
    turbulent, _, truth = generate_paired_recording(**kwargs, presence_turbulence_std=0.2,
                                                    session_id="sA", subject_id="uA")
    for plain_frame, turbulent_frame in zip(plain, turbulent):
        same = np.array_equal(np.asarray(plain_frame.grid), np.asarray(turbulent_frame.grid))
        inside = 0.5 <= plain_frame.timestamp < 1.0
        assert same != inside  # outside spans byte-identical, inside modulated
    assert truth["session_id"] == "sA" and truth["subject_id"] == "uA"
    assert truth["presence_turbulence_std"] == pytest.approx(0.2)


def test_dataset_group_ids_roundtrip(presence_data, tmp_path):
    dataset = presence_data["datasets"][0]
    sample_count = dataset.y.size
    assert list(dataset.session_ids) == ["s0"] * sample_count and (
        list(dataset.subject_ids) == ["u0"] * sample_count)
    reloaded = load_dataset(save_dataset(dataset, tmp_path / "ds"))
    assert list(reloaded.session_ids) == ["s0"] * sample_count
    assert list(reloaded.subject_ids) == ["u0"] * sample_count


def test_model_config_validates():
    assert ModelConfig(stage="presence", k=12).backend == "mlp"
    with pytest.raises(ValueError):
        ModelConfig(stage="posture", k=12)
    with pytest.raises(ValueError, match="'mlp'/'svm'"):
        build_presence_head(ModelConfig(stage="presence", k=12, backend="cnn"))  # cnn is weapon-side only
    with pytest.raises(ValueError):
        ModelConfig(stage="presence", k=0)
    with pytest.raises(ValueError):
        ModelConfig(stage="presence", k=12, fs_tol=1.5)


def test_unregistered_backend_rejected_at_resolution_not_at_config():
    """An unregistered `backend` name is accepted by `ModelConfig` and rejected only where it is
    actually resolved: head construction/`load()`, via `get_backend_class`."""
    config = ModelConfig(stage="presence", k=12, backend="rf")
    with pytest.raises(ValueError, match="rf"):
        build_presence_head(config)
    with pytest.raises(ValueError, match="rf"):
        build_weapon_head(config)


def _blobs(sample_count=120, feature_count=18, seed=0):
    """Two trivially separable feature clusters (per-class mean shift)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (sample_count, feature_count)).astype(np.float32)
    y = (np.arange(sample_count) % 2).astype(np.int64)
    X[y == 1] += 3.0
    return X, y


@pytest.mark.parametrize("backend", ["mlp", "svm"])
def test_head_fit_predict_shapes(backend):
    X, y = _blobs()
    head = build_presence_head(ModelConfig(stage="presence", k=2, backend=backend)).fit(X, y)
    predictions = head.predict(X)
    proba = head.predict_proba(X)
    assert predictions.shape == (X.shape[0],) and set(predictions) <= {0, 1}
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (predictions == y).mean() > 0.95


def test_head_save_load_roundtrip(tmp_path):
    X, y = _blobs()
    head = build_presence_head(ModelConfig(stage="presence", k=2)).fit(X, y)
    path = head.save(tmp_path / "m" / "model.joblib")
    loaded = load_presence_head(path)
    assert loaded.config == head.config
    assert np.array_equal(loaded.predict(X), head.predict(X))
    assert np.allclose(loaded.predict_proba(X), head.predict_proba(X))


def test_head_unfitted_raises():
    head = build_presence_head(ModelConfig(stage="presence", k=2))
    with pytest.raises(ValueError, match="not fitted"):
        head.predict(np.zeros((1, 18), np.float32))


def test_train_presence_persists(presence_data, tmp_path):
    dataset_dirs = [save_dataset(dataset, tmp_path / f"ds{i}")
                    for i, dataset in enumerate(presence_data["datasets"][:2])]
    head, metrics = train_presence(dataset_dirs, tmp_path / "models")
    assert (tmp_path / "models" / "model.joblib").exists()
    with open(tmp_path / "models" / "metrics.json") as metrics_file:
        assert json.load(metrics_file) == metrics
    assert metrics["backend"] == "mlp" and metrics["k"] == presence_data["K"]
    assert metrics["n_samples"] == sum(dataset.y.size for dataset in presence_data["datasets"][:2])
    assert metrics["sessions"] == ["s0", "s1"] and metrics["subjects"] == ["u0", "u1"]
    assert metrics["train_accuracy"] > 0.9
    assert head.predict(presence_data["X"][:4]).shape == (4,)


def test_eval_gate_head_beats_both_baselines(presence_data):
    """Logo accuracy must beat the majority class and PresenceSegmenter baselines."""
    data = presence_data
    report = evaluate_presence(
        data["X"], data["y"], session_ids=data["sess"], subject_ids=data["subj"],
        make_head=lambda: build_presence_head(data["config"]),
        X_image=data["X_image"],
        segmenter_kwargs={"coefficient_of_variation_window": 16, "enter_coefficient_of_variation": 0.01,
                          "exit_coefficient_of_variation": 0.005},
    )
    for split in ("session", "subject"):
        split_report = report[split]
        assert split_report["accuracy"] >= 0.92   # measured 0.98 / 0.975
        assert split_report["accuracy"] >= split_report["majority_accuracy"] + 0.30
        assert split_report["accuracy"] >= report["segmenter_accuracy"] + 0.05
        confusion = split_report["confusion"]
        assert confusion.shape == (2, 2) and confusion.sum() == data["y"].size
    # folds are group-disjoint and cover every group exactly once
    assert sorted(fold["group"] for fold in report["session"]["folds"]) == ["s0", "s1", "s2", "s3"]
    assert sorted(fold["group"] for fold in report["subject"]["folds"]) == ["u0", "u1"]
    for fold in report["session"]["folds"]:
        assert fold["n"] == int((data["sess"] == fold["group"]).sum())


def test_logo_requires_two_groups(presence_data):
    data = presence_data
    with pytest.raises(ValueError, match="2 distinct groups"):
        leave_one_group_out(data["X"], data["y"], np.full(data["y"].size, "only"), lambda: None)


def test_segmenter_baseline_flags_turbulent_windows(presence_data):
    data = presence_data
    predictions = segmenter_baseline(data["X_image"], coefficient_of_variation_window=16,
                                     enter_coefficient_of_variation=0.01,
                                     exit_coefficient_of_variation=0.005)
    assert predictions.shape == data["y"].shape and set(np.unique(predictions)) <= {0, 1}
    # DSP gate baseline: above chance, below the trained head.
    assert (predictions == data["y"]).mean() > 0.75
    assert predictions[data["y"] == 1].mean() > predictions[data["y"] == 0].mean()


@pytest.fixture(scope="module")
def inference_session(presence_data, tmp_path_factory):
    head = build_presence_head(presence_data["config"]).fit(presence_data["X"], presence_data["y"])
    path = head.save(tmp_path_factory.mktemp("models") / "model.joblib")
    return InferenceSession(path)


def test_infer_predict_window_deterministic(presence_data, inference_session):
    data = presence_data
    present_features = data["X"][data["y"] == 1][0]
    absent_features = data["X"][data["y"] == 0][0]
    class_present, probability_present = inference_session.predict_window(present_features)
    class_absent, probability_absent = inference_session.predict_window(absent_features)
    assert (class_present, class_absent) == (1, 0)
    assert 0.5 <= probability_present <= 1.0 and 0.5 <= probability_absent <= 1.0   # argmax probability
    assert inference_session.predict_window(present_features) == (class_present, probability_present)


def test_presence_mode_session(presence_data, tmp_path):
    from wavetrace.recognition import open_session
    data = presence_data
    head = build_presence_head(data["config"]).fit(data["X"], data["y"])
    session = open_session("presence", head.save(tmp_path / "p.joblib"))
    class_id, proba = session.predict_window(data["X"][data["y"] == 1][0])
    assert class_id == 1 and 0.5 <= proba <= 1.0


def test_infer_latency_under_8ms(presence_data, inference_session):
    stats = measure_latency(inference_session, presence_data["X"][0], iters=200)
    assert stats["mean_ms"] < 8.0
    assert stats["p95_ms"] < 8.0  # per-window inference budget


def test_fuse_concat_shape_and_order():
    node_features_0 = np.arange(9, dtype=np.float32)
    node_features_1 = np.arange(9, 27, dtype=np.float32)
    node_features_2 = np.arange(27, 30, dtype=np.float32)
    assert np.array_equal(fuse([node_features_0]), node_features_0)   # 1 node = identity
    fused = fuse([node_features_0, node_features_1, node_features_2])
    assert fused.shape == (30,) and fused.dtype == np.float32
    assert np.array_equal(
        fused, np.concatenate([node_features_0, node_features_1, node_features_2]))   # stable node order
    out = np.empty(30, dtype=np.float32)
    assert fuse([node_features_0, node_features_1, node_features_2], out=out) is out   # per-emit buffer reuse


def test_fuse_validates():
    with pytest.raises(ValueError, match="no node features"):
        fuse([])
    with pytest.raises(ValueError, match="1-D"):
        fuse([np.zeros((2, 9), np.float32)])
    with pytest.raises(ValueError, match="out"):
        fuse([np.zeros(9, np.float32)], out=np.empty(10, np.float32))


