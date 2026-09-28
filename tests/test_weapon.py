"""Weapon head, operating modes, soft voting and the tier harness.

The synthetic weapon signature is a flattening that lowers inter-carrier variance.
"""

import numpy as np
import pytest

from wavetrace.synthetic import generate_stream
from wavetrace.synthetic import generate_paired_recording
from wavetrace.calibration import Calibration
from wavetrace.config import ModelConfig
from wavetrace.groundtruth import build_dataset, load_dataset, save_dataset, weapon_label_fn
from wavetrace.groundtruth.camera_labeler import ScriptedLabeler
from wavetrace.adapters.recognition.heads import build_weapon_head, load_weapon_head
from wavetrace.recognition import (
    SegmentVoter,
    binary_rates,
    concat_arrays,
    concat_datasets,
    evaluate_concealment_gap,
    evaluate_weapon,
    open_session,
    tier_verdict,
)

NUM_ANT = 2
NUM_SUB = 32
FS = 100.0
WEAPON_SPAN = (2.5, 7.5)
# Flattening depth interleaved across subjects to prevent shift.
RECORDINGS = [("s0", "u0", 200, 0.40), ("s1", "u1", 201, 0.55),
              ("s2", "u1", 202, 0.45), ("s3", "u0", 203, 0.60)]


def _calibrate():
    baseline, _ = generate_stream(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, num_frames=60,
        perturbation_hz=0.0, perturbation_depth=0.0, cfo_hz=0.0, noise_std=0.005, seed=7,
    )
    cal = Calibration(baseline_packets=50)
    for fr in baseline:
        cal.observe(fr)
    return cal.finalize()


def _weapon_recording(sess, subj, seed, depth, duration=10.0):
    """Body present throughout recording. Weapon span adds only flattening to the presence turbulence."""
    frames, _, truth = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=duration,
        camera_fps=30.0, presence_spans=[(0.0, duration)], presence_turbulence_std=0.10,
        weapon_spans=[WEAPON_SPAN], weapon_signature_depth=depth,
        amplitude_hz=2.0, amplitude_depth=0.45, session_id=sess, subject_id=subj, seed=seed,
    )
    return frames, truth


@pytest.fixture(scope="module")
def weapon_data():
    result = _calibrate()
    datasets = []
    for sess, subj, seed, depth in RECORDINGS:
        frames, _ = _weapon_recording(sess, subj, seed, depth)
        labeler = ScriptedLabeler([(*WEAPON_SPAN, True)], label_fn=weapon_label_fn)
        # Weapon dataset contract: RAW magnitudes (gain_lock=None, intercarrier=True).
        datasets.append(build_dataset(frames, result, None, labeler, window=32, hop=16,
                                      session_id=sess, subject_id=subj, intercarrier=True))
    _, y, session_ids, subject_ids = concat_datasets(datasets)
    return {
        "datasets": datasets,
        "X_ic": concat_arrays(datasets, "X_intercarrier"),
        "X_image": concat_arrays(datasets, "X_image"),
        "y": y, "sess": session_ids, "subj": subject_ids,
        "K": len(result.subcarriers),
        "result": result,
    }


def _cfg(backend, k=12, **kw):
    return ModelConfig(stage="weapon", k=k, backend=backend, **kw)


def test_weapon_signature_lowers_sigma2(weapon_data):
    # Column 9 is window mean of per-packet inter-carrier variance.
    s2 = weapon_data["X_ic"][:, 9]
    y = weapon_data["y"]
    assert np.median(s2[y == 1]) < 0.5 * np.median(s2[y == 0])  # Metal signature lowers variance.


def test_weapon_signature_touches_only_spans():
    kwargs = dict(num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=2.0,
                  camera_fps=30.0, presence_spans=[(0.0, 2.0)], presence_turbulence_std=0.10,
                  weapon_spans=[(0.5, 1.0)], seed=21)
    plain, _, _ = generate_paired_recording(**kwargs)
    flat, _, truth = generate_paired_recording(**kwargs, weapon_signature_depth=0.5)
    for fp, ft in zip(plain, flat):
        same = np.array_equal(np.asarray(fp.grid), np.asarray(ft.grid))
        assert same != (0.5 <= fp.timestamp < 1.0)  # Modulated inside weapon span only.
    assert truth["weapon_signature_depth"] == pytest.approx(0.5)


def test_dual_block_build_with_gain_lock():
    """intercarrier=True + gain_lock makes dual-block dataset: IC from raw mags, features from locked mags."""
    result = _calibrate()
    cal = Calibration(baseline_packets=50)
    baseline, _ = generate_stream(num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS,
                                 num_frames=60, perturbation_hz=0.0, perturbation_depth=0.0,
                                 cfo_hz=0.0, noise_std=0.005, seed=7)
    for fr in baseline:
        cal.observe(fr)
    cal.finalize()
    frames, _ = _weapon_recording("sX", "uX", 210, 0.5, duration=2.0)
    labeler = ScriptedLabeler([(0.5, 1.0, True)], label_fn=weapon_label_fn)
    ds = build_dataset(frames, result, cal.gain_lock, labeler, window=32, hop=16, intercarrier=True)
    K = len(result.subcarriers)
    assert ds.X_intercarrier is not None
    assert ds.X_intercarrier.shape == (ds.y.size, 27)
    assert ds.X_features.shape == (ds.y.size, 9 * K)
    assert ds.meta["gain_locked"] is True and ds.meta["intercarrier"] is True


def test_train_weapon_ic27_and_fusion(weapon_data, tmp_path):
    from wavetrace.recognition import train_weapon
    d = weapon_data
    K = d["K"]
    dataset_paths = [save_dataset(ds, tmp_path / f"ds{i}") for i, ds in enumerate(d["datasets"])]

    # ic27: 27-feature inter-carrier block
    inter_carrier_head, inter_carrier_metrics = train_weapon(dataset_paths, out_dir=tmp_path / "w_ic",
                                                             feature_mode="ic27")
    assert inter_carrier_metrics["feature_mode"] == "ic27" and inter_carrier_metrics["n_features"] == 27
    assert inter_carrier_metrics["train_accuracy"] > 0.7
    assert (tmp_path / "w_ic" / "model.joblib").exists()
    assert (tmp_path / "w_ic" / "metrics.json").exists()

    # fusion: hstack(X_ic, X_features). X_features is raw-magnitude 9*K block. Width is 27 + 9*K.
    fusion_head, fusion_metrics = train_weapon(dataset_paths, out_dir=tmp_path / "w_fu",
                                               feature_mode="fusion",
                                               config=ModelConfig(stage="weapon", k=K, backend="mlp"))
    assert fusion_metrics["feature_mode"] == "fusion" and fusion_metrics["n_features"] == 27 + 9 * K
    assert fusion_metrics["train_accuracy"] > 0.7


def test_train_weapon_validates_mode():
    with pytest.raises(ValueError, match="feature_mode"):
        from wavetrace.recognition import train_weapon
        train_weapon([], feature_mode="bad")


def test_intercarrier_roundtrip_and_backcompat(weapon_data, tmp_path):
    ds = weapon_data["datasets"][0]
    assert ds.X_intercarrier.shape == (ds.y.size, 27) and ds.meta["intercarrier"] is True
    reloaded = load_dataset(save_dataset(ds, tmp_path / "w"))
    assert np.array_equal(reloaded.X_intercarrier, ds.X_intercarrier)
    # Legacy datasets built without intercarrier block load correctly without it.
    frames, _ = _weapon_recording("sY", "uY", 211, 0.0, duration=2.0)
    labeler = ScriptedLabeler([(0.5, 1.0, True)], label_fn=weapon_label_fn)
    old = build_dataset(frames, weapon_data["result"], None, labeler, window=32, hop=16)
    assert old.X_intercarrier is None
    assert load_dataset(save_dataset(old, tmp_path / "old")).X_intercarrier is None


def test_variance_head_learns_threshold_and_direction(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 0.1, (200, 27)).astype(np.float32)
    y = (np.arange(200) % 2).astype(np.int64)
    X[y == 1, 9] -= 1.0                      # Physics direction: weapon lowers variance.
    head = build_weapon_head(_cfg("variance")).fit(X, y)
    assert (head.predict(X) == y).mean() == 1.0
    proba = head.predict_proba(X)
    assert proba.shape == (200, 2) and np.allclose(proba.sum(axis=1), 1.0)
    loaded = load_weapon_head(head.save(tmp_path / "v.joblib"))
    assert np.allclose(loaded.predict_proba(X), proba)

    X_flipped = X.copy()
    X_flipped[:, 9] *= -1.0                         # Flipped world: weapon increases variance. Model learns direction.
    head2 = build_weapon_head(_cfg("variance")).fit(X_flipped, y)
    assert (head2.predict(X_flipped) == y).mean() == 1.0


def test_variance_head_validates():
    with pytest.raises(ValueError, match="not fitted"):
        build_weapon_head(_cfg("variance")).predict(np.zeros((1, 27), np.float32))
    X = np.zeros((9, 27), np.float32)
    with pytest.raises(ValueError, match="binary"):
        build_weapon_head(_cfg("variance")).fit(X, np.arange(9) % 3)
    with pytest.raises(ValueError, match="constant"):
        build_weapon_head(_cfg("variance")).fit(X, np.arange(9) % 2)


def test_sklearn_weapon_backend(weapon_data):
    head = build_weapon_head(_cfg("mlp", k=weapon_data["K"])).fit(weapon_data["X_ic"], weapon_data["y"])
    assert (head.predict(weapon_data["X_ic"]) == weapon_data["y"]).mean() > 0.9


def test_weapon_eval_gate_passes_on_synthetic(weapon_data):
    """Variance baseline beats majority on LOGO folds and passes tier gate (FP <= 10%, TPR >= 90%)."""
    d = weapon_data
    cfg = _cfg("variance", k=d["K"])
    rep = evaluate_weapon(d["X_ic"], d["y"], session_ids=d["sess"], subject_ids=d["subj"],
                          make_head=lambda: build_weapon_head(cfg))
    for split in ("session", "subject"):
        r = rep[split]
        assert r["accuracy"] >= 0.95
        assert r["accuracy"] >= r["majority_accuracy"] + 0.30
        assert {"tpr", "fp_rate"} <= r.keys()          # Binary rates ride along.
    assert rep["verdict"]["verdict"] == "PASS"
    assert rep["verdict"]["tpr"] >= 0.95 and rep["verdict"]["fp_rate"] <= 0.10
    assert sorted(f["group"] for f in rep["session"]["folds"]) == ["s0", "s1", "s2", "s3"]


def test_concealment_gap_holds_out_concealed_tier(weapon_data):
    """Train on visible tiers, score concealed split separately.
    Concealed set (s3) passes with small gap and never leaks into visible LOGO folds."""
    d = weapon_data
    cfg = _cfg("variance", k=d["K"])
    is_concealed = np.asarray(d["sess"]) == "s3"
    rep = evaluate_concealment_gap(
        d["X_ic"], d["y"], is_concealed, d["subj"], make_head=lambda: build_weapon_head(cfg))

    assert rep["concealed"]["n"] == int(is_concealed.sum())
    assert {"tpr", "fp_rate", "accuracy"} <= rep["concealed"].keys()
    assert rep["verdict"] == "PASS"
    assert rep["concealed"]["tpr"] >= 0.90 and rep["concealed"]["fp_rate"] <= 0.10
    assert "s3" not in [f["group"] for f in rep["visible"]["folds"]]  # Concealed not in visible folds.
    assert isinstance(rep["tpr_gap"], float)


def test_concealment_gap_needs_both_splits(weapon_data):
    d = weapon_data
    cfg = _cfg("variance", k=d["K"])
    with pytest.raises(ValueError):
        evaluate_concealment_gap(d["X_ic"], d["y"], np.zeros(d["y"].size, bool), d["subj"],
                                 make_head=lambda: build_weapon_head(cfg))


def test_cnn_head_trains_roundtrips_deterministic(weapon_data, tmp_path):
    pytest.importorskip("torch")
    d = weapon_data
    X, y = d["X_image"], d["y"]
    head = build_weapon_head(_cfg("cnn", k=d["K"], window=32, seed=3)).fit(X, y, epochs=15)
    assert (head.predict(X) == y).mean() > 0.85
    proba = head.predict_proba(X)
    assert proba.shape == (y.size, 2) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(head.predict_proba(X), proba)              # Predict is deterministic.
    flat = X.reshape(X.shape[0], -1)                              # predict_window seam
    assert np.allclose(head.predict_proba(flat), proba, atol=1e-5)
    loaded = load_weapon_head(head.save(tmp_path / "cnn.joblib"))
    assert np.allclose(loaded.predict_proba(X), proba, atol=1e-6)


def test_weapon_mode_is_standalone(weapon_data, tmp_path):
    # Weapon mode classifies every window independently without presence verdict.
    d = weapon_data
    head = build_weapon_head(_cfg("variance", k=d["K"])).fit(d["X_ic"], d["y"])
    session = open_session("weapon", head.save(tmp_path / "w.joblib"))
    weapon_index = int(np.flatnonzero(d["y"] == 1)[0])
    no_weapon_index = int(np.flatnonzero(d["y"] == 0)[0])
    weapon_class, weapon_proba = session.predict_window(d["X_ic"][weapon_index])
    no_weapon_class, _ = session.predict_window(d["X_ic"][no_weapon_index])
    assert (weapon_class, no_weapon_class) == (1, 0)
    assert 0.5 <= weapon_proba <= 1.0


def test_mode_session_validates_mode():
    with pytest.raises(ValueError, match="presence.*weapon"):
        open_session("gate", "irrelevant")


def test_voter_recovers_segment_label_from_noisy_windows():
    # Weak per-window head. Segment soft vote recovers the correct class.
    rng = np.random.default_rng(5)
    voter = SegmentVoter()
    correct = 0
    n = 40
    for _ in range(n):
        p1 = np.clip(0.55 + rng.normal(0, 0.15), 0.0, 1.0)
        correct += p1 > 0.5
        voter.add([1 - p1, p1])
    assert correct / n < 0.75                     # Per-window head is weak.
    cls, mean = voter.finalize()
    assert cls == 1                               # Segment vote recovers true class.
    assert len(voter) == 0                        # finalize() resets for the next segment.


def test_voter_middle_fraction_and_decimation():
    voter = SegmentVoter(middle_fraction=0.5)
    # Edge windows vote 0, mid-crossing windows vote 1.
    for p in ([0.9, 0.1],) * 5 + ([0.1, 0.9],) * 6 + ([0.9, 0.1],) * 5:
        voter.add(p)
    assert voter.finalize()[0] == 1               # Middle slice isolates the crossing.
    full = SegmentVoter()
    for p in ([0.9, 0.1],) * 5 + ([0.1, 0.9],) * 6 + ([0.9, 0.1],) * 5:
        full.add(p)
    assert full.finalize()[0] == 0                # Without middle slice, edges win.

    dec = SegmentVoter(decimate=2)
    for p in ([0.2, 0.8], [0.8, 0.2]) * 4:
        dec.add(p)
    cls, mean = dec.finalize()                    # Decimated: only 0.8-class votes.
    assert cls == 1 and mean[1] == pytest.approx(0.8)


def test_voter_correlated_windows_gain_is_nil_and_validation():
    # Identical windows: the vote equals the per-window verdict.
    voter = SegmentVoter()
    for _ in range(10):
        voter.add([0.6, 0.4])
    assert voter.finalize()[0] == 0
    with pytest.raises(ValueError, match="no votes"):
        SegmentVoter().finalize()
    with pytest.raises(ValueError, match="middle_fraction"):
        SegmentVoter(middle_fraction=0.0)
    with pytest.raises(ValueError, match="decimate"):
        SegmentVoter(decimate=0)
    bad = SegmentVoter()
    bad.add([0.5, 0.5])
    with pytest.raises(ValueError, match="class count"):
        bad.add([0.2, 0.3, 0.5])


def test_binary_rates_and_tier_verdict_boundaries():
    cm = np.array([[90, 10], [5, 95]])            # fp 0.10, tpr 0.95
    rates = binary_rates(cm)
    assert rates == {"tpr": pytest.approx(0.95), "fp_rate": pytest.approx(0.10)}
    assert tier_verdict({"a": rates})["verdict"] == "PASS"          # Boundaries are inclusive.

    false_positive_failure = tier_verdict({"a": {"tpr": 0.95, "fp_rate": 0.101}})
    assert false_positive_failure["verdict"] == "FAIL" and "fp_rate" in false_positive_failure["reasons"][0]
    missed_detection_failure = tier_verdict({"a": {"tpr": 0.899, "fp_rate": 0.05}})
    assert missed_detection_failure["verdict"] == "FAIL" and "tpr" in missed_detection_failure["reasons"][0]
    # Worst-of-splits: a good split cannot mask a bad one.
    mixed = tier_verdict({"good": {"tpr": 1.0, "fp_rate": 0.0},
                          "bad": {"tpr": 0.5, "fp_rate": 0.5}})
    assert mixed["verdict"] == "FAIL" and mixed["tpr"] == 0.5 and mixed["fp_rate"] == 0.5
    with pytest.raises(ValueError, match="2x2"):
        binary_rates(np.zeros((3, 3)))
