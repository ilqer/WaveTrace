"""Weapon head, operating modes, soft voting, and tier harness tests.

Validates learning/gating pipeline using synthetic weapon signatures (flattening lowers variance).
Tier verdicts use scripted recordings. CNN tests skip if torch is missing.
"""

import numpy as np
import pytest

from wavetrace.Synthetic import generateStream
from wavetrace.Synthetic import generatePairedRecording
from wavetrace.Calibration import Calibration
from wavetrace.Config import ModelConfig
from wavetrace.groundtruth import buildDataset, loadDataset, saveDataset, weaponLabelFn
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler
from wavetrace.recognition import (
    SegmentVoter,
    WeaponHead,
    binaryRates,
    concatArrays,
    concatDatasets,
    evaluateConcealmentGap,
    evaluateWeapon,
    modeSession,
    tierVerdict,
)

NUM_ANT = 2
NUM_SUB = 32
FS = 100.0
WEAPON_SPAN = (2.5, 7.5)
# Flattening depth interleaved across subjects to prevent shift.
RECORDINGS = [("s0", "u0", 200, 0.40), ("s1", "u1", 201, 0.55),
              ("s2", "u1", 202, 0.45), ("s3", "u0", 203, 0.60)]


def _calibrate():
    baseline, _ = generateStream(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, numFrames=60,
        perturbationHz=0.0, perturbationDepth=0.0, cfoHz=0.0, noiseStd=0.005, seed=7,
    )
    cal = Calibration(baseline_packets=50)
    for fr in baseline:
        cal.observe(fr)
    return cal.finalize()


def _weapon_recording(sess, subj, seed, depth, duration=10.0):
    """Body present throughout recording. Weapon span adds only flattening to the presence turbulence."""
    frames, _, truth = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=duration,
        cameraFps=30.0, presenceSpans=[(0.0, duration)], presenceTurbulenceStd=0.10,
        weaponSpans=[WEAPON_SPAN], weaponSignatureDepth=depth,
        amplitudeHz=2.0, amplitudeDepth=0.45, sessionId=sess, subjectId=subj, seed=seed,
    )
    return frames, truth


@pytest.fixture(scope="module")
def weapon_data():
    result = _calibrate()
    datasets = []
    for sess, subj, seed, depth in RECORDINGS:
        frames, _ = _weapon_recording(sess, subj, seed, depth)
        labeler = ScriptedLabeler([(*WEAPON_SPAN, True)], label_fn=weaponLabelFn)
        # Weapon dataset contract: RAW magnitudes (gainLock=None, intercarrier=True).
        datasets.append(buildDataset(frames, result, None, labeler, window=32, hop=16,
                                      session_id=sess, subject_id=subj, intercarrier=True))
    _, y, sessIds, subjIds = concatDatasets(datasets)
    return {
        "datasets": datasets,
        "X_ic": concatArrays(datasets, "X_intercarrier"),
        "X_image": concatArrays(datasets, "X_image"),
        "y": y, "sess": sessIds, "subj": subjIds,
        "K": len(result.subcarriers),
        "result": result,
    }


def _cfg(backend, k=12, **kw):
    return ModelConfig(stage="weapon", k=k, backend=backend, **kw)


# ----- 7p-a: weapon-signature synthetic + X_intercarrier ------------------------------------------

def test_weapon_signature_lowers_sigma2(weapon_data):
    # Column 9 is window mean of per-packet inter-carrier variance.
    s2 = weapon_data["X_ic"][:, 9]
    y = weapon_data["y"]
    assert np.median(s2[y == 1]) < 0.5 * np.median(s2[y == 0])  # Metal signature lowers variance.


def test_weapon_signature_touches_only_spans():
    kwargs = dict(numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=2.0,
                  cameraFps=30.0, presenceSpans=[(0.0, 2.0)], presenceTurbulenceStd=0.10,
                  weaponSpans=[(0.5, 1.0)], seed=21)
    plain, _, _ = generatePairedRecording(**kwargs)
    flat, _, truth = generatePairedRecording(**kwargs, weaponSignatureDepth=0.5)
    for fp, ft in zip(plain, flat):
        same = np.array_equal(np.asarray(fp.grid), np.asarray(ft.grid))
        assert same != (0.5 <= fp.timestamp < 1.0)  # Modulated inside weapon span only.
    assert truth["weapon_signature_depth"] == pytest.approx(0.5)


def test_dual_block_build_with_gain_lock():
    """intercarrier=True + gainLock makes dual-block dataset: IC from raw mags, features from locked mags."""
    result = _calibrate()
    cal = Calibration(baseline_packets=50)
    baseline, _ = generateStream(numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS,
                                 numFrames=60, perturbationHz=0.0, perturbationDepth=0.0,
                                 cfoHz=0.0, noiseStd=0.005, seed=7)
    for fr in baseline:
        cal.observe(fr)
    cal.finalize()
    frames, _ = _weapon_recording("sX", "uX", 210, 0.5, duration=2.0)
    labeler = ScriptedLabeler([(0.5, 1.0, True)], label_fn=weaponLabelFn)
    ds = buildDataset(frames, result, cal.gainLock, labeler, window=32, hop=16, intercarrier=True)
    K = len(result.subcarriers)
    assert ds.X_intercarrier is not None
    assert ds.X_intercarrier.shape == (ds.y.size, 27)
    assert ds.X_features.shape == (ds.y.size, 9 * K)
    assert ds.meta["gain_locked"] is True and ds.meta["intercarrier"] is True


def test_train_weapon_ic27_and_fusion(weapon_data, tmp_path):
    """trainWeapon ic27 and fusion feature modes produce fitted models with correct feature dims."""
    from wavetrace.recognition import trainWeapon
    d = weapon_data
    K = d["K"]
    dsDirs = [saveDataset(ds, tmp_path / f"ds{i}") for i, ds in enumerate(d["datasets"])]

    # ic27: 27-feature inter-carrier block
    headIc, mIc = trainWeapon(dsDirs, out_dir=tmp_path / "w_ic", feature_mode="ic27")
    assert mIc["feature_mode"] == "ic27" and mIc["n_features"] == 27
    assert mIc["train_accuracy"] > 0.7
    assert (tmp_path / "w_ic" / "model.joblib").exists()
    assert (tmp_path / "w_ic" / "metrics.json").exists()

    # fusion: hstack(X_ic, X_features). X_features is raw-magnitude 9*K block. Width is 27 + 9*K.
    headFu, mFu = trainWeapon(dsDirs, out_dir=tmp_path / "w_fu",
                                 feature_mode="fusion",
                                 config=ModelConfig(stage="weapon", k=K, backend="mlp"))
    assert mFu["feature_mode"] == "fusion" and mFu["n_features"] == 27 + 9 * K
    assert mFu["train_accuracy"] > 0.7


def test_train_weapon_validates_mode():
    with pytest.raises(ValueError, match="feature_mode"):
        from wavetrace.recognition import trainWeapon
        trainWeapon([], feature_mode="bad")


def test_intercarrier_roundtrip_and_backcompat(weapon_data, tmp_path):
    ds = weapon_data["datasets"][0]
    assert ds.X_intercarrier.shape == (ds.y.size, 27) and ds.meta["intercarrier"] is True
    reloaded = loadDataset(saveDataset(ds, tmp_path / "w"))
    assert np.array_equal(reloaded.X_intercarrier, ds.X_intercarrier)
    # Legacy datasets built without intercarrier block load correctly without it.
    frames, _ = _weapon_recording("sY", "uY", 211, 0.0, duration=2.0)
    labeler = ScriptedLabeler([(0.5, 1.0, True)], label_fn=weaponLabelFn)
    old = buildDataset(frames, weapon_data["result"], None, labeler, window=32, hop=16)
    assert old.X_intercarrier is None
    assert loadDataset(saveDataset(old, tmp_path / "old")).X_intercarrier is None


# ----- 7p-b: variance-threshold + sklearn backends -------------------------------------------------

def test_variance_head_learns_threshold_and_direction(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 0.1, (200, 27)).astype(np.float32)
    y = (np.arange(200) % 2).astype(np.int64)
    X[y == 1, 9] -= 1.0                      # Physics direction: weapon lowers variance.
    head = WeaponHead(_cfg("variance")).fit(X, y)
    assert (head.predict(X) == y).mean() == 1.0
    proba = head.predict_proba(X)
    assert proba.shape == (200, 2) and np.allclose(proba.sum(axis=1), 1.0)
    loaded = WeaponHead.load(head.save(tmp_path / "v.joblib"))
    assert np.allclose(loaded.predict_proba(X), proba)

    X2 = X.copy()
    X2[:, 9] *= -1.0                         # Flipped world: weapon increases variance. Model learns direction.
    head2 = WeaponHead(_cfg("variance")).fit(X2, y)
    assert (head2.predict(X2) == y).mean() == 1.0


def test_variance_head_validates():
    with pytest.raises(ValueError, match="not fitted"):
        WeaponHead(_cfg("variance")).predict(np.zeros((1, 27), np.float32))
    X = np.zeros((9, 27), np.float32)
    with pytest.raises(ValueError, match="binary"):
        WeaponHead(_cfg("variance")).fit(X, np.arange(9) % 3)
    with pytest.raises(ValueError, match="constant"):
        WeaponHead(_cfg("variance")).fit(X, np.arange(9) % 2)


def test_sklearn_weapon_backend(weapon_data):
    head = WeaponHead(_cfg("mlp", k=weapon_data["K"])).fit(weapon_data["X_ic"], weapon_data["y"])
    assert (head.predict(weapon_data["X_ic"]) == weapon_data["y"]).mean() > 0.9


def test_weapon_eval_gate_passes_on_synthetic(weapon_data):
    """Variance baseline beats majority on LOGO folds and passes tier gate (FP <= 10%, TPR >= 90%)."""
    d = weapon_data
    cfg = _cfg("variance", k=d["K"])
    rep = evaluateWeapon(d["X_ic"], d["y"], session_ids=d["sess"], subject_ids=d["subj"],
                          make_head=lambda: WeaponHead(cfg))
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
    isConcealed = np.asarray(d["sess"]) == "s3"
    rep = evaluateConcealmentGap(
        d["X_ic"], d["y"], isConcealed, d["subj"], make_head=lambda: WeaponHead(cfg))

    assert rep["concealed"]["n"] == int(isConcealed.sum())
    assert {"tpr", "fp_rate", "accuracy"} <= rep["concealed"].keys()
    assert rep["verdict"] == "PASS"
    assert rep["concealed"]["tpr"] >= 0.90 and rep["concealed"]["fp_rate"] <= 0.10
    assert "s3" not in [f["group"] for f in rep["visible"]["folds"]]  # Concealed not in visible folds.
    assert isinstance(rep["tpr_gap"], float)


def test_concealment_gap_needs_both_splits(weapon_data):
    d = weapon_data
    cfg = _cfg("variance", k=d["K"])
    with pytest.raises(ValueError):
        evaluateConcealmentGap(d["X_ic"], d["y"], np.zeros(d["y"].size, bool), d["subj"],
                                 make_head=lambda: WeaponHead(cfg))


# ----- 7p-c: torch CNN backend ---------------------------------------------------------------------

def test_cnn_head_trains_roundtrips_deterministic(weapon_data, tmp_path):
    pytest.importorskip("torch")
    d = weapon_data
    X, y = d["X_image"], d["y"]
    head = WeaponHead(_cfg("cnn", k=d["K"], window=32, seed=3)).fit(X, y, epochs=15)
    assert (head.predict(X) == y).mean() > 0.85
    proba = head.predict_proba(X)
    assert proba.shape == (y.size, 2) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(head.predict_proba(X), proba)              # Predict is deterministic.
    flat = X.reshape(X.shape[0], -1)                              # predictWindow seam
    assert np.allclose(head.predict_proba(flat), proba, atol=1e-5)
    loaded = WeaponHead.load(head.save(tmp_path / "cnn.joblib"))
    assert np.allclose(loaded.predict_proba(X), proba, atol=1e-6)


# ----- 7p-d: the two operating modes (user decision 2026-06-11: independent, no cross-gating) ------

def test_weapon_mode_is_standalone(weapon_data, tmp_path):
    # Weapon mode classifies every window independently without presence verdict.
    d = weapon_data
    head = WeaponHead(_cfg("variance", k=d["K"])).fit(d["X_ic"], d["y"])
    session = modeSession("weapon", head.save(tmp_path / "w.joblib"))
    iWeapon = int(np.flatnonzero(d["y"] == 1)[0])
    iNone = int(np.flatnonzero(d["y"] == 0)[0])
    clsW, probaW = session.predictWindow(d["X_ic"][iWeapon])
    clsN, _ = session.predictWindow(d["X_ic"][iNone])
    assert (clsW, clsN) == (1, 0)
    assert 0.5 <= probaW <= 1.0


def test_mode_session_validates_mode():
    with pytest.raises(ValueError, match="presence.*weapon"):
        modeSession("gate", "irrelevant")


# ----- 7p-e: soft segment voting -------------------------------------------------------------------

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


# ----- 7p-f: tier harness (FP gate) ----------------------------------------------------------------

def test_binary_rates_and_tier_verdict_boundaries():
    cm = np.array([[90, 10], [5, 95]])            # fp 0.10, tpr 0.95
    rates = binaryRates(cm)
    assert rates == {"tpr": pytest.approx(0.95), "fp_rate": pytest.approx(0.10)}
    assert tierVerdict({"a": rates})["verdict"] == "PASS"          # Boundaries are inclusive.

    failFp = tierVerdict({"a": {"tpr": 0.95, "fp_rate": 0.101}})
    assert failFp["verdict"] == "FAIL" and "fp_rate" in failFp["reasons"][0]
    failTpr = tierVerdict({"a": {"tpr": 0.899, "fp_rate": 0.05}})
    assert failTpr["verdict"] == "FAIL" and "tpr" in failTpr["reasons"][0]
    # Worst-of-splits: a good split cannot mask a bad one.
    mixed = tierVerdict({"good": {"tpr": 1.0, "fp_rate": 0.0},
                          "bad": {"tpr": 0.5, "fp_rate": 0.5}})
    assert mixed["verdict"] == "FAIL" and mixed["tpr"] == 0.5 and mixed["fp_rate"] == 0.5
    with pytest.raises(ValueError, match="2x2"):
        binaryRates(np.zeros((3, 3)))
