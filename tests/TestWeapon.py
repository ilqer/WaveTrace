"""Phase 7 plumbing — weapon head + operating modes + soft voting + tier harness (7p-a..7p-f).

Same philosophy as P6: the synthetic weapon signature (cross-subcarrier flattening → lower σ²[p])
validates the LEARNING/GATING PIPELINE only — tier verdicts (7a–7d) come exclusively from real
scripted recordings after Phase-0 firmware. The CNN tests skip when torch is absent ([cnn] extra).
"""

import numpy as np
import pytest

from fixtures.SyntheticCsi import generateStream
from fixtures.SyntheticRecording import generatePairedRecording
from wavetrace.Calibration import Calibration
from wavetrace.Config import ModelConfig
from wavetrace.groundtruth import build_dataset, load_dataset, save_dataset, weapon_label_fn
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler
from wavetrace.recognition import (
    SegmentVoter,
    WeaponHead,
    binary_rates,
    concat_arrays,
    concat_datasets,
    evaluate_weapon,
    mode_session,
    tier_verdict,
)

NUM_ANT = 2
NUM_SUB = 32
FS = 100.0
WEAPON_SPAN = (2.5, 7.5)
# flattening depth interleaved across subjects (same anti-shift rationale as the P6 turbulence)
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
    """Body present THROUGHOUT (Yousaf's body-plus-weapon vs body-only framing); the weapon span
    adds only the σ²[p] flattening on top of the presence turbulence."""
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
        labeler = ScriptedLabeler([(*WEAPON_SPAN, True)], label_fn=weapon_label_fn)
        # gain_lock=None + intercarrier=True = the weapon-dataset contract (RAW magnitudes)
        datasets.append(build_dataset(frames, result, None, labeler, window=32, hop=16,
                                      session_id=sess, subject_id=subj, intercarrier=True))
    _, y, sess_ids, subj_ids = concat_datasets(datasets)
    return {
        "datasets": datasets,
        "X_ic": concat_arrays(datasets, "X_intercarrier"),
        "X_image": concat_arrays(datasets, "X_image"),
        "y": y, "sess": sess_ids, "subj": subj_ids,
        "K": len(result.subcarriers),
        "result": result,
    }


def _cfg(backend, k=12, **kw):
    return ModelConfig(stage="weapon", k=k, backend=backend, **kw)


# ----- 7p-a: weapon-signature synthetic + X_intercarrier ------------------------------------------

def test_weapon_signature_lowers_sigma2(weapon_data):
    # column 9 of the 27-block = window mean of the per-packet inter-carrier σ²[p]
    s2 = weapon_data["X_ic"][:, 9]
    y = weapon_data["y"]
    assert np.median(s2[y == 1]) < 0.5 * np.median(s2[y == 0])  # metal -> clearly lower σ²[p]


def test_weapon_signature_touches_only_spans():
    kwargs = dict(numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=2.0,
                  cameraFps=30.0, presenceSpans=[(0.0, 2.0)], presenceTurbulenceStd=0.10,
                  weaponSpans=[(0.5, 1.0)], seed=21)
    plain, _, _ = generatePairedRecording(**kwargs)
    flat, _, truth = generatePairedRecording(**kwargs, weaponSignatureDepth=0.5)
    for fp, ft in zip(plain, flat):
        same = np.array_equal(np.asarray(fp.grid), np.asarray(ft.grid))
        assert same != (0.5 <= fp.timestamp < 1.0)  # modulated inside the weapon span only
    assert truth["weapon_signature_depth"] == pytest.approx(0.5)


def test_dual_block_build_with_gain_lock():
    """intercarrier=True + gain_lock produces a dual-block dataset: IC from raw mags, features from
    locked mags. Both blocks present, shapes correct, meta flags set."""
    result = _calibrate()
    cal = Calibration(baseline_packets=50)
    baseline, _ = generateStream(numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS,
                                 numFrames=60, perturbationHz=0.0, perturbationDepth=0.0,
                                 cfoHz=0.0, noiseStd=0.005, seed=7)
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
    """train_weapon ic27 and fusion feature modes produce fitted models with correct feature dims."""
    from wavetrace.recognition import train_weapon
    d = weapon_data
    K = d["K"]
    ds_dirs = [save_dataset(ds, tmp_path / f"ds{i}") for i, ds in enumerate(d["datasets"])]

    # ic27: 27-feature inter-carrier block
    head_ic, m_ic = train_weapon(ds_dirs, out_dir=tmp_path / "w_ic", feature_mode="ic27")
    assert m_ic["feature_mode"] == "ic27" and m_ic["n_features"] == 27
    assert m_ic["train_accuracy"] > 0.7
    assert (tmp_path / "w_ic" / "model.joblib").exists()
    assert (tmp_path / "w_ic" / "metrics.json").exists()

    # fusion: hstack(X_ic, X_features) — the ic27 datasets lack gain-locked features (built with
    # gain_lock=None), so X_features is the raw-magnitude 9·K block; width = 27 + 9·K
    head_fu, m_fu = train_weapon(ds_dirs, out_dir=tmp_path / "w_fu",
                                 feature_mode="fusion",
                                 config=ModelConfig(stage="weapon", k=K, backend="mlp"))
    assert m_fu["feature_mode"] == "fusion" and m_fu["n_features"] == 27 + 9 * K
    assert m_fu["train_accuracy"] > 0.7


def test_train_weapon_validates_mode():
    with pytest.raises(ValueError, match="feature_mode"):
        from wavetrace.recognition import train_weapon
        train_weapon([], feature_mode="bad")


def test_intercarrier_roundtrip_and_backcompat(weapon_data, tmp_path):
    ds = weapon_data["datasets"][0]
    assert ds.X_intercarrier.shape == (ds.y.size, 27) and ds.meta["intercarrier"] is True
    reloaded = load_dataset(save_dataset(ds, tmp_path / "w"))
    assert np.array_equal(reloaded.X_intercarrier, ds.X_intercarrier)
    # a dataset built WITHOUT the block round-trips with it absent (pre-P7 compatibility)
    frames, _ = _weapon_recording("sY", "uY", 211, 0.0, duration=2.0)
    labeler = ScriptedLabeler([(0.5, 1.0, True)], label_fn=weapon_label_fn)
    old = build_dataset(frames, weapon_data["result"], None, labeler, window=32, hop=16)
    assert old.X_intercarrier is None
    assert load_dataset(save_dataset(old, tmp_path / "old")).X_intercarrier is None


# ----- 7p-b: variance-threshold + sklearn backends -------------------------------------------------

def test_variance_head_learns_threshold_and_direction(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 0.1, (200, 27)).astype(np.float32)
    y = (np.arange(200) % 2).astype(np.int64)
    X[y == 1, 9] -= 1.0                      # physics direction: weapon BELOW
    head = WeaponHead(_cfg("variance")).fit(X, y)
    assert (head.predict(X) == y).mean() == 1.0
    proba = head.predict_proba(X)
    assert proba.shape == (200, 2) and np.allclose(proba.sum(axis=1), 1.0)
    loaded = WeaponHead.load(head.save(tmp_path / "v.joblib"))
    assert np.allclose(loaded.predict_proba(X), proba)

    X2 = X.copy()
    X2[:, 9] *= -1.0                         # flipped world: weapon ABOVE -> direction is learned
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
    """7p plumbing DoD: variance baseline beats majority on LOGO session+subject AND meets the
    LOCKED tier gate (FP <= 10%, TPR >= 90%) — measured 0.984 acc / TPR 1.0 / FP 0.033."""
    d = weapon_data
    cfg = _cfg("variance", k=d["K"])
    rep = evaluate_weapon(d["X_ic"], d["y"], session_ids=d["sess"], subject_ids=d["subj"],
                          make_head=lambda: WeaponHead(cfg))
    for split in ("session", "subject"):
        r = rep[split]
        assert r["accuracy"] >= 0.95
        assert r["accuracy"] >= r["majority_accuracy"] + 0.30
        assert {"tpr", "fp_rate"} <= r.keys()          # binary rates ride along
    assert rep["verdict"]["verdict"] == "PASS"
    assert rep["verdict"]["tpr"] >= 0.95 and rep["verdict"]["fp_rate"] <= 0.10
    assert sorted(f["group"] for f in rep["session"]["folds"]) == ["s0", "s1", "s2", "s3"]


# ----- 7p-c: torch CNN backend ---------------------------------------------------------------------

def test_cnn_head_trains_roundtrips_deterministic(weapon_data, tmp_path):
    pytest.importorskip("torch")
    d = weapon_data
    X, y = d["X_image"], d["y"]
    head = WeaponHead(_cfg("cnn", k=d["K"], window=32, seed=3)).fit(X, y, epochs=15)
    assert (head.predict(X) == y).mean() > 0.85
    proba = head.predict_proba(X)
    assert proba.shape == (y.size, 2) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(head.predict_proba(X), proba)              # deterministic
    flat = X.reshape(X.shape[0], -1)                              # predict_window seam
    assert np.allclose(head.predict_proba(flat), proba, atol=1e-5)
    loaded = WeaponHead.load(head.save(tmp_path / "cnn.joblib"))
    assert np.allclose(loaded.predict_proba(X), proba, atol=1e-6)


# ----- 7p-d: the two operating modes (user decision 2026-06-11: independent, no cross-gating) ------

def test_weapon_mode_is_standalone(weapon_data, tmp_path):
    # weapon mode classifies EVERY window on its own — no presence verdict in the loop
    d = weapon_data
    head = WeaponHead(_cfg("variance", k=d["K"])).fit(d["X_ic"], d["y"])
    session = mode_session("weapon", head.save(tmp_path / "w.joblib"))
    i_weapon = int(np.flatnonzero(d["y"] == 1)[0])
    i_none = int(np.flatnonzero(d["y"] == 0)[0])
    cls_w, proba_w = session.predict_window(d["X_ic"][i_weapon])
    cls_n, _ = session.predict_window(d["X_ic"][i_none])
    assert (cls_w, cls_n) == (1, 0)
    assert 0.5 <= proba_w <= 1.0


def test_mode_session_validates_mode():
    with pytest.raises(ValueError, match="presence.*weapon"):
        mode_session("gate", "irrelevant")


# ----- 7p-e: soft segment voting -------------------------------------------------------------------

def test_voter_recovers_segment_label_from_noisy_windows():
    # weak per-window head: correct class barely wins on average, often loses per window (Zhou's
    # 51.1% snapshots -> correct walk verdict via the soft vote)
    rng = np.random.default_rng(5)
    voter = SegmentVoter()
    correct = 0
    n = 40
    for _ in range(n):
        p1 = np.clip(0.55 + rng.normal(0, 0.15), 0.0, 1.0)
        correct += p1 > 0.5
        voter.add([1 - p1, p1])
    assert correct / n < 0.75                     # per-window head is genuinely weak
    cls, mean = voter.finalize()
    assert cls == 1                               # the segment vote recovers the true class
    assert len(voter) == 0                        # finalize resets for the next segment


def test_voter_middle_fraction_and_decimation():
    voter = SegmentVoter(middle_fraction=0.5)
    # approach/leave windows (edges) vote class 0; the mid-crossing windows vote class 1
    for p in ([0.9, 0.1],) * 5 + ([0.1, 0.9],) * 6 + ([0.9, 0.1],) * 5:
        voter.add(p)
    assert voter.finalize()[0] == 1               # middle slice isolates the crossing
    full = SegmentVoter()
    for p in ([0.9, 0.1],) * 5 + ([0.1, 0.9],) * 6 + ([0.9, 0.1],) * 5:
        full.add(p)
    assert full.finalize()[0] == 0                # without it, the edges win

    dec = SegmentVoter(decimate=2)
    for p in ([0.2, 0.8], [0.8, 0.2]) * 4:
        dec.add(p)
    cls, mean = dec.finalize()                    # every other window -> only the 0.8-class votes
    assert cls == 1 and mean[1] == pytest.approx(0.8)


def test_voter_correlated_windows_gain_is_nil_and_validation():
    # identical (static-regime) windows: the vote IS the per-window verdict — no lift (rev-5 caveat)
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
    rates = binary_rates(cm)
    assert rates == {"tpr": pytest.approx(0.95), "fp_rate": pytest.approx(0.10)}
    assert tier_verdict({"a": rates})["verdict"] == "PASS"          # boundaries are inclusive

    fail_fp = tier_verdict({"a": {"tpr": 0.95, "fp_rate": 0.101}})
    assert fail_fp["verdict"] == "FAIL" and "fp_rate" in fail_fp["reasons"][0]
    fail_tpr = tier_verdict({"a": {"tpr": 0.899, "fp_rate": 0.05}})
    assert fail_tpr["verdict"] == "FAIL" and "tpr" in fail_tpr["reasons"][0]
    # worst-of-splits: one good split cannot mask a bad one
    mixed = tier_verdict({"good": {"tpr": 1.0, "fp_rate": 0.0},
                          "bad": {"tpr": 0.5, "fp_rate": 0.5}})
    assert mixed["verdict"] == "FAIL" and mixed["tpr"] == 0.5 and mixed["fp_rate"] == 0.5
    with pytest.raises(ValueError, match="2x2"):
        binary_rates(np.zeros((3, 3)))
