"""Phase 0 golden characterization tests (REFACTOR_PLAN.md Phase 0 step 5).

Pin CURRENT behaviour of the full calibrate -> collect -> train -> infer chain through public entry
points only (wavetrace.Cli.calibrateSource / collectSource / runInference, wavetrace.recognition.
trainPresence / trainWeapon) over the small committed fixture recordings in tests/golden/fixtures/.
Every later refactor phase (1-6) must leave these values unchanged — that IS the definition of
"behaviour-preserving" for this project. If a later phase's diff makes one of these fail, either the
phase broke something or the snapshot in tests/golden/expected/ needs a deliberate, reviewed
re-generation — never a silent tolerance widening. Both tests/golden/fixtures/ (the input recordings)
and tests/golden/expected/ (the pinned outputs) were produced once, offline, by scripts equivalent to
this file's own calibrate/collect/train/infer calls with fixed seeds (see fixture_params.json for the
exact generation parameters); re-running those calls is exactly what this file does on every test run.

Deterministic and fully offline: fixed seeds throughout (fixture generation, MLPClassifier
random_state, the weapon 'variance' backend has no RNG at all). Verified stable across repeated runs
in this environment before being committed (two consecutive runs byte-identical).

Coverage note: `capture` (raw UDP/serial acquisition) is hardware-only and is NOT characterized here
-- everything downstream of a saved recording is. That is the largest deterministic sub-chain
reachable without hardware or a camera.
"""

import io
import json
from pathlib import Path

import numpy as np

from wavetrace.Calibration import loadCalibration
from wavetrace.Cli import calibrateSource, collectSource, runInference
from wavetrace.Source import RecordingSource
from wavetrace.output import JsonlPublisher
from wavetrace.recognition import trainPresence, trainWeapon

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = Path(__file__).parent / "expected"
PARAMS = json.loads((FIXTURES / "fixture_params.json").read_text())

WINDOW, HOP = 32, 16


def _calibrate(tmp_path):
    calibrateSource(RecordingSource(FIXTURES / "baseline_recording"), tmp_path / "cal",
                     baseline_packets=PARAMS["baseline_num_frames"])
    return tmp_path / "cal"


def test_calibration_pins_subcarriers_and_baseline(tmp_path):
    cal_dir = _calibrate(tmp_path)
    result, gain_lock = loadCalibration(cal_dir)
    expected_subcarriers = np.load(EXPECTED / "calibration_subcarriers.npy")
    expected_baseline = np.load(EXPECTED / "calibration_baseline_mag.npy")
    assert np.array_equal(np.asarray(result.subcarriers, dtype=np.int64), expected_subcarriers)
    np.testing.assert_allclose(result.baseline_mag, expected_baseline, rtol=1e-6)
    assert gain_lock is not None and gain_lock.locked


def test_collect_presence_pins_features_and_labels(tmp_path):
    cal_dir = _calibrate(tmp_path)
    _, ds = collectSource(RecordingSource(FIXTURES / "main_recording"), cal_dir, tmp_path / "ds_p",
                           PARAMS["presence_spans"], stage="presence", window=WINDOW, hop=HOP)
    expected_x = np.load(EXPECTED / "presence_X_features.npy")
    expected_y = np.load(EXPECTED / "presence_y.npy")
    # Both classes present -- a 1-class window set would silently make PresenceHead.fit unusable.
    assert set(np.unique(ds.y).tolist()) == {0, 1}
    assert np.array_equal(ds.y, expected_y)
    np.testing.assert_allclose(ds.X_features, expected_x, rtol=1e-6)


def test_collect_weapon_pins_intercarrier_and_labels(tmp_path):
    cal_dir = _calibrate(tmp_path)
    _, ds = collectSource(RecordingSource(FIXTURES / "main_recording"), cal_dir, tmp_path / "ds_w",
                           PARAMS["weapon_spans"], stage="weapon", window=WINDOW, hop=HOP)
    expected_x = np.load(EXPECTED / "weapon_X_intercarrier.npy")
    expected_y = np.load(EXPECTED / "weapon_y.npy")
    assert set(np.unique(ds.y).tolist()) == {0, 1}
    assert np.array_equal(ds.y, expected_y)
    assert ds.X_intercarrier.shape[1] == 27  # the ic27 block (see REFACTOR_PLAN.md §1.8 Axis A)
    np.testing.assert_allclose(ds.X_intercarrier, expected_x, rtol=1e-6)


def test_train_presence_pins_accuracy(tmp_path):
    cal_dir = _calibrate(tmp_path)
    collectSource(RecordingSource(FIXTURES / "main_recording"), cal_dir, tmp_path / "ds_p",
                  PARAMS["presence_spans"], stage="presence", window=WINDOW, hop=HOP)
    _, metrics = trainPresence([tmp_path / "ds_p"], out_dir=tmp_path / "model_p")
    expected = json.loads((EXPECTED / "train_metrics.json").read_text())
    np.testing.assert_allclose(metrics["train_accuracy"], expected["presence_train_accuracy"], rtol=1e-6)


def test_train_weapon_pins_accuracy(tmp_path):
    # feature_mode defaults to 'ic27' -> backend defaults to 'variance' (deterministic, no RNG at
    # all -- see wavetrace/recognition/Weapon.py _fitVariance), unlike the presence 'mlp' default.
    cal_dir = _calibrate(tmp_path)
    collectSource(RecordingSource(FIXTURES / "main_recording"), cal_dir, tmp_path / "ds_w",
                  PARAMS["weapon_spans"], stage="weapon", window=WINDOW, hop=HOP)
    _, metrics = trainWeapon([tmp_path / "ds_w"], out_dir=tmp_path / "model_w")
    expected = json.loads((EXPECTED / "train_metrics.json").read_text())
    assert metrics["backend"] == "variance"
    np.testing.assert_allclose(metrics["train_accuracy"], expected["weapon_train_accuracy"], rtol=1e-6)


def test_run_inference_presence_pins_verdicts(tmp_path):
    """The full chain end to end: calibrate -> collect -> train -> infer -> publish."""
    cal_dir = _calibrate(tmp_path)
    collectSource(RecordingSource(FIXTURES / "main_recording"), cal_dir, tmp_path / "ds_p",
                  PARAMS["presence_spans"], stage="presence", window=WINDOW, hop=HOP)
    trainPresence([tmp_path / "ds_p"], out_dir=tmp_path / "model_p")
    buf = io.StringIO()
    with JsonlPublisher(buf, mode="presence") as pub:
        results = runInference(RecordingSource(FIXTURES / "main_recording"), cal_dir,
                                tmp_path / "model_p" / "model.joblib", "presence", pub)
    conf = np.array([r.confidence for r in results], dtype=np.float64)
    cls = np.array([r.class_id for r in results], dtype=np.int64)
    expected_conf = np.load(EXPECTED / "infer_presence_confidence.npy")
    expected_cls = np.load(EXPECTED / "infer_presence_class.npy")
    assert np.array_equal(cls, expected_cls)
    np.testing.assert_allclose(conf, expected_conf, rtol=1e-6)
    # Every published JSONL line parses and carries the wire schema (output/Publisher.py).
    lines = [json.loads(l) for l in buf.getvalue().strip().split("\n")]
    assert len(lines) == len(results)
    assert all(set(l) == {"t", "class", "conf", "mode", "bbox", "keypoints"} for l in lines)


def test_run_inference_weapon_pins_verdicts(tmp_path):
    cal_dir = _calibrate(tmp_path)
    collectSource(RecordingSource(FIXTURES / "main_recording"), cal_dir, tmp_path / "ds_w",
                  PARAMS["weapon_spans"], stage="weapon", window=WINDOW, hop=HOP)
    trainWeapon([tmp_path / "ds_w"], out_dir=tmp_path / "model_w")
    buf = io.StringIO()
    with JsonlPublisher(buf, mode="weapon") as pub:
        results = runInference(RecordingSource(FIXTURES / "main_recording"), cal_dir,
                                tmp_path / "model_w" / "model.joblib", "weapon", pub)
    conf = np.array([r.confidence for r in results], dtype=np.float64)
    cls = np.array([r.class_id for r in results], dtype=np.int64)
    expected_conf = np.load(EXPECTED / "infer_weapon_confidence.npy")
    expected_cls = np.load(EXPECTED / "infer_weapon_class.npy")
    assert np.array_equal(cls, expected_cls)
    np.testing.assert_allclose(conf, expected_conf, rtol=1e-6)
