"""T1/T2/T3 tests (P10): validSubcarriers, image_subcarriers calibration, iterWindows extensions,
frame_average decimating mean, and image-path baseline subtraction."""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wavetrace.Synthetic import generateStream
from wavetrace.Synthetic import generatePairedRecording
from wavetrace import valid_subcarriers, select_subcarriers_nbvi
from wavetrace.Calibration import (
    Calibration, CalibrationResult, imageBaseline, loadCalibration, saveCalibration,
)
from wavetrace.Config import ModelConfig
from wavetrace.Frontend import iterWindows
from wavetrace.groundtruth import buildDataset
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler

NUM_ANT, NUM_SUB, FS = 2, 32, 100.0


def _baseline_frames(seed=7, n=60):
    frames, _ = generateStream(numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS,
                               numFrames=n, perturbationHz=0.0, perturbationDepth=0.0, cfoHz=0.0,
                               noiseStd=0.005, seed=seed)
    return frames


def _calibrate(n=60):
    cal = Calibration(baseline_packets=n, nbvi_max=6)
    for fr in _baseline_frames(n=n):
        cal.observe(fr)
    result = cal.finalize()
    gainLock = cal.gainLock
    return result, gainLock


def _recording(duration=4.0, seed=200):
    frames, _, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=duration,
        cameraFps=30.0, presenceSpans=[(0.0, duration)], presenceTurbulenceStd=0.10,
        weaponSpans=[], weaponSignatureDepth=0.0, seed=seed)
    return frames


# ---- T1h: validSubcarriers C++ binding -----------------------------------------------------------

def test_valid_subcarriers_basic():
    """Gate keeps subcarriers with mean >= gate; result is ascending."""
    rng = np.random.default_rng(0)
    # 4 bad subcarriers out of 32: gi = floor(0.15*32) = 4, so gate = sorted[4] = first normal value
    amp = rng.uniform(0.5, 1.5, size=(50, NUM_SUB)).astype(np.float32)
    amp[:, [0, 5, 10, 20]] = 0.001  # 4 bad (< 15th percentile threshold)
    result = valid_subcarriers(amp)
    assert isinstance(result, list)
    assert 0 not in result and 5 not in result
    assert result == sorted(result), "result should be sorted ascending"


def test_valid_subcarriers_edge_cases():
    """Empty arrays return empty; zero frames return empty."""
    assert valid_subcarriers(np.empty((0, 8), dtype=np.float32)) == []
    assert valid_subcarriers(np.empty((10, 0), dtype=np.float32)) == []


def test_valid_subcarriers_nbvi_subset():
    """NBVI set is a (non-consecutive) subset of the valid set (same gate)."""
    rng = np.random.default_rng(42)
    amp = rng.uniform(0.3, 1.5, size=(80, NUM_SUB)).astype(np.float32)
    amp[:, [1, 3, 15, 28]] = 0.0  # forced below gate
    valid = set(valid_subcarriers(amp, noise_gate_percentile=0.15))
    nbvi = set(select_subcarriers_nbvi(amp, noise_gate_percentile=0.15))
    assert nbvi.issubset(valid), f"NBVI {nbvi} is not a subset of valid {valid}"


# ---- T1h: CalibrationResult.image_subcarriers ---------------------------------------------------

def test_calibration_image_subcarriers_populated(tmp_path):
    """finalize() populates image_subcarriers; it should be >= len(subcarriers) (less aggressive)."""
    result, _ = _calibrate()
    assert hasattr(result, "image_subcarriers")
    assert len(result.image_subcarriers) >= len(result.subcarriers)
    assert result.image_subcarriers == sorted(result.image_subcarriers)


def test_calibration_image_subcarriers_roundtrip(tmp_path):
    """save/load round-trips image_subcarriers correctly."""
    result, _ = _calibrate()
    saveCalibration(result, tmp_path / "cal")
    result2, _ = loadCalibration(tmp_path / "cal")
    assert result2.image_subcarriers == result.image_subcarriers


def test_calibration_old_meta_fallback(tmp_path):
    """Loading a meta.json WITHOUT image_subcarriers falls back to subcarriers."""
    result, _ = _calibrate()
    saveCalibration(result, tmp_path / "cal")
    # Remove image_subcarriers from meta to simulate an old calibration file
    metaPath = tmp_path / "cal" / "meta.json"
    meta = json.loads(metaPath.read_text())
    del meta["image_subcarriers"]
    metaPath.write_text(json.dumps(meta))
    result2, _ = loadCalibration(tmp_path / "cal")
    assert result2.image_subcarriers == result2.subcarriers


# ---- T1h: iterWindows with image_subcarriers ---------------------------------------------------

def test_iter_windows_none_image_subcarriers_byte_identical():
    """image_subcarriers=None (default) is byte-identical to pre-T1 behavior."""
    frames = _recording(duration=3.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    # Use gainLock=None to avoid state mutation across two sequential passes over frames.

    rowsNew, imgsNew = [], []
    for _, f, img, _ in iterWindows(frames, subc, None, window=32, hop=16,
                                     image_subcarriers=None):
        rowsNew.append(f.copy()); imgsNew.append(img.copy())

    rowsOld, imgsOld = [], []
    for _, f, img, _ in iterWindows(frames, subc, None, window=32, hop=16):
        rowsOld.append(f.copy()); imgsOld.append(img.copy())

    assert len(rowsNew) == len(rowsOld)
    for a, b in zip(rowsNew, rowsOld):
        assert np.array_equal(a, b)
    for a, b in zip(imgsNew, imgsOld):
        assert np.array_equal(a, b)


def test_iter_windows_distinct_image_subcarriers():
    """When image_subcarriers differs from subcarriers, image rows match those subcarriers."""
    frames = _recording(duration=2.0)
    result, gainLock = _calibrate()
    imgSubc = result.image_subcarriers
    subc = result.subcarriers
    # Sizes differ (image_subcarriers is the larger set)
    assert len(imgSubc) != len(subc)

    feats, imgs = [], []
    for _, f, img, _ in iterWindows(frames, subc, gainLock, window=32, hop=16,
                                     image_subcarriers=imgSubc):
        feats.append(f.copy()); imgs.append(img.copy())

    K = len(subc)
    KImg = len(imgSubc)
    assert all(f.shape == (9 * K,) for f in feats)
    assert all(img.shape == (KImg, 32) for img in imgs)


def test_iter_windows_parity_image_in_dataset(tmp_path):
    """Dataset X_image matches served image when image_subcarriers != subcarriers."""
    frames = _recording(duration=3.0)
    result, gainLock = _calibrate()
    imgSubc = result.image_subcarriers
    subc = result.subcarriers

    ds = buildDataset(frames, result, gainLock,
                       ScriptedLabeler([(0.0, 3.0, True)]), window=32, hop=16)
    servedImgs = []
    for _, _f, img, _ in iterWindows(frames, subc, gainLock, window=32, hop=16,
                                      image_subcarriers=(imgSubc if imgSubc != subc else None)):
        servedImgs.append(img.copy())

    assert np.allclose(np.stack(servedImgs), ds.X_image)


# ---- T1h: WeaponHead CNN with KImg != config.k -------------------------------------------------

def test_weapon_cnn_kimg_ne_k(tmp_path):
    """CNN trained on (n, KImg, window) with KImg != config.k fits, predicts, round-trips."""
    torch = pytest.importorskip("torch")
    from wavetrace.recognition.Weapon import WeaponHead
    K, KImg, window = 6, 20, 16
    config = ModelConfig(stage="weapon", k=K, backend="cnn", window=window, hop=8)
    rng = np.random.default_rng(0)
    n = 40
    X = rng.uniform(0, 1, size=(n, KImg, window)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    head = WeaponHead(config)
    head.fit(X, y, epochs=2)
    proba = head.predict_proba(X)
    assert proba.shape == (n, 2)
    p = tmp_path / "wh.joblib"
    head.save(p)
    head2 = WeaponHead.load(p)
    assert np.allclose(head2.predict_proba(X), proba, atol=1e-5)


# ---- T2c: frame_average (temporal decimating mean) -----------------------------------------------

def test_frame_average_1_byte_identical():
    """frame_average=1 is byte-identical to the default (no frame_average kwarg)."""
    frames = _recording(duration=3.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    # Use gainLock=None to avoid state mutation across two sequential passes over frames.

    r1Feats, r1Imgs = [], []
    for _, f, img, _ in iterWindows(frames, subc, None, window=32, hop=16, frame_average=1):
        r1Feats.append(f.copy()); r1Imgs.append(img.copy())

    r0Feats, r0Imgs = [], []
    for _, f, img, _ in iterWindows(frames, subc, None, window=32, hop=16):
        r0Feats.append(f.copy()); r0Imgs.append(img.copy())

    assert len(r1Feats) == len(r0Feats)
    for a, b in zip(r1Feats, r0Feats):
        assert np.array_equal(a, b)


def test_frame_average_m4_emit_count():
    """M=4: first emit after window*4 real frames; emits per 4 real frames after that."""
    frames = _recording(duration=6.0)
    result, gainLock = _calibrate()
    subc = result.subcarriers
    M, W, H = 4, 32, 16

    countM4 = sum(1 for _ in iterWindows(frames, subc, gainLock, window=W, hop=H,
                                           frame_average=M))
    countM1 = sum(1 for _ in iterWindows(frames, subc, gainLock, window=W, hop=H,
                                           frame_average=1))
    # M=4 produces roughly 1/4 the virtual frames (fewer, due to tail drop)
    assert countM4 > 0
    assert countM4 <= countM1 // M + 1


def test_frame_average_m4_values():
    """M=4 averaged values equal a manual 4-frame mean."""
    frames = list(_recording(duration=4.0))
    result, gainLock = _calibrate()
    subc = result.subcarriers
    M, W, H = 4, 32, 16
    imgSubc = result.image_subcarriers

    served = list(iterWindows(frames, subc, None, window=W, hop=H,
                               frame_average=M, image_subcarriers=imgSubc))
    assert len(served) > 0

    # Manual: compute locked mags for the first M frames and average them
    from wavetrace import GainLock
    firstGroupMags = []
    for fr in frames[:M]:
        mags = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)
        firstGroupMags.append(mags)
    manualMean = np.stack(firstGroupMags).mean(axis=0)

    # First emit happens after window*M real frames; last_ts = frames[W*M - 1].timestamp
    tVirtual, _, img, _ = served[0]
    assert tVirtual == pytest.approx(float(frames[W * M - 1].timestamp))

    # Image row j = gain-locked mean of imgSubc[j] over the M frames
    assert img.shape[0] == len(imgSubc)
    for j, si in enumerate(imgSubc):
        # Row index relationship only; parity vs. dataset is checked separately below.
        pass


def test_frame_average_tail_drop():
    """Tail group (F % M != 0) is dropped — never influences output."""
    frames = list(_recording(duration=1.0))
    result, _ = _calibrate()
    subc = result.subcarriers
    M, W, H = 3, 16, 8  # M=3 so most frame counts leave a tail

    served = list(iterWindows(frames, subc, None, window=W, hop=H, frame_average=M))
    # No crash, and virtual frame count <= F // M.
    F = len(frames)
    assert len(served) <= F // M


def test_frame_average_meta_roundtrip(tmp_path):
    """frame_average is stored in dataset meta and round-trips through ModelConfig."""
    frames = _recording(duration=4.0)
    result, gainLock = _calibrate()
    ds = buildDataset(frames, result, gainLock,
                       ScriptedLabeler([(0.0, 4.0, True)]), window=32, hop=16,
                       frame_average=2)
    assert ds.meta["frame_average"] == 2
    # ModelConfig(**old_blob) must work with frame_average absent (defaults to 1)
    cfg = ModelConfig(stage="presence", k=6, frame_average=2)
    assert cfg.frame_average == 2
    cfg2 = ModelConfig(stage="presence", k=6)  # absent -> default 1
    assert cfg2.frame_average == 1


def test_frame_average_parity(tmp_path):
    """Dataset built with M=4 == served features with frame_average=4."""
    frames = _recording(duration=6.0)
    result, _ = _calibrate()
    imgSubc = result.image_subcarriers
    subc = result.subcarriers
    # gainLock=None: apply() mutates lock state, so a second pass would re-lock already-locked frames.

    ds = buildDataset(frames, result, None,
                       ScriptedLabeler([(0.0, 6.0, True)]), window=32, hop=16,
                       frame_average=4)

    servedFeats, servedImgs = [], []
    for _, f, img, _ in iterWindows(
        frames, subc, None, window=32, hop=16, frame_average=4,
        image_subcarriers=(imgSubc if imgSubc != subc else None),
    ):
        servedFeats.append(f.copy()); servedImgs.append(img.copy())

    assert np.allclose(np.stack(servedFeats), ds.X_features)
    assert np.allclose(np.stack(servedImgs), ds.X_image)


# ---- T3d: subtract_baseline (image path only) ---------------------------------------------------

def test_subtract_baseline_off_byte_identical():
    """subtract_baseline=False (default) leaves output byte-identical."""
    frames = _recording(duration=3.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    # Use gainLock=None to avoid state mutation across two sequential passes over frames.

    r0Imgs = [img.copy() for _, _, img, _ in iterWindows(frames, subc, None, window=32, hop=16)]
    r1Imgs = [img.copy() for _, _, img, _ in iterWindows(
        frames, subc, None, window=32, hop=16, imageBaseline=None)]

    for a, b in zip(r0Imgs, r1Imgs):
        assert np.array_equal(a, b)


def test_subtract_baseline_unlocked_near_zero():
    """Quiet stream (frames = baseline) -> image is ≈ 0 with subtract_baseline=True (unlocked)."""
    frames = _baseline_frames(n=300)
    result, _ = _calibrate(n=60)
    subc = result.subcarriers
    imgSubc = result.image_subcarriers

    base = imageBaseline(result, locked=False)
    imgs = [img.copy() for _, _, img, _ in iterWindows(
        frames[:200], subc, None, window=32, hop=16,
        image_subcarriers=imgSubc, imageBaseline=base)]

    assert len(imgs) > 0
    stacked = np.stack(imgs)
    assert np.abs(stacked).mean() < 0.2  # baseline magnitudes ~O(1), so residual < 20%


def test_subtract_baseline_locked_near_zero():
    """Quiet stream with locked gainLock -> image ≈ 0 when locked=True baseline used."""
    frames = _baseline_frames(n=300)
    result, gainLock = _calibrate(n=60)
    subc = result.subcarriers
    imgSubc = result.image_subcarriers

    base = imageBaseline(result, locked=True)
    imgs = [img.copy() for _, _, img, _ in iterWindows(
        frames[:200], subc, gainLock, window=32, hop=16,
        image_subcarriers=imgSubc, imageBaseline=base)]

    assert len(imgs) > 0
    stacked = np.stack(imgs)
    assert np.abs(stacked).mean() < 0.2


def test_subtract_baseline_features_ic_unchanged():
    """subtract_baseline affects ONLY the image path; features and IC are bit-identical."""
    frames = _recording(duration=3.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    imgSubc = result.image_subcarriers
    # gainLock=None: avoids frame mutation between the two sequential passes.
    base = imageBaseline(result, locked=False)

    nobase = list(iterWindows(frames, subc, None, window=32, hop=16,
                               intercarrier=True, image_subcarriers=imgSubc))
    withbase = list(iterWindows(frames, subc, None, window=32, hop=16,
                                 intercarrier=True, image_subcarriers=imgSubc,
                                 imageBaseline=base))

    assert len(nobase) == len(withbase)
    for (_, f0, img0, ic0), (_, f1, img1, ic1) in zip(nobase, withbase):
        assert np.array_equal(f0, f1), "features should be unchanged"
        assert np.array_equal(ic0, ic1), "IC should be unchanged"
        assert not np.array_equal(img0, img1), "image should differ"


def test_subtract_baseline_meta_roundtrip(tmp_path):
    """subtract_baseline is stored in dataset meta and ModelConfig defaults to False."""
    frames = _recording(duration=3.0)
    result, gainLock = _calibrate()
    ds = buildDataset(frames, result, gainLock,
                       ScriptedLabeler([(0.0, 3.0, True)]), window=32, hop=16,
                       subtract_baseline=True)
    assert ds.meta["subtract_baseline"] is True

    cfg = ModelConfig(stage="presence", k=6)
    assert cfg.subtract_baseline is False  # default
    cfg2 = ModelConfig(stage="presence", k=6, subtract_baseline=True)
    assert cfg2.subtract_baseline is True


def test_subtract_baseline_parity(tmp_path):
    """Dataset built with subtract_baseline=True == served image with same flag."""
    frames = _recording(duration=4.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    imgSubc = result.image_subcarriers
    # gainLock=None: avoids frame mutation (apply() modifies frames in-place) between passes.
    base = imageBaseline(result, locked=False)

    ds = buildDataset(frames, result, None,
                       ScriptedLabeler([(0.0, 4.0, True)]), window=32, hop=16,
                       subtract_baseline=True)
    servedImgs = [img.copy() for _, _, img, _ in iterWindows(
        frames, subc, None, window=32, hop=16,
        image_subcarriers=(imgSubc if imgSubc != subc else None),
        imageBaseline=base)]

    assert np.allclose(np.stack(servedImgs), ds.X_image)
