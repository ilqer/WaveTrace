import json

import numpy as np
import pytest

from wavetrace.synthetic import generate_stream
from wavetrace.synthetic import generate_paired_recording
from wavetrace import CsiFrame, valid_subcarriers, select_subcarriers_nbvi
from wavetrace.calibration import (
    Calibration, build_image_baseline, load_calibration, save_calibration,
)
from wavetrace.config import ModelConfig
from wavetrace.frontend import iter_windows
from wavetrace.groundtruth import build_dataset
from wavetrace.groundtruth.camera_labeler import ScriptedLabeler

NUM_ANT, NUM_SUB, FS = 2, 32, 100.0


def _baseline_frames(seed=7, n=60):
    frames, _ = generate_stream(num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS,
                               num_frames=n, perturbation_hz=0.0, perturbation_depth=0.0, cfo_hz=0.0,
                               noise_std=0.005, seed=seed)
    return frames


def _calibrate(n=60):
    cal = Calibration(baseline_packets=n, nbvi_max=6)
    for fr in _baseline_frames(n=n):
        cal.observe(fr)
    result = cal.finalize()
    gain_lock = cal.gain_lock
    return result, gain_lock


def _recording(duration=4.0, seed=200):
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=duration,
        camera_fps=30.0, presence_spans=[(0.0, duration)], presence_turbulence_std=0.10,
        weapon_spans=[], weapon_signature_depth=0.0, seed=seed)
    return frames


def test_valid_subcarriers_basic():
    """Gate keeps subcarriers with mean >= gate; result is ascending."""
    rng = np.random.default_rng(0)
    # 4 bad subcarriers out of 32: gi = floor(0.15*32) = 4, so gate = sorted[4] = first normal value
    amp = rng.uniform(0.5, 1.5, size=(50, NUM_SUB)).astype(np.float32)
    amp[:, [0, 5, 10, 20]] = 0.001
    result = valid_subcarriers(amp)
    assert isinstance(result, list)
    assert 0 not in result and 5 not in result
    assert result == sorted(result), "result should be sorted ascending"


def test_valid_subcarriers_edge_cases():
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


def test_calibration_image_subcarriers_populated(tmp_path):
    """finalize() populates image_subcarriers; it should be >= len(subcarriers) (less aggressive)."""
    result, _ = _calibrate()
    assert hasattr(result, "image_subcarriers")
    assert len(result.image_subcarriers) >= len(result.subcarriers)
    assert result.image_subcarriers == sorted(result.image_subcarriers)


def test_calibration_image_subcarriers_roundtrip(tmp_path):
    result, _ = _calibrate()
    save_calibration(result, tmp_path / "cal")
    result2, _ = load_calibration(tmp_path / "cal")
    assert result2.image_subcarriers == result.image_subcarriers


def test_calibration_old_meta_fallback(tmp_path):
    """Loading a meta.json WITHOUT image_subcarriers falls back to subcarriers."""
    result, _ = _calibrate()
    save_calibration(result, tmp_path / "cal")
    # Remove image_subcarriers from meta to simulate an old calibration file
    meta_path = tmp_path / "cal" / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["image_subcarriers"]
    meta_path.write_text(json.dumps(meta))
    result2, _ = load_calibration(tmp_path / "cal")
    assert result2.image_subcarriers == result2.subcarriers


def test_iter_windows_none_image_subcarriers_byte_identical():
    frames = _recording(duration=3.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    # Use gain_lock=None to avoid state mutation across two sequential passes over frames.

    explicit_features, explicit_images = [], []
    for _, f, img, _ in iter_windows(frames, subc, None, window=32, hop=16,
                                     image_subcarriers=None):
        explicit_features.append(f.copy()); explicit_images.append(img.copy())

    default_features, default_images = [], []
    for _, f, img, _ in iter_windows(frames, subc, None, window=32, hop=16):
        default_features.append(f.copy()); default_images.append(img.copy())

    assert len(explicit_features) == len(default_features)
    for a, b in zip(explicit_features, default_features):
        assert np.array_equal(a, b)
    for a, b in zip(explicit_images, default_images):
        assert np.array_equal(a, b)


def test_iter_windows_distinct_image_subcarriers():
    """When image_subcarriers differs from subcarriers, image rows match those subcarriers."""
    frames = _recording(duration=2.0)
    result, gain_lock = _calibrate()
    image_subcarriers = result.image_subcarriers
    subc = result.subcarriers
    # Sizes differ (image_subcarriers is the larger set)
    assert len(image_subcarriers) != len(subc)

    feats, imgs = [], []
    for _, f, img, _ in iter_windows(frames, subc, gain_lock, window=32, hop=16,
                                     image_subcarriers=image_subcarriers):
        feats.append(f.copy()); imgs.append(img.copy())

    K = len(subc)
    K_img = len(image_subcarriers)
    assert all(f.shape == (9 * K,) for f in feats)
    assert all(img.shape == (K_img, 32) for img in imgs)


def test_iter_windows_parity_image_in_dataset(tmp_path):
    """Dataset X_image matches served image when image_subcarriers != subcarriers."""
    frames = _recording(duration=3.0)
    result, gain_lock = _calibrate()
    image_subcarriers = result.image_subcarriers
    subc = result.subcarriers

    ds = build_dataset(frames, result, gain_lock,
                       ScriptedLabeler([(0.0, 3.0, True)]), window=32, hop=16)
    served_images = []
    for _, _f, img, _ in iter_windows(frames, subc, gain_lock, window=32, hop=16,
                                      image_subcarriers=(image_subcarriers
                                                         if image_subcarriers != subc else None)):
        served_images.append(img.copy())

    assert np.allclose(np.stack(served_images), ds.X_image)


def test_weapon_cnn_kimg_ne_k(tmp_path):
    """CNN trained on (n, K_img, window) with K_img != config.k fits, predicts, round-trips."""
    pytest.importorskip("torch")
    from wavetrace.adapters.recognition.heads import build_weapon_head, load_weapon_head
    K, K_img, window = 6, 20, 16
    config = ModelConfig(stage="weapon", k=K, backend="cnn", window=window, hop=8)
    rng = np.random.default_rng(0)
    n = 40
    X = rng.uniform(0, 1, size=(n, K_img, window)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    head = build_weapon_head(config)
    head.fit(X, y, epochs=2)
    proba = head.predict_proba(X)
    assert proba.shape == (n, 2)
    p = tmp_path / "wh.joblib"
    head.save(p)
    head2 = load_weapon_head(p)
    assert np.allclose(head2.predict_proba(X), proba, atol=1e-5)


def test_frame_average_1_byte_identical():
    frames = _recording(duration=3.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    # Use gain_lock=None to avoid state mutation across two sequential passes over frames.

    explicit_features, explicit_images = [], []
    for _, f, img, _ in iter_windows(frames, subc, None, window=32, hop=16, frame_average=1):
        explicit_features.append(f.copy()); explicit_images.append(img.copy())

    default_features, default_images = [], []
    for _, f, img, _ in iter_windows(frames, subc, None, window=32, hop=16):
        default_features.append(f.copy()); default_images.append(img.copy())

    assert len(explicit_features) == len(default_features)
    for a, b in zip(explicit_features, default_features):
        assert np.array_equal(a, b)


def test_frame_average_m4_emit_count():
    """M=4: first emit after window*4 real frames; emits per 4 real frames after that."""
    frames = _recording(duration=6.0)
    result, gain_lock = _calibrate()
    subc = result.subcarriers
    M, W, H = 4, 32, 16

    averaged_window_count = sum(1 for _ in iter_windows(frames, subc, gain_lock, window=W, hop=H,
                                                        frame_average=M))
    unaveraged_window_count = sum(1 for _ in iter_windows(frames, subc, gain_lock, window=W, hop=H,
                                                          frame_average=1))
    # M=4 produces roughly 1/4 the virtual frames (fewer, due to tail drop)
    assert averaged_window_count > 0
    assert averaged_window_count <= unaveraged_window_count // M + 1


def test_frame_average_m4_first_window():
    """M=4 averaging: the first window lands on real frame W*M - 1, and its features and image equal
    the M=1 path fed one virtual frame per group holding that group's mean magnitude."""
    frames = list(_recording(duration=4.0))
    result, _ = _calibrate()
    M, W, H = 4, 32, 16
    window_options = dict(window=W, hop=H, image_subcarriers=result.image_subcarriers)

    averaged_windows = iter_windows(frames, result.subcarriers, None, frame_average=M, **window_options)
    window_end_timestamp_s, features, image, _ = next(averaged_windows)
    features, image = np.array(features), np.array(image)  # yielded buffers are reused by the next push
    assert window_end_timestamp_s == pytest.approx(float(frames[W * M - 1].timestamp))
    assert image.shape[0] == len(result.image_subcarriers)

    virtual_frames = []
    for group_start in range(0, W * M, M):
        magnitude_sum = np.zeros(NUM_SUB, dtype=np.float32)
        for frame in frames[group_start:group_start + M]:
            magnitude_sum += np.abs(np.asarray(frame.grid)).mean(axis=0).astype(np.float32)
        virtual_frame = CsiFrame(NUM_ANT, NUM_SUB)
        virtual_frame.grid[:, :] = magnitude_sum / M  # identical antenna rows: their mean is the group mean
        virtual_frame.timestamp = frames[group_start + M - 1].timestamp
        virtual_frames.append(virtual_frame)
    reference_windows = iter_windows(virtual_frames, result.subcarriers, None, **window_options)
    _, expected_features, expected_image, _ = next(reference_windows)

    np.testing.assert_allclose(features, expected_features, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(image, expected_image, rtol=1e-5, atol=1e-6)


def test_frame_average_tail_drop():
    """Tail group (F % M != 0) is dropped — never influences output."""
    frames = list(_recording(duration=1.0))
    result, _ = _calibrate()
    subc = result.subcarriers
    M, W, H = 3, 16, 8  # M=3 so most frame counts leave a tail

    served = list(iter_windows(frames, subc, None, window=W, hop=H, frame_average=M))
    F = len(frames)
    assert len(served) <= F // M


def test_frame_average_meta_roundtrip(tmp_path):
    frames = _recording(duration=4.0)
    result, gain_lock = _calibrate()
    ds = build_dataset(frames, result, gain_lock,
                       ScriptedLabeler([(0.0, 4.0, True)]), window=32, hop=16,
                       frame_average=2)
    assert ds.meta["frame_average"] == 2
    # ModelConfig(**old_blob) must work with frame_average absent (defaults to 1)
    cfg = ModelConfig(stage="presence", k=6, frame_average=2)
    assert cfg.frame_average == 2
    cfg2 = ModelConfig(stage="presence", k=6)
    assert cfg2.frame_average == 1


def test_frame_average_parity(tmp_path):
    """Dataset built with M=4 == served features with frame_average=4."""
    frames = _recording(duration=6.0)
    result, _ = _calibrate()
    image_subcarriers = result.image_subcarriers
    subc = result.subcarriers
    # gain_lock=None: apply() mutates lock state, so a second pass would re-lock already-locked frames.

    ds = build_dataset(frames, result, None,
                       ScriptedLabeler([(0.0, 6.0, True)]), window=32, hop=16,
                       frame_average=4)

    served_features, served_images = [], []
    for _, f, img, _ in iter_windows(
        frames, subc, None, window=32, hop=16, frame_average=4,
        image_subcarriers=(image_subcarriers if image_subcarriers != subc else None),
    ):
        served_features.append(f.copy()); served_images.append(img.copy())

    assert np.allclose(np.stack(served_features), ds.X_features)
    assert np.allclose(np.stack(served_images), ds.X_image)


def test_subtract_baseline_off_byte_identical():
    frames = _recording(duration=3.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    # Use gain_lock=None to avoid state mutation across two sequential passes over frames.

    default_images = [img.copy() for _, _, img, _ in iter_windows(frames, subc, None, window=32, hop=16)]
    explicit_images = [img.copy() for _, _, img, _ in iter_windows(
        frames, subc, None, window=32, hop=16, image_baseline=None)]

    for a, b in zip(default_images, explicit_images):
        assert np.array_equal(a, b)


def test_subtract_baseline_unlocked_near_zero():
    """Quiet stream (frames = baseline) -> image is ≈ 0 with subtract_baseline=True (unlocked)."""
    frames = _baseline_frames(n=300)
    result, _ = _calibrate(n=60)
    subc = result.subcarriers
    image_subcarriers = result.image_subcarriers

    base = build_image_baseline(result, locked=False)
    imgs = [img.copy() for _, _, img, _ in iter_windows(
        frames[:200], subc, None, window=32, hop=16,
        image_subcarriers=image_subcarriers, image_baseline=base)]

    assert len(imgs) > 0
    stacked = np.stack(imgs)
    assert np.abs(stacked).mean() < 0.2  # baseline magnitudes ~O(1), so residual < 20%


def test_subtract_baseline_locked_near_zero():
    """Quiet stream with locked gain_lock -> image ≈ 0 when locked=True baseline used."""
    frames = _baseline_frames(n=300)
    result, gain_lock = _calibrate(n=60)
    subc = result.subcarriers
    image_subcarriers = result.image_subcarriers

    base = build_image_baseline(result, locked=True)
    imgs = [img.copy() for _, _, img, _ in iter_windows(
        frames[:200], subc, gain_lock, window=32, hop=16,
        image_subcarriers=image_subcarriers, image_baseline=base)]

    assert len(imgs) > 0
    stacked = np.stack(imgs)
    assert np.abs(stacked).mean() < 0.2


def test_subtract_baseline_features_ic_unchanged():
    """subtract_baseline affects ONLY the image path; features and IC are bit-identical."""
    frames = _recording(duration=3.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    image_subcarriers = result.image_subcarriers
    # gain_lock=None: avoids frame mutation between the two sequential passes.
    base = build_image_baseline(result, locked=False)

    nobase = list(iter_windows(frames, subc, None, window=32, hop=16,
                               intercarrier=True, image_subcarriers=image_subcarriers))
    withbase = list(iter_windows(frames, subc, None, window=32, hop=16,
                                 intercarrier=True, image_subcarriers=image_subcarriers,
                                 image_baseline=base))

    assert len(nobase) == len(withbase)
    for (_, f0, img0, ic0), (_, f1, img1, ic1) in zip(nobase, withbase):
        assert np.array_equal(f0, f1), "features should be unchanged"
        assert np.array_equal(ic0, ic1), "IC should be unchanged"
        assert not np.array_equal(img0, img1), "image should differ"


def test_subtract_baseline_meta_roundtrip(tmp_path):
    frames = _recording(duration=3.0)
    result, gain_lock = _calibrate()
    ds = build_dataset(frames, result, gain_lock,
                       ScriptedLabeler([(0.0, 3.0, True)]), window=32, hop=16,
                       subtract_baseline=True)
    assert ds.meta["subtract_baseline"] is True

    cfg = ModelConfig(stage="presence", k=6)
    assert cfg.subtract_baseline is False
    cfg2 = ModelConfig(stage="presence", k=6, subtract_baseline=True)
    assert cfg2.subtract_baseline is True


def test_subtract_baseline_parity(tmp_path):
    """Dataset built with subtract_baseline=True == served image with same flag."""
    frames = _recording(duration=4.0)
    result, _ = _calibrate()
    subc = result.subcarriers
    image_subcarriers = result.image_subcarriers
    # gain_lock=None: avoids frame mutation (apply() modifies frames in-place) between passes.
    base = build_image_baseline(result, locked=False)

    ds = build_dataset(frames, result, None,
                       ScriptedLabeler([(0.0, 4.0, True)]), window=32, hop=16,
                       subtract_baseline=True)
    served_images = [img.copy() for _, _, img, _ in iter_windows(
        frames, subc, None, window=32, hop=16,
        image_subcarriers=(image_subcarriers if image_subcarriers != subc else None),
        image_baseline=base)]

    assert np.allclose(np.stack(served_images), ds.X_image)
