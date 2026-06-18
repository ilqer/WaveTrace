"""T4/P10 — multi-node stacking: demux_by_node, iter_windows_stacked, build_dataset_stacked."""

import numpy as np
import pytest

from fixtures.SyntheticCsi import generateStream
from fixtures.SyntheticRecording import generatePairedRecording
from wavetrace import CsiFrame
from wavetrace.Calibration import Calibration
from wavetrace.Frontend import demux_by_node, iter_windows, iter_windows_stacked
from wavetrace.groundtruth import save_dataset, load_dataset
from wavetrace.groundtruth.DatasetBuilder import build_dataset_stacked
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler

NUM_ANT, NUM_SUB, FS = 2, 32, 100.0
SCALE1 = 1.5  # node 1 amplitude scale for distinguishability


def _baseline_frames(seed=7, n=60):
    frames, _ = generateStream(numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS,
                               numFrames=n, perturbationHz=0.0, perturbationDepth=0.0, cfoHz=0.0,
                               noiseStd=0.005, seed=seed)
    return frames


def _calibrate(seed=7, n=60, nbvi_max=6):
    cal = Calibration(baseline_packets=n, nbvi_max=nbvi_max)
    for fr in _baseline_frames(seed=seed, n=n):
        cal.observe(fr)
    return cal.finalize(), cal.gain_lock


def _single_node_frames(duration=4.0, seed=200):
    frames, _, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=duration,
        cameraFps=30.0, presenceSpans=[(0.0, duration)], weaponSpans=[], seed=seed)
    return frames


def _two_node_frames(duration=4.0, seed=200, scale1=SCALE1, ts_offset1=0.0):
    """Interleaved 2-node stream: node 0 = original, node 1 = grid ×scale1."""
    frames0, _, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=duration,
        cameraFps=30.0, presenceSpans=[(0.0, duration)], weaponSpans=[], seed=seed)
    result = []
    for fr in frames0:
        grid = np.asarray(fr.grid).copy()
        ts = float(fr.timestamp)

        fr0 = CsiFrame(NUM_ANT, NUM_SUB)
        fr0.grid[:, :] = grid
        fr0.timestamp = ts
        fr0.node_id = 0

        fr1 = CsiFrame(NUM_ANT, NUM_SUB)
        fr1.grid[:, :] = grid * scale1
        fr1.timestamp = ts + ts_offset1
        fr1.node_id = 1

        result.append(fr0)
        result.append(fr1)
    return result


# ---- T4d.1 + T4d.2: demux_by_node ---------------------------------------------------

def test_two_node_recording_and_demux():
    """2-node interleaved stream demuxes into two independent lists; node_ids and counts match."""
    frames = _two_node_frames(duration=2.0)
    by_node = demux_by_node(frames)
    assert set(by_node.keys()) == {0, 1}
    assert len(by_node[0]) == len(by_node[1])
    for fr in by_node[0]:
        assert fr.node_id == 0
    for fr in by_node[1]:
        assert fr.node_id == 1
    # Timestamps match between nodes (ts_offset1=0)
    for f0, f1 in zip(by_node[0], by_node[1]):
        assert f0.timestamp == pytest.approx(f1.timestamp)


def test_demux_order_preserved():
    """demux_by_node preserves capture order (ascending timestamps) within each node."""
    frames = _two_node_frames(duration=2.0)
    by_node = demux_by_node(frames)
    for nid in (0, 1):
        ts = [fr.timestamp for fr in by_node[nid]]
        assert ts == sorted(ts)


# ---- T4d.3: stacked shapes -----------------------------------------------------------

def test_stacked_shapes():
    """iter_windows_stacked yields (N·9·K,), (N, K_img, W), (N·27,) shapes for N=2."""
    frames = _two_node_frames(duration=6.0)
    result, _ = _calibrate()
    K = len(result.subcarriers)
    K_img = len(result.image_subcarriers)
    subc = list(result.subcarriers)
    img_subc = result.image_subcarriers
    W = 32

    per_node_calib = {
        0: (subc, img_subc, None, None),
        1: (subc, img_subc, None, None),
    }
    by_node = demux_by_node(frames)
    items = list(iter_windows_stacked(by_node, per_node_calib,
                                      window=W, hop=16, intercarrier=True))
    assert len(items) > 0
    t, feat, img, ic = items[0]
    assert feat.shape == (2 * 9 * K,)
    assert img.shape == (2, K_img, W)
    assert ic.shape == (2 * 27,)


# ---- T4d.4: single-node parity -------------------------------------------------------

def test_single_node_parity():
    """1-node stacked == plain iter_windows: features allclose, timestamps match, image allclose."""
    frames = list(_single_node_frames(duration=4.0))
    result, _ = _calibrate()
    subc = list(result.subcarriers)
    img_subc = result.image_subcarriers
    W, H = 32, 16

    for fr in frames:
        fr.node_id = 0  # ensure single node

    by_node = demux_by_node(frames)
    per_node_calib = {0: (subc, img_subc, None, None)}

    stacked = list(iter_windows_stacked(by_node, per_node_calib,
                                        window=W, hop=H, intercarrier=False))
    # iter_windows reuses its internal buffers; copy each item while iterating.
    plain = [(t, f.copy(), img.copy(), ic) for t, f, img, ic in iter_windows(
        frames, subc, None, window=W, hop=H,
        image_subcarriers=(img_subc if img_subc != subc else None),
    )]

    assert len(stacked) == len(plain) and len(stacked) > 0
    for (ts, fs, is_, _), (tp, fp, ip, _) in zip(stacked, plain):
        assert ts == pytest.approx(tp)
        assert np.allclose(fs, fp)
        assert np.allclose(is_[0], ip)


# ---- T4d.5: timestamp shift > node_tolerance → ValueError ---------------------------

def test_timestamp_shift_raises():
    """Timestamp gap > node_tolerance raises ValueError (node de-sync)."""
    frames = _two_node_frames(duration=4.0, ts_offset1=1.0)  # 1 s >> 0.05 s tolerance
    result, _ = _calibrate()
    subc = list(result.subcarriers)
    img_subc = result.image_subcarriers
    by_node = demux_by_node(frames)
    per_node_calib = {
        0: (subc, img_subc, None, None),
        1: (subc, img_subc, None, None),
    }
    with pytest.raises(ValueError, match="not node-synced"):
        list(iter_windows_stacked(by_node, per_node_calib,
                                  window=32, hop=16, node_tolerance=0.05))


# ---- T4d.6: unequal lengths → stops at shorter, no error ----------------------------

def test_unequal_lengths_stops_at_shorter():
    """If one node has fewer frames, iteration stops cleanly at the shorter stream."""
    frames_long = list(_single_node_frames(duration=6.0))
    frames_short = list(_single_node_frames(duration=2.0))
    result, _ = _calibrate()
    subc = list(result.subcarriers)
    img_subc = result.image_subcarriers

    per_node_frames = {0: frames_long, 1: frames_short}
    per_node_calib = {
        0: (subc, img_subc, None, None),
        1: (subc, img_subc, None, None),
    }
    items_mixed = list(iter_windows_stacked(per_node_frames, per_node_calib,
                                            window=32, hop=16))
    items_short = list(iter_windows(frames_short, subc, None, window=32, hop=16))
    assert len(items_mixed) == len(items_short) and len(items_mixed) > 0


# ---- T4d.7: mismatched K → ValueError -----------------------------------------------

def test_mismatched_k_raises():
    """Nodes with different NBVI subcarrier count K raise ValueError before iteration."""
    result_a, _ = _calibrate(nbvi_max=6)
    result_b, _ = _calibrate(nbvi_max=2)

    if len(result_a.subcarriers) == len(result_b.subcarriers):
        pytest.skip("nbvi_max difference did not produce different K in this environment")

    frames = list(_single_node_frames())
    per_node_frames = {0: frames, 1: frames}
    per_node_calib = {
        0: (list(result_a.subcarriers), result_a.image_subcarriers, None, None),
        1: (list(result_b.subcarriers), result_b.image_subcarriers, None, None),
    }
    with pytest.raises(ValueError, match="different K"):
        list(iter_windows_stacked(per_node_frames, per_node_calib, window=32, hop=16))


# ---- T4d.8: CNN on (n, N, K_img, W) + pre-P10 legacy blob ---------------------------

def test_cnn_multichannel_fit_predict_roundtrip(tmp_path):
    """CNN accepts (n, N, K_img, W) images: fit/predict/save→load→identical probas."""
    pytest.importorskip("torch")
    import joblib
    from wavetrace.Config import ModelConfig
    from wavetrace.recognition.Weapon import WeaponHead

    K, K_img, W, N = 6, 20, 16, 2
    config = ModelConfig(stage="weapon", k=K, backend="cnn", window=W, hop=8)
    rng = np.random.default_rng(1)
    n = 40
    y = np.array([0] * 20 + [1] * 20, dtype=np.int64)

    # 2-node (N=2 channels) model — P10 path.
    X_2ch = rng.uniform(0, 1, size=(n, N, K_img, W)).astype(np.float32)
    head = WeaponHead(config)
    head.fit(X_2ch, y, epochs=2)
    proba = head.predict_proba(X_2ch)
    assert proba.shape == (n, 2)

    p = tmp_path / "wh_multi.joblib"
    head.save(p)
    head2 = WeaponHead.load(p)
    assert np.allclose(head2.predict_proba(X_2ch), proba, atol=1e-5)

    # Pre-P10 blob: image_shape is a 2-tuple (K_img, W); model was single-channel (in_channels=1).
    # Simulate by training a 1-channel model and stripping the channel dim from image_shape.
    X_1ch = rng.uniform(0, 1, size=(n, K_img, W)).astype(np.float32)
    head_1ch = WeaponHead(config)
    head_1ch.fit(X_1ch, y, epochs=2)
    p_1ch = tmp_path / "wh_1ch.joblib"
    head_1ch.save(p_1ch)
    proba_1ch = head_1ch.predict_proba(X_1ch)

    blob = joblib.load(p_1ch)
    blob["image_shape"] = (K_img, W)  # 2-tuple: pre-P10 file format
    p_legacy = tmp_path / "wh_legacy.joblib"
    joblib.dump(blob, p_legacy)
    head3 = WeaponHead.load(p_legacy)
    assert head3._image_shape == (1, K_img, W)
    assert np.allclose(head3.predict_proba(X_1ch), proba_1ch, atol=1e-5)


# ---- T4d.9: build_dataset_stacked end-to-end ----------------------------------------

def test_build_dataset_stacked_end_to_end(tmp_path):
    """build_dataset_stacked: shapes, meta keys, save→load round-trip with frame_average=2."""
    frames = _two_node_frames(duration=6.0)
    result, _ = _calibrate()
    calibrations = {0: (result, None), 1: (result, None)}
    W, H = 32, 16
    K = len(result.subcarriers)
    K_img = len(result.image_subcarriers)
    N = 2

    label = ScriptedLabeler([(0.0, 6.0, True)])
    ds = build_dataset_stacked(frames, calibrations, label,
                               window=W, hop=H, frame_average=2, subtract_baseline=False)

    assert ds.X_features.shape[1] == N * 9 * K
    assert ds.X_image.shape[1:] == (N, K_img, W)
    assert ds.y.shape[0] == ds.X_features.shape[0]
    assert ds.meta["num_nodes"] == N
    assert ds.meta["node_ids"] == [0, 1]
    assert ds.meta["K"] == K
    assert ds.meta["K_img"] == K_img
    assert ds.meta["frame_average"] == 2
    assert ds.meta["subtract_baseline"] is False

    out = save_dataset(ds, tmp_path / "stacked")
    ds2 = load_dataset(out)
    assert np.array_equal(ds2.X_features, ds.X_features)
    assert np.array_equal(ds2.X_image, ds.X_image)
    assert np.array_equal(ds2.y, ds.y)
