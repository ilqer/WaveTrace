import numpy as np
import pytest

from wavetrace.synthetic import generate_paired_recording
from wavetrace import CsiFrame
from wavetrace.frontend import group_frames_by_node, iter_windows, iter_stacked_windows
from wavetrace.groundtruth import save_dataset, load_dataset
from wavetrace.groundtruth.dataset_builder import build_stacked_dataset
from wavetrace.groundtruth.camera_labeler import ScriptedLabeler
from fixtures.synthetic_csi import calibrate_quiet_baseline

NUM_ANTENNAS, NUM_SUBCARRIERS, SAMPLE_RATE_HZ = 2, 32, 100.0
NODE_1_AMPLITUDE_SCALE = 1.5


def _calibrate(nbvi_max=6):
    return calibrate_quiet_baseline(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        num_frames=60, seed=7, baseline_packets=60, nbvi_max=nbvi_max)


def _single_node_frames(duration_s=4.0, seed=200):
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        duration_s=duration_s, camera_frame_rate_hz=30.0, presence_spans=[(0.0, duration_s)],
        weapon_spans=[], seed=seed)
    return frames


def _two_node_frames(duration_s=4.0, seed=200, node_1_amplitude_scale=NODE_1_AMPLITUDE_SCALE,
                     node_1_timestamp_offset_s=0.0):
    """Interleaved 2-node stream: node 0 = original, node 1 = grid scaled by node_1_amplitude_scale."""
    frames_node_0, _, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        duration_s=duration_s, camera_frame_rate_hz=30.0, presence_spans=[(0.0, duration_s)],
        weapon_spans=[], seed=seed)
    result = []
    for frame in frames_node_0:
        grid = np.asarray(frame.grid).copy()
        timestamp_s = float(frame.timestamp)

        frame_node_0 = CsiFrame(NUM_ANTENNAS, NUM_SUBCARRIERS)
        frame_node_0.grid[:, :] = grid
        frame_node_0.timestamp = timestamp_s
        frame_node_0.node_id = 0

        frame_node_1 = CsiFrame(NUM_ANTENNAS, NUM_SUBCARRIERS)
        frame_node_1.grid[:, :] = grid * node_1_amplitude_scale
        frame_node_1.timestamp = timestamp_s + node_1_timestamp_offset_s
        frame_node_1.node_id = 1

        result.append(frame_node_0)
        result.append(frame_node_1)
    return result


def test_two_node_recording_and_demux():
    frames = _two_node_frames(duration_s=2.0)
    frames_by_node = group_frames_by_node(frames)
    assert set(frames_by_node.keys()) == {0, 1}
    assert len(frames_by_node[0]) == len(frames_by_node[1])
    for frame in frames_by_node[0]:
        assert frame.node_id == 0
    for frame in frames_by_node[1]:
        assert frame.node_id == 1
    for frame_node_0, frame_node_1 in zip(frames_by_node[0], frames_by_node[1]):
        assert frame_node_0.timestamp == pytest.approx(frame_node_1.timestamp)


def test_demux_order_preserved():
    """group_frames_by_node preserves capture order (ascending timestamps) within each node."""
    frames = _two_node_frames(duration_s=2.0)
    frames_by_node = group_frames_by_node(frames)
    for node_id in (0, 1):
        timestamps_s = [frame.timestamp for frame in frames_by_node[node_id]]
        assert timestamps_s == sorted(timestamps_s)


def test_stacked_shapes():
    """iter_stacked_windows yields (nodes·9·subcarriers,), (nodes, image_subcarriers, window), (nodes·27,)."""
    frames = _two_node_frames(duration_s=6.0)
    result, _ = _calibrate()
    subcarrier_count = len(result.subcarriers)
    image_subcarrier_count = len(result.image_subcarriers)
    subcarriers = list(result.subcarriers)
    image_subcarriers = result.image_subcarriers
    window_frames = 32

    calibration_by_node = {
        0: (subcarriers, image_subcarriers, None, None),
        1: (subcarriers, image_subcarriers, None, None),
    }
    frames_by_node = group_frames_by_node(frames)
    items = list(iter_stacked_windows(frames_by_node, calibration_by_node,
                                      window=window_frames, hop=16, intercarrier=True))
    assert len(items) > 0
    timestamp_s, features, image, intercarrier_features = items[0]
    assert features.shape == (2 * 9 * subcarrier_count,)
    assert image.shape == (2, image_subcarrier_count, window_frames)
    assert intercarrier_features.shape == (2 * 27,)


def test_single_node_parity():
    """1-node stacked == plain iter_windows: features allclose, timestamps match, image allclose."""
    frames = list(_single_node_frames(duration_s=4.0))
    result, _ = _calibrate()
    subcarriers = list(result.subcarriers)
    image_subcarriers = result.image_subcarriers
    window_frames, hop_frames = 32, 16

    for frame in frames:
        frame.node_id = 0

    frames_by_node = group_frames_by_node(frames)
    calibration_by_node = {0: (subcarriers, image_subcarriers, None, None)}

    stacked = list(iter_stacked_windows(frames_by_node, calibration_by_node,
                                        window=window_frames, hop=hop_frames, intercarrier=False))
    # iter_windows reuses its internal buffers; copy each item while iterating.
    plain = [(timestamp_s, features.copy(), image.copy(), intercarrier_features)
             for timestamp_s, features, image, intercarrier_features in iter_windows(
        frames, subcarriers, None, window=window_frames, hop=hop_frames,
        image_subcarriers=(image_subcarriers if image_subcarriers != subcarriers else None),
    )]

    assert len(stacked) == len(plain) and len(stacked) > 0
    for (stacked_timestamp_s, stacked_features, stacked_image, _), (
            plain_timestamp_s, plain_features, plain_image, _) in zip(stacked, plain):
        assert stacked_timestamp_s == pytest.approx(plain_timestamp_s)
        assert np.allclose(stacked_features, plain_features)
        assert np.allclose(stacked_image[0], plain_image)


def test_timestamp_shift_raises():
    frames = _two_node_frames(duration_s=4.0, node_1_timestamp_offset_s=1.0)  # 1 s >> 0.05 s tolerance
    result, _ = _calibrate()
    subcarriers = list(result.subcarriers)
    image_subcarriers = result.image_subcarriers
    frames_by_node = group_frames_by_node(frames)
    calibration_by_node = {
        0: (subcarriers, image_subcarriers, None, None),
        1: (subcarriers, image_subcarriers, None, None),
    }
    with pytest.raises(ValueError, match="not node-synced"):
        list(iter_stacked_windows(frames_by_node, calibration_by_node,
                                  window=32, hop=16, node_tolerance=0.05))


def test_unequal_lengths_stops_at_shorter():
    long_frames = list(_single_node_frames(duration_s=6.0))
    short_frames = list(_single_node_frames(duration_s=2.0))
    result, _ = _calibrate()
    subcarriers = list(result.subcarriers)
    image_subcarriers = result.image_subcarriers

    frames_by_node = {0: long_frames, 1: short_frames}
    calibration_by_node = {
        0: (subcarriers, image_subcarriers, None, None),
        1: (subcarriers, image_subcarriers, None, None),
    }
    mixed_items = list(iter_stacked_windows(frames_by_node, calibration_by_node,
                                            window=32, hop=16))
    short_items = list(iter_windows(short_frames, subcarriers, None, window=32, hop=16))
    assert len(mixed_items) == len(short_items) and len(mixed_items) > 0


def test_mismatched_k_raises():
    """Nodes with different NBVI subcarrier counts raise ValueError before iteration."""
    result_a, _ = _calibrate(nbvi_max=6)
    result_b, _ = _calibrate(nbvi_max=2)

    if len(result_a.subcarriers) == len(result_b.subcarriers):
        pytest.skip("nbvi_max difference did not produce different K in this environment")

    frames = list(_single_node_frames())
    frames_by_node = {0: frames, 1: frames}
    calibration_by_node = {
        0: (list(result_a.subcarriers), result_a.image_subcarriers, None, None),
        1: (list(result_b.subcarriers), result_b.image_subcarriers, None, None),
    }
    with pytest.raises(ValueError, match="different K"):
        list(iter_stacked_windows(frames_by_node, calibration_by_node, window=32, hop=16))


def test_cnn_multichannel_fit_predict_roundtrip(tmp_path):
    """CNN takes (samples, nodes, image_subcarriers, window) images: save→load gives identical probas."""
    pytest.importorskip("torch")
    import joblib
    from wavetrace.config import ModelConfig
    from wavetrace.adapters.recognition.heads import build_weapon_head, load_weapon_head

    subcarrier_count, image_subcarrier_count, window_frames, node_count = 6, 20, 16, 2
    config = ModelConfig(stage="weapon", k=subcarrier_count, backend="cnn", window=window_frames, hop=8)
    rng = np.random.default_rng(1)
    sample_count = 40
    y = np.array([0] * 20 + [1] * 20, dtype=np.int64)

    X_two_nodes = rng.uniform(
        0, 1, size=(sample_count, node_count, image_subcarrier_count, window_frames)).astype(np.float32)
    head_two_nodes = build_weapon_head(config)
    head_two_nodes.fit(X_two_nodes, y, epochs=2)
    proba = head_two_nodes.predict_proba(X_two_nodes)
    assert proba.shape == (sample_count, 2)

    head_path = tmp_path / "wh_multi.joblib"
    head_two_nodes.save(head_path)
    reloaded_head = load_weapon_head(head_path)
    assert np.allclose(reloaded_head.predict_proba(X_two_nodes), proba, atol=1e-5)

    # A legacy single-channel blob stores image_shape as a 2-tuple (image_subcarriers, window).
    X_one_node = rng.uniform(
        0, 1, size=(sample_count, image_subcarrier_count, window_frames)).astype(np.float32)
    head_one_node = build_weapon_head(config)
    head_one_node.fit(X_one_node, y, epochs=2)
    head_one_node_path = tmp_path / "wh_1ch.joblib"
    head_one_node.save(head_one_node_path)
    proba_one_node = head_one_node.predict_proba(X_one_node)

    blob = joblib.load(head_one_node_path)
    blob["image_shape"] = (image_subcarrier_count, window_frames)  # 2-tuple: legacy file format
    legacy_path = tmp_path / "wh_legacy.joblib"
    joblib.dump(blob, legacy_path)
    legacy_head = load_weapon_head(legacy_path)
    assert legacy_head._backend._image_shape == (1, image_subcarrier_count, window_frames)
    assert np.allclose(legacy_head.predict_proba(X_one_node), proba_one_node, atol=1e-5)


def test_build_dataset_stacked_end_to_end(tmp_path):
    frames = _two_node_frames(duration_s=6.0)
    result, _ = _calibrate()
    calibrations = {0: (result, None), 1: (result, None)}
    window_frames, hop_frames = 32, 16
    subcarrier_count = len(result.subcarriers)
    image_subcarrier_count = len(result.image_subcarriers)
    node_count = 2

    labeler = ScriptedLabeler([(0.0, 6.0, True)])
    dataset = build_stacked_dataset(frames, calibrations, labeler, window=window_frames, hop=hop_frames,
                                    frame_average=2, subtract_baseline=False)

    assert dataset.X_features.shape[1] == node_count * 9 * subcarrier_count
    assert dataset.X_image.shape[1:] == (node_count, image_subcarrier_count, window_frames)
    assert dataset.y.shape[0] == dataset.X_features.shape[0]
    assert dataset.meta["num_nodes"] == node_count
    assert dataset.meta["node_ids"] == [0, 1]
    assert dataset.meta["K"] == subcarrier_count
    assert dataset.meta["K_img"] == image_subcarrier_count
    assert dataset.meta["frame_average"] == 2
    assert dataset.meta["subtract_baseline"] is False

    out = save_dataset(dataset, tmp_path / "stacked")
    reloaded_dataset = load_dataset(out)
    assert np.array_equal(reloaded_dataset.X_features, dataset.X_features)
    assert np.array_equal(reloaded_dataset.X_image, dataset.X_image)
    assert np.array_equal(reloaded_dataset.y, dataset.y)
