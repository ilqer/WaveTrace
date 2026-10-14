import json

import numpy as np
import pytest

from wavetrace.synthetic import generate_paired_recording
from wavetrace import Label
from wavetrace.groundtruth import (
    Dataset,
    LocationChipLabeler,
    ReplayLabeler,
    ScriptedLabeler,
    ThermalLabeler,
    align,
    build_dataset,
    estimate_clock_offset,
    load_dataset,
    presence_label_function,
    save_dataset,
    weapon_label_function,
)
from fixtures.synthetic_csi import calibrate_quiet_baseline

NUM_ANTENNAS = 2
NUM_SUBCARRIERS = 32
SAMPLE_RATE_HZ = 100.0


def _calibrate():
    return calibrate_quiet_baseline(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        num_frames=60, seed=7, baseline_packets=50)


def test_replay_labeler_roundtrips_presence():
    observations = [
        {"t": 0.10, "raw": {"present": True, "bbox": [0.4, 0.3, 0.2, 0.5],
                            "keypoints": [0.5, 0.2], "weapon": False}},
        {"t": 0.20, "raw": {"present": False, "bbox": None, "keypoints": [], "weapon": False}},
    ]
    labels = ReplayLabeler(presence_label_function).label_stream(observations)
    assert [label.class_id for label in labels] == [1, 0]
    assert [label.name for label in labels] == ["present", "absent"]
    assert labels[0].bbox == pytest.approx([0.4, 0.3, 0.2, 0.5])
    assert labels[0].timestamp == pytest.approx(0.10)
    assert labels[1].bbox is None


def test_replay_labeler_weapon_policy():
    observations = [{"t": 0.0, "raw": {"present": True, "weapon": True, "position": [0.45, 0.55, 0.1, 0.2]}}]
    label = ReplayLabeler(weapon_label_function).label(observations[0], observations[0]["t"])
    assert label.class_id == 1 and label.name == "weapon"


def test_scripted_labeler_spans_and_manifest(tmp_path):
    labeler = ScriptedLabeler([(1.0, 2.0, True), (3.0, 4.0, False)])
    assert labeler(1.5).class_id == 1
    assert labeler(0.5).class_id == 0
    assert labeler(3.5).class_id == 0  # Explicit absent span.

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"spans": [{"start": 1.0, "end": 2.0, "present": True}]}))
    manifest_labeler = ScriptedLabeler.from_manifest(path)
    assert manifest_labeler(1.5).class_id == 1 and manifest_labeler(2.5).class_id == 0


def test_location_chip_labeler_stores_position():
    chip = LocationChipLabeler([
        (0.0, False, None),
        (1.5, True, [0.45, 0.55, 0.10, 0.20]),
    ])
    present = chip(1.4)  # Nearest sample.
    assert present.class_id == 1
    assert present.bbox == pytest.approx([0.45, 0.55, 0.10, 0.20])  # Weapon location preserved.
    absent = chip(0.0)
    assert absent.class_id == 0 and absent.bbox is None


def test_thermal_labeler_is_a_seam():
    with pytest.raises(NotImplementedError):
        ThermalLabeler().label({"raw": {}}, 0.0)


def _window_timestamps(frames, window=32, hop=16):
    """CSI window-END timestamps."""
    return [frames[i].timestamp for i in range(window - 1, len(frames), hop)]


def test_align_bounds_sync_error_and_pairs_correct():
    # Shared host clock: Δt is bounded.
    camera_frame_rate_hz = 30.0
    frames, observations, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS,
        sample_rate_hz=SAMPLE_RATE_HZ, duration_s=3.0,
        camera_frame_rate_hz=camera_frame_rate_hz, clock_offset_s=0.0, jitter_std_s=0.002,
        presence_spans=[(1.0, 2.0)], seed=1,
    )
    labels = ReplayLabeler(presence_label_function).label_stream(observations)
    window_timestamps_s = _window_timestamps(frames)
    alignment = align(window_timestamps_s, labels, tolerance=0.1)

    assert alignment.stats["dropped"] == 0
    assert abs(alignment.stats["mean_dt"]) < 0.01
    # Worst case: half a camera frame plus five sigma of timestamp jitter.
    assert alignment.stats["max_abs_dt"] < (0.5 / camera_frame_rate_hz + 5 * 0.002)
    # Only windows well inside or well outside the span carry a certain label.
    for window_index, label in alignment.matched:
        window_timestamp_s = window_timestamps_s[window_index]
        if 1.2 <= window_timestamp_s <= 1.8:
            assert label.class_id == 1
        elif window_timestamp_s <= 0.8 or window_timestamp_s >= 2.2:
            assert label.class_id == 0


def test_align_drops_windows_with_no_label_in_tolerance():
    # Drop camera span -> no label within tolerance.
    frames, observations, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS,
        sample_rate_hz=SAMPLE_RATE_HZ, duration_s=3.0,
        camera_frame_rate_hz=30.0, clock_offset_s=0.0, jitter_std_s=0.0, seed=2,
    )
    observations = [observation for observation in observations if not (1.0 <= observation["true_t"] < 2.0)]
    labels = ReplayLabeler(presence_label_function).label_stream(observations)
    window_timestamps_s = _window_timestamps(frames)
    alignment = align(window_timestamps_s, labels, tolerance=0.05)
    assert alignment.stats["dropped"] > 0
    assert alignment.stats["matched"] > 0
    assert all(abs(time_delta_s) <= 0.05 for time_delta_s in alignment.time_deltas_s)
    dropped_timestamps_s = [window_timestamps_s[i] for i in alignment.dropped]
    assert all(1.0 - 0.05 <= timestamp_s <= 2.0 + 0.05 for timestamp_s in dropped_timestamps_s)


def test_estimate_clock_offset_recovers_injection():
    offset = 0.05
    frames, observations, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS,
        sample_rate_hz=SAMPLE_RATE_HZ, duration_s=3.0,
        camera_frame_rate_hz=30.0, clock_offset_s=offset, jitter_std_s=0.001,
        presence_spans=[(1.0, 2.0)], seed=5,
    )
    labels = ReplayLabeler(presence_label_function).label_stream(observations)
    window_timestamps_s = _window_timestamps(frames)
    truth_timestamps_s = np.arange(0.0, 3.0, 1.0 / SAMPLE_RATE_HZ)
    truth_class_ids = [1 if 1.0 <= timestamp_s < 2.0 else 0 for timestamp_s in truth_timestamps_s]

    # Nearest-match Δt does not reveal offset.
    assert abs(align(window_timestamps_s, labels, tolerance=0.1).stats["mean_dt"]) < 0.5 / 30.0
    # Content correlation reveals offset.
    estimated_offset_s, agreement = estimate_clock_offset(truth_timestamps_s, truth_class_ids, labels,
                                                          max_lag_s=0.2, step_s=0.005)
    assert estimated_offset_s == pytest.approx(offset, abs=0.02)
    assert agreement > 0.95


def test_dataset_builder_camera_shapes_and_roundtrip(tmp_path):
    result, gain = _calibrate()
    subcarrier_count = len(result.subcarriers)
    frames, observations, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS,
        sample_rate_hz=SAMPLE_RATE_HZ, duration_s=3.0,
        camera_frame_rate_hz=30.0, clock_offset_s=0.0, jitter_std_s=0.001,
        presence_spans=[(1.0, 2.0)], seed=3,
    )
    labels = ReplayLabeler(presence_label_function).label_stream(observations)
    dataset = build_dataset(frames, result, gain, labels, window=32, hop=16, tolerance=0.1,
                            class_names={0: "absent", 1: "present"})

    window_count = dataset.y.shape[0]
    assert window_count > 0
    image_subcarrier_count = dataset.meta["K_img"]  # Image uses all valid subcarriers.
    assert dataset.X_features.shape == (window_count, 9 * subcarrier_count)
    assert dataset.X_image.shape == (window_count, image_subcarrier_count, 32)
    assert dataset.t.shape == (window_count,)
    assert (dataset.meta["K"] == subcarrier_count
            and dataset.meta["fs"] == pytest.approx(SAMPLE_RATE_HZ, rel=0.05))
    # Half a camera frame is the floor on sync error.
    assert dataset.meta["sync_error"]["max_abs_dt"] < 0.5 / 30.0 + 0.01

    out = save_dataset(dataset, tmp_path / "ds")
    assert (out / "manifest.jsonl").exists() and (out / "meta.json").exists()
    reloaded = load_dataset(out)
    assert np.array_equal(reloaded.X_features, dataset.X_features)
    assert np.array_equal(reloaded.X_image, dataset.X_image)
    assert np.array_equal(reloaded.y, dataset.y)
    assert reloaded.meta["sync_error"] == dataset.meta["sync_error"]
    assert [label.class_id for label in reloaded.labels] == list(dataset.y)


def test_dataset_roundtrips_heatmap_mask(tmp_path):
    grid = 4
    labels = []
    for i in range(3):
        label = Label()
        label.class_id = 1
        label.name = "present"
        label.timestamp = float(i)
        label.mask = [float((i + j) % 2) for j in range(grid * grid)]
        label.mask_grid = grid
        labels.append(label)
    dataset = Dataset(
        X_features=np.zeros((3, 9), np.float32),
        X_image=np.zeros((3, 4, 8), np.float32),
        y=np.array([1, 1, 1], np.int64),
        t=np.array([0.0, 1.0, 2.0]),
        labels=labels,
        session_ids=np.full(3, "s0", dtype=object),
        subject_ids=np.full(3, "p0", dtype=object),
    )
    reloaded = load_dataset(save_dataset(dataset, tmp_path / "mask_ds"))
    assert [item.mask_grid for item in reloaded.labels] == [grid] * 3
    assert [list(item.mask) for item in reloaded.labels] == [list(item.mask) for item in labels]


def test_dataset_builder_skips_gain_lock_when_none():
    result, _ = _calibrate()
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS,
        sample_rate_hz=SAMPLE_RATE_HZ, duration_s=3.0,
        camera_frame_rate_hz=30.0, weapon_spans=[(1.0, 2.0)], seed=4,
    )
    scripted = ScriptedLabeler([(1.0, 2.0, True)], label_function=weapon_label_function)
    dataset = build_dataset(frames, result, None, scripted, window=32, hop=16)

    assert dataset.meta["gain_locked"] is False
    assert dataset.y.shape[0] > 0


def test_dataset_builder_scripted_callable_no_drop():
    result, gain = _calibrate()
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS,
        sample_rate_hz=SAMPLE_RATE_HZ, duration_s=3.0,
        camera_frame_rate_hz=30.0, weapon_spans=[(1.0, 2.0)], seed=4,
    )
    scripted = ScriptedLabeler([(1.0, 2.0, True)], label_function=weapon_label_function)
    dataset = build_dataset(frames, result, gain, scripted, window=32, hop=16)

    assert dataset.meta["n_dropped"] == 0
    assert set(dataset.y.tolist()) == {0, 1}
    for class_id, timestamp_s in zip(dataset.y.tolist(), dataset.t.tolist()):
        if 1.1 <= timestamp_s <= 1.9:
            assert class_id == 1
