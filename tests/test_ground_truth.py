import json

import numpy as np
import pytest

from wavetrace.synthetic import generate_stream
from wavetrace.synthetic import generate_paired_recording
from wavetrace import Label
from wavetrace.calibration import Calibration
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
    presence_label_fn,
    save_dataset,
    weapon_label_fn,
)

NUM_ANT = 2
NUM_SUB = 32
FS = 100.0


def _calibrate():
    """Quiet-baseline calibration."""
    baseline, _ = generate_stream(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, num_frames=60,
        perturbation_hz=0.0, perturbation_depth=0.0, cfo_hz=0.0, noise_std=0.005, seed=7,
    )
    cal = Calibration(baseline_packets=50)
    for fr in baseline:
        cal.observe(fr)
    result = cal.finalize()
    return result, cal.gain_lock


def test_replay_labeler_roundtrips_presence():
    obs = [
        {"t": 0.10, "raw": {"present": True, "bbox": [0.4, 0.3, 0.2, 0.5],
                            "keypoints": [0.5, 0.2], "weapon": False}},
        {"t": 0.20, "raw": {"present": False, "bbox": None, "keypoints": [], "weapon": False}},
    ]
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    assert [l.class_id for l in labels] == [1, 0]
    assert [l.name for l in labels] == ["present", "absent"]
    assert labels[0].bbox == pytest.approx([0.4, 0.3, 0.2, 0.5])
    assert labels[0].timestamp == pytest.approx(0.10)
    assert labels[1].bbox is None


def test_replay_labeler_weapon_policy():
    obs = [{"t": 0.0, "raw": {"present": True, "weapon": True, "position": [0.45, 0.55, 0.1, 0.2]}}]
    lab = ReplayLabeler(weapon_label_fn).label(obs[0], obs[0]["t"])
    assert lab.class_id == 1 and lab.name == "weapon"


def test_scripted_labeler_spans_and_manifest(tmp_path):
    sl = ScriptedLabeler([(1.0, 2.0, True), (3.0, 4.0, False)])
    assert sl(1.5).class_id == 1          # Inside present span.
    assert sl(0.5).class_id == 0          # Outside span.
    assert sl(3.5).class_id == 0          # Explicit absent span.

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"spans": [{"start": 1.0, "end": 2.0, "present": True}]}))
    sl2 = ScriptedLabeler.from_manifest(path)
    assert sl2(1.5).class_id == 1 and sl2(2.5).class_id == 0


def test_location_chip_labeler_stores_position():
    chip = LocationChipLabeler([
        (0.0, False, None),
        (1.5, True, [0.45, 0.55, 0.10, 0.20]),
    ])
    present = chip(1.4)                    # Nearest sample.
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
    camera_fps = 30.0
    frames, obs, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=3.0,
        camera_fps=camera_fps, clock_offset_s=0.0, jitter_std_s=0.002,
        presence_spans=[(1.0, 2.0)], seed=1,
    )
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    window_timestamps_s = _window_timestamps(frames)
    res = align(window_timestamps_s, labels, tolerance=0.1)

    assert res.stats["dropped"] == 0
    assert abs(res.stats["mean_dt"]) < 0.01
    # Worst case: half a camera frame plus five sigma of timestamp jitter.
    assert res.stats["max_abs_dt"] < (0.5 / camera_fps + 5 * 0.002)
    # Only windows well inside or well outside the span carry a certain label.
    for wi, lab in res.matched:
        wt = window_timestamps_s[wi]
        if 1.2 <= wt <= 1.8:
            assert lab.class_id == 1
        elif wt <= 0.8 or wt >= 2.2:
            assert lab.class_id == 0


def test_align_drops_windows_with_no_label_in_tolerance():
    # Drop camera span -> no label within tolerance.
    frames, obs, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=3.0,
        camera_fps=30.0, clock_offset_s=0.0, jitter_std_s=0.0, seed=2,
    )
    obs = [o for o in obs if not (1.0 <= o["true_t"] < 2.0)]
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    window_timestamps_s = _window_timestamps(frames)
    res = align(window_timestamps_s, labels, tolerance=0.05)
    assert res.stats["dropped"] > 0
    assert res.stats["matched"] > 0
    assert all(abs(dt) <= 0.05 for dt in res.dts)
    dropped_timestamps_s = [window_timestamps_s[i] for i in res.dropped]
    assert all(1.0 - 0.05 <= t <= 2.0 + 0.05 for t in dropped_timestamps_s)  # Gap windows dropped.


def test_estimate_clock_offset_recovers_injection():
    offset = 0.05
    frames, obs, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=3.0,
        camera_fps=30.0, clock_offset_s=offset, jitter_std_s=0.001,
        presence_spans=[(1.0, 2.0)], seed=5,
    )
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    window_timestamps_s = _window_timestamps(frames)
    truth_timestamps_s = np.arange(0.0, 3.0, 1.0 / FS)
    truth_class_ids = [1 if 1.0 <= t < 2.0 else 0 for t in truth_timestamps_s]

    # Nearest-match Δt does not reveal offset.
    assert abs(align(window_timestamps_s, labels, tolerance=0.1).stats["mean_dt"]) < 0.5 / 30.0
    # Content correlation reveals offset.
    est, agree = estimate_clock_offset(truth_timestamps_s, truth_class_ids, labels, max_lag=0.2, step=0.005)
    assert est == pytest.approx(offset, abs=0.02)
    assert agree > 0.95


def test_dataset_builder_camera_shapes_and_roundtrip(tmp_path):
    result, gain = _calibrate()
    K = len(result.subcarriers)
    frames, obs, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=3.0,
        camera_fps=30.0, clock_offset_s=0.0, jitter_std_s=0.001,
        presence_spans=[(1.0, 2.0)], seed=3,
    )
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    ds = build_dataset(frames, result, gain, labels, window=32, hop=16, tolerance=0.1,
                       class_names={0: "absent", 1: "present"})

    n = ds.y.shape[0]
    assert n > 0
    K_img = ds.meta["K_img"]  # Image uses all valid subcarriers.
    assert ds.X_features.shape == (n, 9 * K)
    assert ds.X_image.shape == (n, K_img, 32)
    assert ds.t.shape == (n,)
    assert ds.meta["K"] == K and ds.meta["fs"] == pytest.approx(FS, rel=0.05)
    # Half a camera frame is the floor on sync error.
    assert ds.meta["sync_error"]["max_abs_dt"] < 0.5 / 30.0 + 0.01

    out = save_dataset(ds, tmp_path / "ds")
    assert (out / "manifest.jsonl").exists() and (out / "meta.json").exists()
    reloaded = load_dataset(out)
    assert np.array_equal(reloaded.X_features, ds.X_features)
    assert np.array_equal(reloaded.X_image, ds.X_image)
    assert np.array_equal(reloaded.y, ds.y)
    assert reloaded.meta["sync_error"] == ds.meta["sync_error"]
    assert [l.class_id for l in reloaded.labels] == list(ds.y)


def test_dataset_roundtrips_heatmap_mask(tmp_path):
    grid = 4
    labels = []
    for i in range(3):
        lab = Label()
        lab.class_id = 1
        lab.name = "present"
        lab.timestamp = float(i)
        lab.mask = [float((i + j) % 2) for j in range(grid * grid)]
        lab.mask_grid = grid
        labels.append(lab)
    ds = Dataset(
        X_features=np.zeros((3, 9), np.float32),
        X_image=np.zeros((3, 4, 8), np.float32),
        y=np.array([1, 1, 1], np.int64),
        t=np.array([0.0, 1.0, 2.0]),
        labels=labels,
        session_ids=np.full(3, "s0", dtype=object),
        subject_ids=np.full(3, "p0", dtype=object),
    )
    reloaded = load_dataset(save_dataset(ds, tmp_path / "mask_ds"))
    assert [l.mask_grid for l in reloaded.labels] == [grid] * 3
    assert [list(l.mask) for l in reloaded.labels] == [list(l.mask) for l in labels]


def test_dataset_builder_skips_gain_lock_when_none():
    # gain_lock=None: no per-frame rescale.
    result, _ = _calibrate()
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=3.0,
        camera_fps=30.0, weapon_spans=[(1.0, 2.0)], seed=4,
    )
    scripted = ScriptedLabeler([(1.0, 2.0, True)], label_fn=weapon_label_fn)
    ds = build_dataset(frames, result, None, scripted, window=32, hop=16)

    assert ds.meta["gain_locked"] is False
    assert ds.y.shape[0] > 0


def test_dataset_builder_scripted_callable_no_drop():
    result, gain = _calibrate()
    frames, _, _ = generate_paired_recording(
        num_antennas=NUM_ANT, num_subcarriers=NUM_SUB, sample_rate_hz=FS, duration_s=3.0,
        camera_fps=30.0, weapon_spans=[(1.0, 2.0)], seed=4,
    )
    scripted = ScriptedLabeler([(1.0, 2.0, True)], label_fn=weapon_label_fn)
    ds = build_dataset(frames, result, gain, scripted, window=32, hop=16)

    assert ds.meta["n_dropped"] == 0
    assert set(ds.y.tolist()) == {0, 1}
    for cls, t in zip(ds.y.tolist(), ds.t.tolist()):
        if 1.1 <= t <= 1.9:
            assert cls == 1
