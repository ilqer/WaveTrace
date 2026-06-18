"""Phase 5 — ground-truth pipeline tests (CameraLabeler, Align, DatasetBuilder) on the synthetic
paired recording. Validates the alignment/dataset PLUMBING + the sync-error measurement (no hardware,
no real CSI signatures)."""

import json

import numpy as np
import pytest

from fixtures.SyntheticCsi import generateStream
from fixtures.SyntheticRecording import generatePairedRecording
from wavetrace import Label
from wavetrace.Calibration import Calibration
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
    """Quiet-baseline Calibration -> (result, locked GainLock)."""
    baseline, _ = generateStream(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, numFrames=60,
        perturbationHz=0.0, perturbationDepth=0.0, cfoHz=0.0, noiseStd=0.005, seed=7,
    )
    cal = Calibration(baseline_packets=50)
    for fr in baseline:
        cal.observe(fr)
    result = cal.finalize()
    return result, cal.gain_lock


# ----- 5a CameraLabeler -------------------------------------------------------------------------

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
    assert sl(1.5).class_id == 1          # inside present span -> weapon
    assert sl(0.5).class_id == 0          # outside -> no_weapon
    assert sl(3.5).class_id == 0          # explicit absent span

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"spans": [{"start": 1.0, "end": 2.0, "present": True}]}))
    sl2 = ScriptedLabeler.from_manifest(path)
    assert sl2(1.5).class_id == 1 and sl2(2.5).class_id == 0


def test_location_chip_labeler_stores_position():
    chip = LocationChipLabeler([
        (0.0, False, None),
        (1.5, True, [0.45, 0.55, 0.10, 0.20]),
    ])
    present = chip(1.4)                    # nearest sample is the present one at 1.5
    assert present.class_id == 1
    assert present.bbox == pytest.approx([0.45, 0.55, 0.10, 0.20])  # weapon location preserved
    absent = chip(0.0)
    assert absent.class_id == 0 and absent.bbox is None


def test_thermal_labeler_is_a_seam():
    with pytest.raises(NotImplementedError):
        ThermalLabeler().label({"raw": {}}, 0.0)


# ----- 5b Align (sync-error measurement) --------------------------------------------------------

def _window_timestamps(frames, window=32, hop=16):
    """CSI window-END timestamps emulating the front-end emit cadence."""
    return [frames[i].timestamp for i in range(window - 1, len(frames), hop)]


def test_align_bounds_sync_error_and_pairs_correct():
    # Shared host clock (Q6 default): no offset, only jitter + camera quantization -> Δt bounded.
    cam_fps = 30.0
    frames, obs, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=3.0,
        cameraFps=cam_fps, clockOffsetS=0.0, jitterStdS=0.002,
        presenceSpans=[(1.0, 2.0)], seed=1,
    )
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    win_ts = _window_timestamps(frames)
    res = align(win_ts, labels, tolerance=0.1)

    assert res.stats["dropped"] == 0
    assert abs(res.stats["mean_dt"]) < 0.01
    # bounded by half a camera period + a few jitter sigma (the measured, bounded sync error)
    assert res.stats["max_abs_dt"] < (0.5 / cam_fps + 5 * 0.002)
    # pairs correct: interior windows inside the presence span are labeled present
    for wi, lab in res.matched:
        wt = win_ts[wi]
        if 1.2 <= wt <= 1.8:
            assert lab.class_id == 1
        elif wt <= 0.8 or wt >= 2.2:
            assert lab.class_id == 0


def test_align_drops_windows_with_no_label_in_tolerance():
    # Drop a span of camera observations -> CSI windows there have no label within tolerance.
    frames, obs, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=3.0,
        cameraFps=30.0, clockOffsetS=0.0, jitterStdS=0.0, seed=2,
    )
    obs = [o for o in obs if not (1.0 <= o["true_t"] < 2.0)]  # 1 s camera gap
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    win_ts = _window_timestamps(frames)
    res = align(win_ts, labels, tolerance=0.05)
    assert res.stats["dropped"] > 0
    assert res.stats["matched"] > 0
    assert all(abs(dt) <= 0.05 for dt in res.dts)            # survivors are within tolerance
    dropped_ts = [win_ts[i] for i in res.dropped]
    assert all(1.0 - 0.05 <= t <= 2.0 + 0.05 for t in dropped_ts)  # only the gap windows dropped


def test_estimate_clock_offset_recovers_injection():
    # A constant offset is invisible to nearest-match Δt; recover it by content cross-correlation.
    offset = 0.05
    frames, obs, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=3.0,
        cameraFps=30.0, clockOffsetS=offset, jitterStdS=0.001,
        presenceSpans=[(1.0, 2.0)], seed=5,
    )
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    win_ts = _window_timestamps(frames)
    # fine staged-truth grid (CSI clock) -> offset resolvable to ~the camera period
    truth_t = np.arange(0.0, 3.0, 1.0 / FS)
    truth_c = [1 if 1.0 <= t < 2.0 else 0 for t in truth_t]

    # nearest-match Δt does NOT reveal the offset (stays within half a camera period)
    assert abs(align(win_ts, labels, tolerance=0.1).stats["mean_dt"]) < 0.5 / 30.0
    # content correlation does
    est, agree = estimate_clock_offset(truth_t, truth_c, labels, max_lag=0.2, step=0.005)
    assert est == pytest.approx(offset, abs=0.02)
    assert agree > 0.95


# ----- 5c DatasetBuilder ------------------------------------------------------------------------

def test_dataset_builder_camera_shapes_and_roundtrip(tmp_path):
    result, gain = _calibrate()
    K = len(result.subcarriers)
    frames, obs, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=3.0,
        cameraFps=30.0, clockOffsetS=0.0, jitterStdS=0.001,
        presenceSpans=[(1.0, 2.0)], seed=3,
    )
    labels = ReplayLabeler(presence_label_fn).label_stream(obs)
    ds = build_dataset(frames, result, gain, labels, window=32, hop=16, tolerance=0.1,
                       class_names={0: "absent", 1: "present"})

    n = ds.y.shape[0]
    assert n > 0
    K_img = ds.meta["K_img"]  # T1/P10: image uses all valid subcarriers (>= K NBVI)
    assert ds.X_features.shape == (n, 9 * K)
    assert ds.X_image.shape == (n, K_img, 32)
    assert ds.t.shape == (n,)
    assert ds.meta["K"] == K and ds.meta["fs"] == pytest.approx(FS, rel=0.05)
    # stored sync error = the bounded matched-Δt residual (shared clock -> small)
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
    # the camera mask (heatmap target) must survive save->load, else the heatmap head loses its label.
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
    # gain_lock=None (material/weapon path): no per-frame rescale; meta records the raw basis.
    result, _ = _calibrate()
    frames, _, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=3.0,
        cameraFps=30.0, weaponSpans=[(1.0, 2.0)], seed=4,
    )
    scripted = ScriptedLabeler([(1.0, 2.0, True)], label_fn=weapon_label_fn)
    ds = build_dataset(frames, result, None, scripted, window=32, hop=16)

    assert ds.meta["gain_locked"] is False
    assert ds.y.shape[0] > 0


def test_dataset_builder_scripted_callable_no_drop():
    result, gain = _calibrate()
    frames, _, _ = generatePairedRecording(
        numAntennas=NUM_ANT, numSubcarriers=NUM_SUB, sampleRateHz=FS, durationS=3.0,
        cameraFps=30.0, weaponSpans=[(1.0, 2.0)], seed=4,
    )
    scripted = ScriptedLabeler([(1.0, 2.0, True)], label_fn=weapon_label_fn)
    ds = build_dataset(frames, result, gain, scripted, window=32, hop=16)

    assert ds.meta["n_dropped"] == 0                     # time-style: same clock, nothing dropped
    assert set(ds.y.tolist()) == {0, 1}                  # both classes present across the recording
    for cls, t in zip(ds.y.tolist(), ds.t.tolist()):
        if 1.1 <= t <= 1.9:
            assert cls == 1
