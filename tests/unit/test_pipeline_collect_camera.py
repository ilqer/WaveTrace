import builtins
import json
import sys
import threading

import pytest

from fixtures.synthetic_csi import generate_quiet_baseline
from scripts import collect_camera as collect_camera_script
from wavetrace import Label, pipeline
from wavetrace.adapters.store.model_runs import HEATMAP_FILE_NAME, resolve_run_dir
from wavetrace.pipeline import calibrate_source, collect_camera
from wavetrace.synthetic import generate_paired_recording

NUM_SUBCARRIERS = 32
SAMPLE_RATE_HZ = 100.0
CAPTURE_S = 6.0
LABEL_RATE_HZ = 15.0
IN_VIEW_SPAN_S = (2.0, 4.0)
RX_NODE_IDS = (1, 2)
TX_SHORTS = ("4f:9c", "64:b8")


class _FrameListSource:
    def __init__(self, frames):
        self._frames = frames

    def frames(self):
        return iter(self._frames)


def _calibrate(root):
    for rx_node_id in RX_NODE_IDS:
        baseline = generate_quiet_baseline(num_antennas=1, num_subcarriers=NUM_SUBCARRIERS,
                                           sample_rate_hz=SAMPLE_RATE_HZ, num_frames=100, seed=rx_node_id)
        calibrate_source(_FrameListSource(baseline), root / "cal" / f"node{rx_node_id}", baseline_packets=100)
    return {rx_node_id: str(root / "cal" / f"node{rx_node_id}") for rx_node_id in RX_NODE_IDS}


def _link_frames(rx_node_id, tx_index):
    frames, _, _ = generate_paired_recording(
        num_antennas=1, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ, duration_s=CAPTURE_S,
        camera_frame_rate_hz=10.0, presence_spans=[IN_VIEW_SPAN_S], presence_turbulence_std=0.15,
        weapon_spans=[IN_VIEW_SPAN_S], weapon_signature_depth=0.4, seed=10 * rx_node_id + tx_index)
    for frame in frames:
        frame.node_id = rx_node_id
    return frames


class _ReplayedCapture:
    def __init__(self, port, node_ids):
        pass

    def capture_for(self, duration_s, *, stop):
        return {(tx_short, rx_node_id): _link_frames(rx_node_id, tx_index)
                for rx_node_id in RX_NODE_IDS for tx_index, tx_short in enumerate(TX_SHORTS)}


class _ScriptedCamera:
    """The subject, holding the object up, is in view through IN_VIEW_SPAN_S."""

    def __init__(self, **options):
        pass

    def record_labels(self, duration_s, *, stop, on_label):
        labels = []
        for i in range(int(CAPTURE_S * LABEL_RATE_HZ)):
            label = Label()
            label.timestamp = i / LABEL_RATE_HZ
            label.class_id = int(IN_VIEW_SPAN_S[0] <= label.timestamp < IN_VIEW_SPAN_S[1])
            label.name = "in view" if label.class_id else "out of view"
            if label.class_id:
                label.mask = [1.0] * 8 + [0.0] * 248
                label.mask_grid = 16
            labels.append(label)
        return labels


@pytest.fixture
def replayed_mesh(monkeypatch):
    monkeypatch.setattr(pipeline, "UdpTimedLinkCapture", _ReplayedCapture)


def _collect(root, calibration_dir_by_node_id, *, stage="presence", model_root=None):
    collect_camera(_ScriptedCamera(), stage=stage, port=5599, root=str(root),
                   calibration_dir_by_node_id=calibration_dir_by_node_id, subject_id="p0", duration_s=CAPTURE_S,
                   stop=threading.Event(), model_root=model_root, log=lambda message: None,
                   on_capture_end=lambda: None)


def test_each_run_takes_the_next_free_session_id_so_runs_accumulate(tmp_path, replayed_mesh):
    calibration_dir_by_node_id = _calibrate(tmp_path)
    for _ in range(2):
        _collect(tmp_path, calibration_dir_by_node_id)
    assert sorted(path.name for path in (tmp_path / "cam_rec").iterdir()) == ["p0_cam_s0", "p0_cam_s1"]
    link_dir = tmp_path / "cam_ds" / "presence" / "node1" / "link_4f9c"
    assert sorted(path.name for path in link_dir.iterdir()) == ["p0_cam_s0", "p0_cam_s1"]
    assert sorted(path.name for path in (tmp_path / "cam_ds" / "heatmap").iterdir()) == ["p0_cam_s0", "p0_cam_s1"]


def test_a_calibrated_node_that_sent_nothing_is_left_out_of_the_heatmap(tmp_path, replayed_mesh):
    calibration_dir_by_node_id = _calibrate(tmp_path)
    baseline = generate_quiet_baseline(num_antennas=1, num_subcarriers=NUM_SUBCARRIERS,
                                       sample_rate_hz=SAMPLE_RATE_HZ, num_frames=100, seed=3)
    calibrate_source(_FrameListSource(baseline), tmp_path / "cal" / "node3", baseline_packets=100)
    calibration_dir_by_node_id[3] = str(tmp_path / "cal" / "node3")
    _collect(tmp_path, calibration_dir_by_node_id)
    heatmap_meta = json.loads((tmp_path / "cam_ds" / "heatmap" / "p0_cam_s0" / "meta.json").read_text())
    assert heatmap_meta["node_ids"] == [1, 2]


def test_presence_writes_one_dataset_per_link(tmp_path, replayed_mesh):
    _collect(tmp_path, _calibrate(tmp_path))
    for rx_node_id in RX_NODE_IDS:
        node_dir = tmp_path / "cam_ds" / "presence" / f"node{rx_node_id}"
        assert sorted(path.name for path in node_dir.iterdir()) == ["link_4f9c", "link_64b8"]


def _run_script(monkeypatch, root, *extra_argv):
    monkeypatch.setattr(collect_camera_script, "WebcamLabelCapture", _ScriptedCamera)
    monkeypatch.setattr(builtins, "input", lambda *prompt: "")
    monkeypatch.setattr(sys, "argv", ["collect_camera", "--root", str(root), "--duration", f"{CAPTURE_S:g}",
                                      "--train", *extra_argv])
    collect_camera_script.main()


def test_presence_training_writes_only_the_camera_model_root(tmp_path, replayed_mesh, monkeypatch, capsys):
    _calibrate(tmp_path)
    _run_script(monkeypatch, tmp_path)
    assert capsys.readouterr().out.count("\a") == 1
    for rx_node_id in RX_NODE_IDS:
        assert resolve_run_dir(tmp_path / "model_camera" / f"node{rx_node_id}") is not None
    assert resolve_run_dir(tmp_path / "model_camera", HEATMAP_FILE_NAME) is not None
    assert not (tmp_path / "model").exists() and not (tmp_path / "model_weapon").exists()


def test_weapon_training_joins_the_weapon_pool_and_writes_only_the_weapon_model_root(tmp_path, replayed_mesh,
                                                                                   monkeypatch, capsys):
    _calibrate(tmp_path)
    _run_script(monkeypatch, tmp_path, "--stage", "weapon")
    for rx_node_id in RX_NODE_IDS:
        pool_dir = tmp_path / "weapon_ds" / f"node{rx_node_id}"
        assert sorted(path.name for path in pool_dir.iterdir()) == ["cam_cam_s0_link4f9c", "cam_cam_s0_link64b8"]
        assert resolve_run_dir(tmp_path / "model_weapon" / f"node{rx_node_id}") is not None
    assert not (tmp_path / "model").exists() and not (tmp_path / "model_camera").exists()
    assert not (tmp_path / "cam_ds" / "weapon").exists()
