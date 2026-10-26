from types import SimpleNamespace

import pytest

from wavetrace.domain.model_config import ModelConfig
from wavetrace.recognition import camera_collect_modes, collect_modes
from wavetrace.recognition.camera_collect_modes import (
    CameraPresenceMode, CameraPresenceTrainingPolicy, default_camera_model_root, find_heatmap_capture_root,
    next_camera_session_id, train_heatmap_head,
)


@pytest.mark.parametrize("taken_dir", [
    "cam_rec/p0_cam_s0",
    "cam_ds/heatmap/p0_cam_s0",
    "cam_ds/presence/node2/link_64b8/p0_cam_s0",
    "weapon_ds/node1/p0_cam_s0_link4f9c",
])
def test_a_session_with_any_data_left_on_disk_is_taken(tmp_path, taken_dir):
    (tmp_path / taken_dir).mkdir(parents=True)
    assert next_camera_session_id(str(tmp_path), "p0") == "p0_cam_s1"


def test_the_lowest_free_index_is_reused_and_other_subjects_do_not_count(tmp_path):
    for taken_dir in ("cam_rec/p0_cam_s1", "weapon_ds/node1/p1_cam_s0_link4f9c",
                      "weapon_ds/node1/p0_x_s0_link4f9c"):
        (tmp_path / taken_dir).mkdir(parents=True)
    assert next_camera_session_id(str(tmp_path), "p0") == "p0_cam_s0"


def test_the_camera_presence_pool_finds_every_dataset_camera_sessions_write(tmp_path):
    root = str(tmp_path)
    for session_id in ("cam_cam_s0", "p0_cam_s1"):
        mode = CameraPresenceMode(root=root, session_id=session_id, subject_id="cam", model_root=None)
        for link_key in (("4f:9c", 1), ("64:b8", 2)):
            (tmp_path / mode.describe_link_dataset(link_key, [], []).dataset_dir).mkdir(parents=True)
    (tmp_path / "ds_0" / "node1" / "link_4f9c").mkdir(parents=True)
    policy = CameraPresenceTrainingPolicy(root=root, model_root="m")
    assert policy.find_dataset_dirs_by_node() == {
        1: [f"{root}/cam_ds/presence/node1/link_4f9c/cam_cam_s0", f"{root}/cam_ds/presence/node1/link_4f9c/p0_cam_s1"],
        2: [f"{root}/cam_ds/presence/node2/link_64b8/cam_cam_s0", f"{root}/cam_ds/presence/node2/link_64b8/p0_cam_s1"]}


def test_camera_sessions_train_next_to_their_stage_heads():
    assert default_camera_model_root("r", "presence") == default_camera_model_root("r", "count") == "r/model_camera"
    assert default_camera_model_root("r", "weapon") == "r/model_weapon"


@pytest.mark.parametrize(("dataset_dir", "capture_root"), [
    ("data/2g4/cam_ds/heatmap/cam_cam_s0", "data/2g4"),
    ("data/2g4/cam_ds/heatmap/cam_cam_s0/", "data/2g4"),
    ("data/2g4/cam_ds/presence/node1/link_4f9c/cam_cam_s0", None),
    ("data/2g4/ds_0/node1/link_4f9c", None),
])
def test_only_a_camera_sessions_heatmap_dataset_has_a_capture_root(dataset_dir, capture_root):
    assert find_heatmap_capture_root(dataset_dir) == capture_root


class _FakeHeatmapHead:
    grid = 4
    config = ModelConfig(stage="presence", k=1)

    def save(self, path):
        self.saved_path = path


def test_a_heatmap_head_trains_with_the_epoch_report_and_carries_its_metrics(monkeypatch):
    calls = []

    def train_heatmap(dataset, *, report=None):
        calls.append((dataset, report))
        return _FakeHeatmapHead(), 7

    monkeypatch.setattr(camera_collect_modes, "load_dataset", lambda dataset_dir: f"loaded {dataset_dir}")
    monkeypatch.setattr(camera_collect_modes, "train_heatmap", train_heatmap)
    model_roots = []
    monkeypatch.setattr(camera_collect_modes, "write_model_run",
                        lambda model_root, manifest, write_files: model_roots.append(model_root))
    report = train_heatmap_head("ds", "m", report=print)
    assert calls == [("loaded ds", print)] and model_roots == ["m"]
    assert (report.line, report.trained_head_key, report.metrics) == (
        "   [OK]   heatmap (4x4, 7 masks) -> m/heatmap.joblib", "heatmap", {"grid": 4, "n": 7})


def test_a_heatmap_dataset_without_masks_is_skipped(monkeypatch):
    def train_heatmap(dataset, *, report=None):
        raise ValueError("dataset has no Label.mask")

    monkeypatch.setattr(camera_collect_modes, "load_dataset", lambda dataset_dir: SimpleNamespace())
    monkeypatch.setattr(camera_collect_modes, "train_heatmap", train_heatmap)
    report = train_heatmap_head("ds", "m")
    assert (report.line, report.trained_head_key) == ("   [SKIP] heatmap: dataset has no Label.mask", None)


def test_the_camera_presence_policy_trains_the_datasets_it_is_given_and_a_session_its_whole_pool(monkeypatch, tmp_path):
    calls = []

    def train_presence(dataset_dirs, *, out_dir, backend, run_provenance):
        calls.append((list(dataset_dirs), out_dir))
        return None, {"n_samples": 1, "class_counts": {}, "train_accuracy": 1.0, "logo": {}}

    monkeypatch.setattr(collect_modes, "train_presence", train_presence)
    root = str(tmp_path)
    (tmp_path / "cam_ds" / "presence" / "node1" / "link_4f9c" / "cam_cam_s0").mkdir(parents=True)
    list(CameraPresenceTrainingPolicy(root=root, model_root="m").train_node(1, ["given"]))
    mode = CameraPresenceMode(root=root, session_id="cam_cam_s1", subject_id="cam", model_root="m")
    list(mode.train_node(1, ["this run only"]))
    assert calls == [(["given"], "m/node1"), ([f"{root}/cam_ds/presence/node1/link_4f9c/cam_cam_s0"], "m/node1")]
