import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wavetrace import Label
from wavetrace.adapters.store.dataset import save_dataset
from wavetrace.domain.groundtruth.dataset import Dataset
from web.routers.paths import scan_paths
from web.routers.pipeline import StartRequest

pytestmark = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")  # torch, in the heatmap fit

NUM_SUBCARRIERS = 2
NUM_WINDOWS_PER_CLASS = 6  # the calibrated svm needs 5 per class in every LOGO fold


def _save_dataset(dataset_dir, class_ids, *, session_id="s0", mask_grid=None, seed=0):
    rng = np.random.default_rng(seed)
    class_ids = np.asarray(class_ids)
    labels = []
    for class_id in class_ids:
        label = Label()
        label.class_id = int(class_id)
        if mask_grid is not None:
            label.mask = [float(class_id)] * (mask_grid * mask_grid)
            label.mask_grid = mask_grid
        labels.append(label)
    num_windows = class_ids.size
    save_dataset(Dataset(
        X_features=(rng.normal(size=(num_windows, 9 * NUM_SUBCARRIERS)) + class_ids[:, None]).astype(np.float32),
        X_image=(rng.normal(size=(num_windows, 4, 8)) + class_ids[:, None, None]).astype(np.float32),
        y=class_ids, t=np.arange(float(num_windows)), labels=labels,
        meta={"K": NUM_SUBCARRIERS, "window": 32, "hop": 16}, session_ids=np.asarray([session_id] * num_windows),
        subject_ids=np.asarray(["u0"] * num_windows),
        X_intercarrier=(rng.normal(size=(num_windows, 27)) + class_ids[:, None]).astype(np.float32)), dataset_dir)


def _both_classes():
    return [0, 1] * NUM_WINDOWS_PER_CLASS


def _train(root, build_recording_runner, *, stage, backend, per_link=False, is_camera=False, train_data=None,
           on_training=None, on_log=None):
    runner, messages = build_recording_runner({"training": on_training, "logs": on_log})
    request = SimpleNamespace(col_stage=stage, train_backend=backend, per_link=per_link, camera_collect=is_camera,
                              train_data=str(train_data or root), train_out=str(root / "ignored_train_out"))
    for callback in (on_training, on_log):
        if callback is not None:
            callback.runner = runner
    runner.run_training_request(request)
    assert not runner.is_running
    assert not (root / "ignored_train_out").exists()
    assert not any("\a" in message for _, message in messages)
    return messages


def _assert_one_done_then_the_pipeline_ends(messages):
    training = messages.payloads("training")
    assert [payload["type"] for payload in training if payload["type"] != "epoch"] == ["train_init", "done"]
    assert set(training[-1]) == {"type", "metrics"}
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]
    return training[-1]["metrics"]


def test_presence_pools_every_session_of_a_node_into_its_per_node_head(tmp_path, build_recording_runner):
    for session_index in range(2):
        for rx_node_id in (1, 2):
            _save_dataset(tmp_path / f"ds_{session_index}" / f"node{rx_node_id}" / "link_4f9c", _both_classes(),
                          session_id=f"sess{session_index}", seed=10 * session_index + rx_node_id)
    messages = _train(tmp_path, build_recording_runner, stage="presence", backend="svm")
    metrics = _assert_one_done_then_the_pipeline_ends(messages)
    node1_metrics = json.loads((tmp_path / "model" / "node1" / "metrics.json").read_text())
    assert node1_metrics["sessions"] == ["sess0", "sess1"] and node1_metrics["backend"] == "svm"
    assert (tmp_path / "model" / "node2" / "model.joblib").exists() and not (tmp_path / "model" / "model.joblib").exists()
    assert metrics["n_samples"] == 2 * len(_both_classes())  # the last head trained is node 2's


def test_train_init_describes_the_whole_pool(tmp_path, build_recording_runner):
    _save_dataset(tmp_path / "ds_0" / "node1" / "link_4f9c", [0, 0, 1])
    _save_dataset(tmp_path / "ds_1" / "node2" / "link_4f9c", [1, 1, 0, 1], session_id="s1")
    messages = _train(tmp_path, build_recording_runner, stage="presence", backend="mlp")
    [train_init] = [payload for payload in messages.payloads("training") if payload["type"] == "train_init"]
    assert (train_init["n_samples"], train_init["distribution"], train_init["sessions"]) == (
        7, {"0": 3, "1": 4}, ["s0", "s1"])


def test_count_trains_every_node_into_the_count_model_root(tmp_path, build_recording_runner):
    for session_index in range(2):
        for level in range(3):
            for rx_node_id in (1, 2):
                _save_dataset(tmp_path / f"count_ds_{session_index}" / f"c{level}" / f"node{rx_node_id}" / "link_4f9c",
                              [level] * NUM_WINDOWS_PER_CLASS, session_id=f"sess{session_index}", seed=level)
    messages = _train(tmp_path, build_recording_runner, stage="count", backend="mlp")
    metrics = _assert_one_done_then_the_pipeline_ends(messages)
    assert metrics["class_counts"] == {"0": 12, "1": 12, "2": 12} and metrics["stage"] == "presence"
    assert sorted(path.parent.name for path in (tmp_path / "model_count").glob("*/model.joblib")) == ["node1", "node2"]
    logs = messages.logs()
    assert any(line.startswith("          per-class: {'0'") and "'2+'" in line for line in logs)
    assert all(line.strip() and "\n" not in line for line in logs), "one non-empty dashboard line per log message"


def _write_weapon_pool(root, *, tags=("4f9c", "64b8"), conditions=("clear", "weapon")):
    for tag in tags:
        for class_id, condition in enumerate(conditions):
            _save_dataset(root / "weapon_ds" / "node1" / f"p0_na_s0_{condition}_link{tag}",
                          [class_id if condition == "weapon" else 0] * NUM_WINDOWS_PER_CLASS, session_id="p0_na_s0",
                          seed=class_id)


def test_per_link_weapon_trains_one_head_per_link_of_the_weapon_pool(tmp_path, build_recording_runner):
    _write_weapon_pool(tmp_path)
    messages = _train(tmp_path, build_recording_runner, stage="weapon", backend="variance", per_link=True)
    metrics = _assert_one_done_then_the_pipeline_ends(messages)
    assert (metrics["backend"], metrics["feature_mode"]) == ("variance", "ic27")
    assert sorted(path.parent.name for path in (tmp_path / "model_weapon" / "node1").glob("*/model.joblib")) == [
        "link4f9c", "link64b8"]


def test_per_node_weapon_pools_the_nodes_single_class_datasets_with_the_chosen_backend(tmp_path,
                                                                                     build_recording_runner):
    _write_weapon_pool(tmp_path)
    metrics = _assert_one_done_then_the_pipeline_ends(
        _train(tmp_path, build_recording_runner, stage="weapon", backend="svm"))
    assert metrics["class_counts"] == {"0": 12, "1": 12} and metrics["backend"] == "svm"
    assert (tmp_path / "model_weapon" / "node1" / "model.joblib").exists()


def test_no_trained_head_logs_the_error_and_still_ends_the_pipeline(tmp_path, build_recording_runner):
    _write_weapon_pool(tmp_path, conditions=("clear",))
    messages = _train(tmp_path, build_recording_runner, stage="weapon", backend="variance")
    assert [payload["type"] for payload in messages.payloads("training")] == ["train_init"]
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]
    assert "[ERROR] nothing trained — need both no-weapon and weapon captures in the pool." in "\n".join(messages.logs())


def test_a_failed_training_logs_the_error_and_still_ends_the_pipeline(tmp_path, build_recording_runner):
    _write_weapon_pool(tmp_path)
    messages = _train(tmp_path, build_recording_runner, stage="weapon", backend="no-such-backend")
    assert any(line.startswith("ERROR Training failed: ValueError: unknown recognition backend") for line in messages.logs())
    assert [payload["type"] for payload in messages.payloads("training")] == []
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]


def test_a_root_without_the_stages_datasets_trains_nothing_and_ends_the_pipeline(tmp_path,
                                                                                build_recording_runner):
    _write_weapon_pool(tmp_path)
    messages = _train(tmp_path, build_recording_runner, stage="presence", backend="mlp")
    assert messages.payloads("training") == [] and messages.payloads("inference") == [{"event": "pipeline_done"}]
    assert any(line.startswith("No presence datasets under") and "capture-profile root" in line
               for line in messages.logs())


def test_a_heatmap_dataset_without_masks_is_skipped_without_a_traceback(tmp_path, build_recording_runner):
    session_dir = tmp_path / "cam_ds" / "heatmap" / "cam_cam_s0"
    _save_dataset(session_dir, _both_classes())
    messages = _train(tmp_path, build_recording_runner, stage="presence", backend="heatmap",
                      train_data=session_dir)
    logs = messages.logs()
    assert any(line.startswith("   [SKIP] heatmap: dataset has no Label.mask") for line in logs)
    assert not any("Traceback" in line or line.startswith("ERROR") for line in logs)
    assert [payload["type"] for payload in messages.payloads("training")] == ["train_init"]
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]


def test_stop_ends_a_heatmap_fit_at_its_next_epoch(tmp_path, build_recording_runner):
    session_dir = tmp_path / "cam_ds" / "heatmap" / "cam_cam_s0"
    _save_dataset(session_dir, _both_classes(), mask_grid=2)

    def stop_after_the_first_epoch(message):
        if json.loads(message)["type"] == "epoch":
            stop_after_the_first_epoch.runner.stop()

    messages = _train(tmp_path, build_recording_runner, stage="presence", backend="heatmap",
                      train_data=session_dir,
                      on_training=stop_after_the_first_epoch)
    assert [payload["type"] for payload in messages.payloads("training")] == ["train_init", "epoch"]
    assert "Training stopped by UI." in messages.logs()
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]
    assert not (tmp_path / "model_camera" / "heatmap.joblib").exists()


@pytest.mark.parametrize("stage", ["presence", "weapon"])
def test_a_heatmap_trains_next_to_its_stages_camera_heads(tmp_path, build_recording_runner, stage):
    session_dir = tmp_path / "cam_ds" / "heatmap" / "cam_cam_s0"
    _save_dataset(session_dir, _both_classes(), mask_grid=2)
    messages = _train(tmp_path, build_recording_runner, stage=stage, backend="heatmap",
                      train_data=session_dir)
    assert _assert_one_done_then_the_pipeline_ends(messages) == {"grid": 2, "n": 2 * NUM_WINDOWS_PER_CLASS}
    model_dir_name = "model_weapon" if stage == "weapon" else "model_camera"
    assert (tmp_path / model_dir_name / "heatmap.joblib").exists()


def test_path_scan_lists_saved_datasets_and_their_parents(tmp_path, monkeypatch):
    _save_dataset(tmp_path / "data" / "2g4" / "ds_0" / "node1" / "link_4f9c", _both_classes())
    monkeypatch.chdir(tmp_path)
    datasets = asyncio.run(scan_paths())["datasets"]
    assert datasets == ["data/2g4/ds_0/node1/link_4f9c", "data/2g4/ds_0/node1"]


def test_the_camera_toggle_trains_the_camera_presence_pool_into_the_camera_model_root(tmp_path,
                                                                                    build_recording_runner):
    for rx_node_id in (1, 2):
        _save_dataset(tmp_path / "cam_ds" / "presence" / f"node{rx_node_id}" / "link_4f9c" / "cam_cam_s0",
                      _both_classes(), seed=rx_node_id)
    _save_dataset(tmp_path / "ds_0" / "node1" / "link_4f9c", _both_classes())
    messages = _train(tmp_path, build_recording_runner, stage="presence", backend="mlp", is_camera=True)
    _assert_one_done_then_the_pipeline_ends(messages)
    assert sorted(path.parent.name for path in (tmp_path / "model_camera").glob("*/model.joblib")) == ["node1", "node2"]
    assert not (tmp_path / "model").exists()


def test_the_dashboard_request_carries_the_camera_toggle():
    assert StartRequest(camera_collect=True).camera_collect is True
    assert StartRequest().camera_collect is False


def test_the_camera_toggle_has_no_count_pool_and_trains_nothing(tmp_path, build_recording_runner):
    _save_dataset(tmp_path / "cam_ds" / "presence" / "node1" / "link_4f9c" / "cam_cam_s0", _both_classes())
    _save_dataset(tmp_path / "count_ds_0" / "c1" / "node1" / "link_4f9c", [1] * NUM_WINDOWS_PER_CLASS)
    messages = _train(tmp_path, build_recording_runner, stage="count", backend="mlp", is_camera=True)
    assert any(line.startswith("ERROR: no camera count pool; untick camera") for line in messages.logs())
    assert messages.payloads("training") == [] and messages.payloads("inference") == [{"event": "pipeline_done"}]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cam_ds", "count_ds_0"]


def test_stop_ends_sklearn_training_before_the_next_nodes_head(tmp_path, build_recording_runner):
    for rx_node_id in (1, 2):
        _save_dataset(tmp_path / "ds_0" / f"node{rx_node_id}" / "link_4f9c", _both_classes(), seed=rx_node_id)

    def stop_after_the_first_head(message):
        if "[OK]   node 1" in message:
            stop_after_the_first_head.runner.stop()

    messages = _train(tmp_path, build_recording_runner, stage="presence", backend="mlp",
                      on_log=stop_after_the_first_head)
    assert [path.parent.name for path in (tmp_path / "model").glob("*/model.joblib")] == ["node1"]
    assert "stopped: trained heads [1]; every other head keeps its previous model." in messages.logs()
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]
