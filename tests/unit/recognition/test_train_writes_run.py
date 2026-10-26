"""Training writes each head as one versioned run: model, metrics and run manifest together, made current
only once all are on disk, with the manifest hashing exactly the dataset files training read."""

import hashlib
import json
import logging

import numpy as np
import pytest

from wavetrace import Label
from wavetrace.adapters.recognition.heads import load_presence_head, load_weapon_head
from wavetrace.adapters.store.dataset import list_dataset_files, save_dataset
from wavetrace.adapters.store.model_runs import (
    HEATMAP_FILE_NAME, MANIFEST_FILE_NAME, METRICS_FILE_NAME, MODEL_FILE_NAME, is_run_id, resolve_run_dir,
)
from wavetrace.adapters.store import run_manifest
from wavetrace.adapters.store.run_manifest import hash_files
from wavetrace.application.collect_mesh import train_every_node
from wavetrace.domain.groundtruth.dataset import Dataset
from wavetrace.recognition import train_presence, train_weapon
from wavetrace.recognition.collect_modes import PresenceCollectMode


def _labels(names, class_ids, masks=None):
    labels = []
    for i, (name, class_id) in enumerate(zip(names, class_ids)):
        label = Label()
        label.class_id = int(class_id)
        label.name = name
        label.timestamp = float(i)
        if masks is not None:
            label.mask = masks[i]
        labels.append(label)
    return labels


def _presence_dataset(shift=3.0, sample_count=60, k=2):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (sample_count, 9 * k)).astype(np.float32)
    y = (np.arange(sample_count) % 2).astype(np.int64)
    X[y == 1] += shift
    return Dataset(
        X_features=X, X_image=np.zeros((sample_count, 1, 1), np.float32), y=y,
        t=np.arange(sample_count, dtype=np.float64),
        labels=_labels(["present" if class_id else "absent" for class_id in y], y),
        meta={"K": k, "window": 64, "hop": 16, "num_subcarriers": 128},
        session_ids=np.full(sample_count, "s0", dtype=object), subject_ids=np.full(sample_count, "u0", dtype=object))


def _manifest(run_dir):
    return json.loads((run_dir / MANIFEST_FILE_NAME).read_text())


def test_train_presence_writes_model_metrics_and_manifest_as_one_current_run(tmp_path):
    dataset_dir = save_dataset(_presence_dataset(), tmp_path / "ds")
    head, metrics = train_presence([dataset_dir], tmp_path / "model")

    run_dir = resolve_run_dir(tmp_path / "model")
    assert is_run_id(run_dir.name)
    assert sorted(path.name for path in run_dir.iterdir()) == [METRICS_FILE_NAME, MODEL_FILE_NAME, MANIFEST_FILE_NAME]
    assert json.loads((run_dir / METRICS_FILE_NAME).read_text()) == metrics
    manifest = _manifest(run_dir)
    assert manifest["dataset_hash"] == hash_files(list_dataset_files(dataset_dir))
    assert manifest["contract"] == head.contract.to_dict() and manifest["contract"]["subcarrier_width"] == 128
    assert manifest["seed"] == head.config.seed
    assert load_presence_head(tmp_path / "model" / MODEL_FILE_NAME).contract == head.contract


def test_retraining_on_other_data_makes_a_new_current_run_and_keeps_the_old_one(tmp_path):
    first_dir = save_dataset(_presence_dataset(shift=3.0), tmp_path / "ds_first")
    second_dir = save_dataset(_presence_dataset(shift=4.0), tmp_path / "ds_second")
    train_presence([first_dir], tmp_path / "model")
    first_run = resolve_run_dir(tmp_path / "model")
    train_presence([second_dir], tmp_path / "model")
    second_run = resolve_run_dir(tmp_path / "model")

    assert second_run != first_run and first_run.is_dir()
    assert _manifest(second_run)["dataset_hash"] != _manifest(first_run)["dataset_hash"]
    assert _manifest(second_run)["config_hash"] == _manifest(first_run)["config_hash"]


def test_one_training_call_reads_the_git_state_once_and_hashes_each_heads_own_files(tmp_path, monkeypatch):
    git_reads = []

    def count_git_reads(source_dir):
        git_reads.append(source_dir)
        return "0" * 40, True

    monkeypatch.setattr(run_manifest, "read_git_state", count_git_reads)
    dataset_dirs_by_node = {
        node_id: [str(save_dataset(_presence_dataset(shift=2.0 + node_id), tmp_path / f"ds{node_id}"))]
        for node_id in (1, 2)}
    policy = PresenceCollectMode(root=str(tmp_path), model_root=str(tmp_path / "model"))

    reports = train_every_node(policy, [1, 2], dataset_dirs_by_node, log=lambda line: None,
                               log_error=lambda line: None, should_stop=lambda: False)

    assert [report.trained_head_key for report in reports] == [1, 2] and len(git_reads) == 1
    for node_id, [dataset_dir] in dataset_dirs_by_node.items():
        manifest = _manifest(resolve_run_dir(tmp_path / "model" / f"node{node_id}"))
        assert manifest["git_commit"] == "0" * 40 and manifest["has_uncommitted_changes"] is True
        assert manifest["dataset_hash"] == hash_files(list_dataset_files(dataset_dir))


def test_train_weapon_writes_its_head_as_a_run(tmp_path):
    sample_count = 60
    rng = np.random.default_rng(0)
    X_intercarrier = rng.normal(0, 1, (sample_count, 27)).astype(np.float32)
    y = (np.arange(sample_count) % 2).astype(np.int64)
    X_intercarrier[y == 1, 9] -= 3.0
    dataset = Dataset(
        X_features=np.zeros((sample_count, 108), np.float32), X_image=np.zeros((sample_count, 1, 1), np.float32),
        y=y, t=np.arange(sample_count, dtype=np.float64),
        labels=_labels(["weapon" if class_id else "no_weapon" for class_id in y], y),
        meta={"K": 12, "window": 64, "hop": 8}, session_ids=np.full(sample_count, "s0", dtype=object),
        subject_ids=np.full(sample_count, "u0", dtype=object), X_intercarrier=X_intercarrier)
    dataset_dir = save_dataset(dataset, tmp_path / "ds")
    head, _ = train_weapon([dataset_dir], tmp_path / "model_weapon" / "node1")

    run_dir = resolve_run_dir(tmp_path / "model_weapon" / "node1")
    assert _manifest(run_dir)["dataset_hash"] == hash_files(list_dataset_files(dataset_dir))
    assert load_weapon_head(tmp_path / "model_weapon" / "node1" / MODEL_FILE_NAME).feature_mode == head.feature_mode


def test_the_heatmap_head_trains_as_a_run_of_its_model_root(tmp_path):
    pytest.importorskip("torch")
    from wavetrace.recognition.camera_collect_modes import train_heatmap_head
    from wavetrace.recognition.heatmap import HeatmapHead

    sample_count, grid = 8, 2
    masks = [[float(i % 2)] * (grid * grid) for i in range(sample_count)]
    dataset = Dataset(
        X_features=np.zeros((sample_count, 9), np.float32),
        X_image=np.random.default_rng(0).normal(0, 1, (sample_count, 4, 8)).astype(np.float32),
        y=np.zeros(sample_count, np.int64), t=np.arange(sample_count, dtype=np.float64),
        labels=_labels(["absent"] * sample_count, [0] * sample_count, masks), meta={"K": 1},
        session_ids=np.full(sample_count, "s0", dtype=object), subject_ids=np.full(sample_count, "u0", dtype=object))
    dataset_dir = save_dataset(dataset, tmp_path / "cam_ds" / "heatmap" / "cam_s0")

    report = train_heatmap_head(str(dataset_dir), str(tmp_path / "model_camera"))

    run_dir = resolve_run_dir(tmp_path / "model_camera", HEATMAP_FILE_NAME)
    assert report.trained_head_key == "heatmap" and is_run_id(run_dir.name)
    assert _manifest(run_dir)["dataset_hash"] == hash_files(list_dataset_files(dataset_dir))
    assert HeatmapHead.load(tmp_path / "model_camera" / HEATMAP_FILE_NAME).grid == grid


def test_dataset_dirs_given_as_a_generator_are_trained_on_and_hashed(tmp_path):
    dataset_dir = save_dataset(_presence_dataset(), tmp_path / "ds")
    train_presence((dataset_dir for _ in range(1)), tmp_path / "model")

    dataset_hash = _manifest(resolve_run_dir(tmp_path / "model"))["dataset_hash"]
    assert dataset_hash == hash_files(list_dataset_files(dataset_dir)) != hashlib.sha256().hexdigest()


@pytest.mark.parametrize("hash_error", [PermissionError("permission denied"),
                                        ValueError("Paths don't have the same drive")])
def test_dataset_files_that_cannot_be_hashed_keep_the_fitted_head_with_a_null_hash(tmp_path, monkeypatch, caplog,
                                                                                  hash_error):
    def fail_to_hash(paths):
        raise hash_error

    monkeypatch.setattr(run_manifest, "hash_files", fail_to_hash)
    dataset_dir = save_dataset(_presence_dataset(), tmp_path / "ds")
    with caplog.at_level(logging.WARNING):
        head, _ = train_presence([dataset_dir], tmp_path / "model")

    run_dir = resolve_run_dir(tmp_path / "model")
    assert _manifest(run_dir)["dataset_hash"] is None
    assert load_presence_head(run_dir / MODEL_FILE_NAME).contract == head.contract
    [record] = caplog.records
    assert str(hash_error) in record.getMessage()
