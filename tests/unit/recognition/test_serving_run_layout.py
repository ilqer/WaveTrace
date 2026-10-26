"""The live loaders serve each node's current run, a node trained before runs from its flat files, and
never a partial run: a retrain interrupted between model.joblib and metrics.json leaves the node on its
previous model and that model's own LOGO weight."""

import json

import numpy as np
import pytest

from fixtures.synthetic_csi import calibrate_quiet_baseline
from wavetrace import Label
from wavetrace.adapters.recognition.heads import build_presence_head, build_weapon_head
from wavetrace.adapters.store.calibration import save_calibration
from wavetrace.adapters.store.dataset import save_dataset
from wavetrace.adapters.store.model_runs import METRICS_FILE_NAME, MODEL_FILE_NAME, is_run_id, write_model_run
from wavetrace.adapters.store.run_manifest import build_run_manifest
from wavetrace.domain.groundtruth.dataset import Dataset
from wavetrace.domain.model_config import ModelConfig
from wavetrace.recognition import train_presence
from wavetrace.recognition.serving import load_presence_node_servers
from wavetrace.recognition.weapon_serving import load_weapon_links


def _calibrate(calibration_root, node_ids):
    result, _gain_lock = calibrate_quiet_baseline(num_antennas=1, num_subcarriers=32, sample_rate_hz=100.0,
                                                  num_frames=60, seed=7, baseline_packets=60)
    for node_id in node_ids:
        save_calibration(result, calibration_root / f"node{node_id}")


def _presence_head():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 18)).astype(np.float32)
    y = (np.arange(40) % 2).astype(np.int64)
    X[y == 1] += 3.0
    return build_presence_head(ModelConfig(stage="presence", k=2)).fit(X, y)


def _weapon_head():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 27)).astype(np.float32)
    y = (np.arange(40) % 2).astype(np.int64)
    X[y == 1, 9] -= 3.0
    return build_weapon_head(ModelConfig(stage="weapon", k=12, backend="variance")).fit(X, y)


def _write_files(head, logo_accuracy):
    def write_files(model_dir):
        head.save(model_dir / MODEL_FILE_NAME)
        (model_dir / METRICS_FILE_NAME).write_text(json.dumps({"logo": {"session": {"accuracy": logo_accuracy}}}))
    return write_files


def _write_run(model_dir, head, logo_accuracy):
    return write_model_run(model_dir, build_run_manifest(head.config, head.contract, []),
                           _write_files(head, logo_accuracy))


def test_presence_nodes_load_from_current_runs_and_from_flat_dirs_trained_before_runs(tmp_path):
    _calibrate(tmp_path / "cal", [1, 2])
    model_dir = tmp_path / "model" / "node1"
    model_dir.mkdir(parents=True)
    _write_files(_presence_head(), 0.6)(model_dir)  # the flat layout every model before runs has
    _write_run(tmp_path / "model" / "node2", _presence_head(), 0.7)
    _write_run(tmp_path / "model" / "node2", _presence_head(), 0.8)

    nodes, logo_accuracy_by_node = load_presence_node_servers(tmp_path / "cal", tmp_path / "model")

    assert sorted(nodes) == [1, 2]
    assert logo_accuracy_by_node == {1: 0.6, 2: 0.8}


def _two_session_dataset(shift):
    sample_count = 80
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (sample_count, 18)).astype(np.float32)
    y = (np.arange(sample_count) % 2).astype(np.int64)
    X[y == 1] += shift
    labels = []
    for i, class_id in enumerate(y):
        label = Label()
        label.class_id = int(class_id)
        label.name = "present" if class_id else "absent"
        label.timestamp = float(i)
        labels.append(label)
    return Dataset(
        X_features=X, X_image=np.zeros((sample_count, 1, 1), np.float32), y=y,
        t=np.arange(sample_count, dtype=np.float64), labels=labels, meta={"K": 2, "window": 64, "hop": 16},
        session_ids=np.asarray([f"s{i // 40}" for i in range(sample_count)], dtype=object),
        subject_ids=np.full(sample_count, "u0", dtype=object))


def test_a_retrain_interrupted_between_model_and_metrics_leaves_the_node_on_its_previous_run(tmp_path,
                                                                                             monkeypatch):
    _calibrate(tmp_path / "cal", [1])
    node_dir = tmp_path / "model" / "node1"
    _, first_metrics = train_presence([save_dataset(_two_session_dataset(3.0), tmp_path / "ds_a")], node_dir)
    nodes_before, logo_before = load_presence_node_servers(tmp_path / "cal", tmp_path / "model")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt  # Ctrl+C once model.joblib is written, before metrics.json

    monkeypatch.setattr(json, "dump", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train_presence([save_dataset(_two_session_dataset(0.5), tmp_path / "ds_b")], node_dir)
    monkeypatch.undo()

    nodes, logo_accuracy_by_node = load_presence_node_servers(tmp_path / "cal", tmp_path / "model")
    assert logo_accuracy_by_node == logo_before == {1: first_metrics["logo"]["session"]["accuracy"]}
    probe_rows = np.random.default_rng(1).normal(0, 1, (8, 18)).astype(np.float32)
    assert np.array_equal(nodes[1].session.head.predict_proba(probe_rows),
                          nodes_before[1].session.head.predict_proba(probe_rows))
    run_dirs = [path for path in node_dir.iterdir() if is_run_id(path.name)]
    assert len(run_dirs) == 1 and all((run_dir / METRICS_FILE_NAME).is_file() for run_dir in run_dirs)


def test_weapon_links_load_per_link_runs_and_a_flat_per_node_head(tmp_path):
    _calibrate(tmp_path / "cal", [1, 2])
    _write_run(tmp_path / "model_weapon" / "node1" / "link4f9c", _weapon_head(), 0.9)
    _write_run(tmp_path / "model_weapon" / "node1" / "link64b8", _weapon_head(), 0.5)
    flat_node_dir = tmp_path / "model_weapon" / "node2"
    flat_node_dir.mkdir(parents=True)
    _write_files(_weapon_head(), 0.8)(flat_node_dir)

    entries = load_weapon_links(tmp_path / "cal", tmp_path / "model_weapon")

    assert sorted(entries, key=str) == sorted([("4f9c", 1), ("64b8", 1), (None, 2)], key=str)
    assert entries[("4f9c", 1)].weight > entries[("64b8", 1)].weight  # each from its own run's metrics


def test_a_per_node_weapon_run_serves_any_tx_into_that_node(tmp_path):
    _calibrate(tmp_path / "cal", [1])
    _write_run(tmp_path / "model_weapon" / "node1", _weapon_head(), 0.9)
    assert list(load_weapon_links(tmp_path / "cal", tmp_path / "model_weapon")) == [(None, 1)]
