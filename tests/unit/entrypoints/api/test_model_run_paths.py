"""The dashboard finds models trained as runs: the path scan lists each model dir's logical
`<model_dir>/model.joblib` once (never a run dir or the `current` link), the heatmap head loads from
its model root's current run, and an uploaded model becomes a new current run marked uploaded."""

import asyncio
import base64
import json
import os

import numpy as np
import pytest

from wavetrace import Label
from wavetrace.adapters.recognition.heads import build_presence_head
from wavetrace.adapters.store.dataset import save_dataset
from wavetrace.adapters.store.model_runs import (
    HEATMAP_FILE_NAME, MANIFEST_FILE_NAME, MODEL_FILE_NAME, is_run_id, resolve_run_dir, write_model_run,
)
from wavetrace.adapters.store.run_manifest import build_run_manifest, build_upload_manifest
from wavetrace.domain.groundtruth.dataset import Dataset
from wavetrace.domain.model_config import ModelConfig
from wavetrace.entrypoints.api.routers import model as model_router
from wavetrace.entrypoints.api.routers.paths import scan_paths


def _presence_head():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 18)).astype(np.float32)
    y = (np.arange(40) % 2).astype(np.int64)
    X[y == 1] += 3.0
    return build_presence_head(ModelConfig(stage="presence", k=2)).fit(X, y)


def _write_run(model_dir, head):
    write_model_run(model_dir, build_run_manifest(head.config, head.contract, []),
                    lambda run_dir: head.save(run_dir / MODEL_FILE_NAME))


def test_the_path_scan_lists_each_model_dir_once_whatever_its_layout(tmp_path, monkeypatch):
    head = _presence_head()
    for node_dir in ("node1", "node2"):
        _write_run(tmp_path / "data" / "2g4" / "model" / node_dir, head)
    _write_run(tmp_path / "data" / "2g4" / "model" / "node2", head)
    flat_dir = tmp_path / "output" / "model"
    flat_dir.mkdir(parents=True)
    head.save(flat_dir / MODEL_FILE_NAME)
    monkeypatch.chdir(tmp_path)

    models = asyncio.run(scan_paths())["models"]

    assert models == ["data/2g4/model/node1/model.joblib", "data/2g4/model/node2/model.joblib",
                      "output/model/model.joblib", "data/2g4/model/node1", "data/2g4/model/node2"]


def test_the_heatmap_head_loads_from_its_model_roots_current_run(tmp_path, build_recording_runner):
    pytest.importorskip("torch")
    from wavetrace.recognition.camera_collect_modes import train_heatmap_head

    sample_count, grid = 8, 2
    labels = []
    for i in range(sample_count):
        label = Label()
        label.mask = [float(i % 2)] * (grid * grid)
        labels.append(label)
    dataset_dir = save_dataset(Dataset(
        X_features=np.zeros((sample_count, 9), np.float32),
        X_image=np.random.default_rng(0).normal(0, 1, (sample_count, 4, 8)).astype(np.float32),
        y=np.zeros(sample_count, np.int64), t=np.arange(sample_count, dtype=np.float64), labels=labels,
        meta={"K": 1}, session_ids=np.full(sample_count, "s0", dtype=object),
        subject_ids=np.full(sample_count, "u0", dtype=object)), tmp_path / "cam_ds" / "heatmap" / "cam_s0")
    train_heatmap_head(str(dataset_dir), str(tmp_path / "model_camera"))
    runner, _ = build_recording_runner()

    assert not (tmp_path / "model_camera" / HEATMAP_FILE_NAME).exists()  # only inside the run
    assert runner._load_heatmap_head(str(tmp_path / "model_camera")).grid == grid
    assert runner._load_heatmap_head(str(tmp_path / "elsewhere")) is None


def test_an_uploaded_model_becomes_the_current_run_marked_uploaded(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / MODEL_FILE_NAME).write_bytes(b"flat model from before runs")
    monkeypatch.setattr(model_router, "ALLOWED_ROOT", os.path.realpath(output_dir))

    def upload(model_bytes):
        request = model_router.ModelUploadRequest(file_b64=base64.b64encode(model_bytes).decode(),
                                                  dest=str(model_dir / MODEL_FILE_NAME))
        return asyncio.run(model_router.model_upload(request))

    assert upload(b"first upload")["status"] == "uploaded"
    first_run = resolve_run_dir(model_dir)
    assert upload(b"second upload")["status"] == "uploaded"
    second_run = resolve_run_dir(model_dir)

    assert is_run_id(first_run.name) and is_run_id(second_run.name) and first_run != second_run
    assert (second_run / MODEL_FILE_NAME).read_bytes() == b"second upload"
    assert (first_run / MODEL_FILE_NAME).read_bytes() == b"first upload"
    manifest = json.loads((second_run / MANIFEST_FILE_NAME).read_text())
    assert manifest["origin"] == "uploaded" and manifest["dataset_hash"] is None


@pytest.fixture
def upload(tmp_path, monkeypatch):
    """upload(dest, model_bytes) -> the endpoint's response, with cwd at tmp_path and output/m made."""
    (tmp_path / "output" / "m").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_router, "ALLOWED_ROOT", os.path.realpath("output"))

    def upload(dest, model_bytes=b"model"):
        request = model_router.ModelUploadRequest(file_b64=base64.b64encode(model_bytes).decode(), dest=dest)
        return asyncio.run(model_router.model_upload(request))

    return upload


def _tree(directory):
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*"))


def test_an_upload_through_current_or_into_a_run_dir_becomes_a_new_run_of_the_model_dir(tmp_path, upload):
    assert upload("output/m/model.joblib", b"first")["status"] == "uploaded"
    first_run = resolve_run_dir("output/m")

    assert upload("output/m/current/model.joblib", b"through current")["dest"] == os.path.realpath(
        "output/m/model.joblib")
    assert (resolve_run_dir("output/m") / MODEL_FILE_NAME).read_bytes() == b"through current"
    upload(f"output/m/{first_run.name}/model.joblib", b"into a run dir")
    assert (resolve_run_dir("output/m") / MODEL_FILE_NAME).read_bytes() == b"into a run dir"

    assert sorted(path.name for path in first_run.iterdir()) == [MODEL_FILE_NAME, MANIFEST_FILE_NAME]
    assert sum(is_run_id(path.name) for path in (tmp_path / "output" / "m").iterdir()) == 3


def test_an_upload_keeps_the_files_of_the_run_it_replaces_but_the_uploaded_one(tmp_path, upload):
    write_model_run("output/m", build_upload_manifest(), lambda run_dir: (
        (run_dir / MODEL_FILE_NAME).write_bytes(b"old"), (run_dir / HEATMAP_FILE_NAME).write_bytes(b"heatmap")))

    assert upload("output/m/model.joblib", b"new")["status"] == "uploaded"

    assert (resolve_run_dir("output/m") / MODEL_FILE_NAME).read_bytes() == b"new"
    assert (resolve_run_dir("output/m", HEATMAP_FILE_NAME) / HEATMAP_FILE_NAME).read_bytes() == b"heatmap"


@pytest.mark.parametrize("dest", ["output", "output/", "output/m/", "output/m"])
def test_an_upload_to_a_directory_is_refused_and_writes_nothing(tmp_path, upload, dest):
    tree_before = _tree(tmp_path)
    response = upload(dest)
    assert response == {"error": f"dest must name a model file inside output/, not a directory: {dest}"}
    assert _tree(tmp_path) == tree_before


def test_an_upload_never_writes_a_run_beside_a_sandbox_named_like_a_run_dir(tmp_path, upload, monkeypatch):
    sandbox = tmp_path / "20261026T153012Z-a1b2c3"  # its model dir would be its parent
    sandbox.mkdir()
    monkeypatch.setattr(model_router, "ALLOWED_ROOT", os.path.realpath(sandbox))
    tree_before = _tree(tmp_path)
    dest = f"{sandbox.name}/model.joblib"
    assert upload(dest) == {"error": f"path escapes output/: {dest}"}
    assert _tree(tmp_path) == tree_before
