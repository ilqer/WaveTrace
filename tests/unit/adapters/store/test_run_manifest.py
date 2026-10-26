"""The RunManifest a run saves: for a trained run its git state, config and dataset hashes, seed, library
versions, PipelineContract and creation time, read once per training call; for an uploaded run its time."""

import hashlib
import json
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import sklearn

from wavetrace.adapters.store.dataset import list_dataset_files, save_dataset
from wavetrace.adapters.store import run_manifest
from wavetrace.adapters.store.run_manifest import (
    RunProvenance,
    build_run_manifest,
    build_upload_manifest,
    hash_files,
    read_git_state,
)
from wavetrace.domain.contracts import SCHEMA_VERSION, PipelineContract
from wavetrace.domain.groundtruth.dataset import Dataset
from wavetrace.domain.model_config import ModelConfig

_REPO_DIR = Path(__file__).resolve().parents[4]


def _write(root, files):
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return [root / relative_path for relative_path in files]


def test_the_dataset_hash_ignores_input_order_and_where_the_data_root_lives(tmp_path):
    files = {"ds_0/node1/features.npy": b"x" * 10, "ds_1/node1/features.npy": b"y" * 10}
    here = _write(tmp_path / "here", files)
    moved = _write(tmp_path / "elsewhere" / "deeper", files)
    assert hash_files(here) == hash_files(list(reversed(here))) == hash_files(moved)


def test_the_dataset_hash_covers_every_path_and_every_byte(tmp_path):
    reference = hash_files(_write(tmp_path / "a", {"one.npy": b"x", "two.npy": b"yz"}))
    assert hash_files(_write(tmp_path / "b", {"one.npy": b"x", "TWO.npy": b"yz"})) != reference
    assert hash_files(_write(tmp_path / "c", {"one.npy": b"x", "two.npy": b"yZ"})) != reference
    # the same bytes moved across a file boundary must not collide
    assert hash_files(_write(tmp_path / "d", {"one.npy": b"xy", "two.npy": b"z"})) != reference


def test_the_dataset_hash_reads_files_larger_than_one_chunk_whole(tmp_path):
    large = bytes(range(256)) * 12_289  # just over 3 MiB: three full 1 MiB chunks and a tail
    reference = hash_files(_write(tmp_path / "a", {"images.npy": large}))
    assert hash_files(_write(tmp_path / "b", {"images.npy": large[:-1] + b"\x01"})) != reference
    assert hash_files(_write(tmp_path / "c", {"images.npy": large})) == reference


def test_the_dataset_hash_format_is_pinned(tmp_path):
    """Pins the on-disk digest of a tiny set, so a format change cannot silently orphan saved manifests."""
    files = _write(tmp_path, {"ds/meta.json": b"{}", "ds/features.npy": b"\x00\x01"})
    assert hash_files(files) == "f29625a64f2f45457ee257aab68c8d4ed8c7bfe10f0c91947a4240ad8db549e8"


def test_outside_a_git_checkout_the_git_state_is_unknown(tmp_path):
    assert read_git_state(tmp_path) == (None, None)


def test_inside_the_checkout_the_git_state_is_head_and_a_dirty_flag(tmp_path):
    if shutil.which("git") is None or not (_REPO_DIR / ".git").exists():
        pytest.skip("an installed wheel has no checkout to read")
    git_commit, has_uncommitted_changes = read_git_state(_REPO_DIR)
    assert re.fullmatch(r"[0-9a-f]{40}", git_commit) and isinstance(has_uncommitted_changes, bool)


def test_the_manifest_records_config_contract_seed_versions_and_time(tmp_path):
    config = ModelConfig(stage="presence", k=2, seed=7)
    contract = PipelineContract(target_sample_rate_hz=100.0, window_frames=128, hop_frames=32, subcarrier_width=192)
    dataset_files = _write(tmp_path, {"ds/features.npy": b"data"})

    manifest = build_run_manifest(config, contract, dataset_files)

    canonical_config = ('{"backend":"mlp","frame_average":1,"fs_tol":0.1,"hidden":32,"hop":32,"k":2,"seed":7,'
                        '"stage":"presence","subtract_baseline":false,"subtract_ic_baseline":false,"window":128}')
    assert manifest.origin == "trained"
    assert manifest.config_hash == hashlib.sha256(canonical_config.encode()).hexdigest()
    assert manifest.dataset_hash == hash_files(dataset_files)
    assert manifest.seed == 7 and manifest.contract == contract and manifest.schema_version == SCHEMA_VERSION
    assert manifest.library_versions.python == platform.python_version()
    assert manifest.library_versions.numpy == np.__version__
    assert manifest.library_versions.scikit_learn == sklearn.__version__
    assert abs(time.time() - manifest.created_timestamp_s) < 60
    assert json.loads(json.dumps(manifest.to_dict()))["contract"]["subcarrier_width"] == 192


def test_the_saved_manifest_keys_are_pinned():
    """The keys are a file format: renaming one orphans every saved run_manifest.json."""
    config = ModelConfig(stage="presence", k=2)
    contract = PipelineContract(target_sample_rate_hz=100.0, window_frames=128, hop_frames=32)
    saved = build_run_manifest(config, contract, []).to_dict()
    assert sorted(saved) == ["config_hash", "contract", "created_timestamp_s", "dataset_hash", "git_commit",
                             "has_uncommitted_changes", "library_versions", "origin", "schema_version", "seed"]
    assert sorted(saved["library_versions"]) == ["numpy", "python", "scikit_learn", "torch", "wavetrace"]


def test_an_uploaded_run_records_its_origin_and_time_and_claims_nothing_else():
    manifest = build_upload_manifest()
    assert manifest.origin == "uploaded" and abs(time.time() - manifest.created_timestamp_s) < 60
    unknown_fields = {key: value for key, value in manifest.to_dict().items()
                      if key not in ("origin", "created_timestamp_s")}
    assert set(unknown_fields.values()) == {None}


def test_one_run_provenance_reads_the_git_state_once_for_every_run_it_builds(monkeypatch):
    git_reads = []

    def count_git_reads(source_dir):
        git_reads.append(source_dir)
        return "0" * 40, False

    monkeypatch.setattr(run_manifest, "read_git_state", count_git_reads)
    contract = PipelineContract(target_sample_rate_hz=100.0, window_frames=128, hop_frames=32)
    run_provenance = RunProvenance()
    manifests = [build_run_manifest(ModelConfig(stage="presence", k=2), contract, [], run_provenance)
                 for _ in range(3)]

    assert len(git_reads) == 1
    assert {manifest.git_commit for manifest in manifests} == {"0" * 40}


def test_any_config_change_changes_the_config_hash(tmp_path):
    contract = PipelineContract(target_sample_rate_hz=100.0, window_frames=128, hop_frames=32)
    first = build_run_manifest(ModelConfig(stage="presence", k=2), contract, [])
    again = build_run_manifest(ModelConfig(stage="presence", k=2), contract, [])
    other = build_run_manifest(ModelConfig(stage="presence", k=2, hidden=16), contract, [])
    assert first.config_hash == again.config_hash != other.config_hash


def _dataset(has_intercarrier):
    sample_count = 4
    return Dataset(
        X_features=np.zeros((sample_count, 9), np.float32), X_image=np.zeros((sample_count, 1, 1), np.float32),
        y=np.zeros(sample_count, np.int64), t=np.arange(sample_count, dtype=np.float64), labels=[],
        meta={"K": 1}, session_ids=np.array([], dtype=object), subject_ids=np.array([], dtype=object),
        X_intercarrier=np.zeros((sample_count, 27), np.float32) if has_intercarrier else None)


def test_the_dataset_files_are_every_file_a_saved_dataset_holds(tmp_path):
    with_intercarrier = save_dataset(_dataset(has_intercarrier=True), tmp_path / "ic")
    without_intercarrier = save_dataset(_dataset(has_intercarrier=False), tmp_path / "plain")
    for dataset_dir in (with_intercarrier, without_intercarrier):
        assert sorted(list_dataset_files(dataset_dir)) == sorted(dataset_dir.iterdir())
