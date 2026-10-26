"""Builds the RunManifest saved beside each model run: for a trained run its code state, config and dataset
hashes, seed, library versions, PipelineContract and creation time; for an uploaded run its time only."""

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict
from functools import cached_property
from importlib import metadata
from pathlib import Path

import numpy as np
import sklearn

from wavetrace.domain.contracts import SCHEMA_VERSION, LibraryVersions, PipelineContract, RunManifest
from wavetrace.domain.model_config import ModelConfig

_HASH_CHUNK_BYTES = 1 << 20
_PACKAGE_DIR = Path(__file__).resolve().parents[2]
_GIT_TIMEOUT_S = 10


class RunProvenance:
    """The code one training call runs, shared by every run it writes: the repo state (two subprocess calls)
    and the library versions are read for its first manifest only. Make one per training call, not one per
    process: a long-running dashboard outlives commits."""

    def __init__(self, source_dir=_PACKAGE_DIR):
        self._source_dir = source_dir

    @cached_property
    def git_state(self) -> tuple[str | None, bool | None]:
        return read_git_state(self._source_dir)

    @cached_property
    def library_versions(self) -> LibraryVersions:
        return LibraryVersions(python=platform.python_version(), numpy=np.__version__,
                               scikit_learn=sklearn.__version__, wavetrace=_find_installed_version("wavetrace"),
                               torch=_find_installed_version("torch"))


def build_run_manifest(config: ModelConfig, contract: PipelineContract, dataset_files: Iterable,
                       run_provenance: RunProvenance | None = None) -> RunManifest:
    """`dataset_files`: every file the run's training read. O(dataset bytes) for the dataset hash, plus the
    repo-state calls when no `run_provenance` of the training call is given. Files that cannot be hashed
    leave `dataset_hash` None with a warning, never cost the fitted head."""
    if run_provenance is None:
        run_provenance = RunProvenance()
    git_commit, has_uncommitted_changes = run_provenance.git_state
    canonical_config = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return RunManifest(
        origin="trained", git_commit=git_commit, has_uncommitted_changes=has_uncommitted_changes,
        config_hash=hashlib.sha256(canonical_config.encode()).hexdigest(),
        dataset_hash=_hash_dataset_files_or_none(dataset_files),
        seed=config.seed, library_versions=run_provenance.library_versions, contract=contract,
        created_timestamp_s=int(time.time()), schema_version=SCHEMA_VERSION,
    )


def build_upload_manifest() -> RunManifest:
    return RunManifest(origin="uploaded", git_commit=None, has_uncommitted_changes=None, config_hash=None,
                       dataset_hash=None, seed=None, library_versions=None, contract=None,
                       created_timestamp_s=int(time.time()), schema_version=None)


def _hash_dataset_files_or_none(dataset_files: Iterable) -> str | None:
    try:
        return hash_files(dataset_files)
    except (OSError, ValueError) as error:  # ValueError: os.path.commonpath across Windows drives
        logging.getLogger(__name__).warning("could not hash the dataset files (%s); saving the run with "
                                            "dataset_hash null", error)
        return None


def hash_files(paths: Iterable) -> str:
    """Hex sha256 over every file, in order of its path relative to the files' common parent dir, so
    moving the data root keeps the hash. Each file enters as path, NUL, 8-byte big-endian size, then its
    bytes, so no two file sets share a digest by moving bytes across a file boundary. O(total bytes) time
    plus O(F log F) for the sort; O(1) memory, one reused 1 MiB buffer."""
    paths = [Path(os.path.abspath(path)) for path in paths]
    digest = hashlib.sha256()
    if not paths:
        return digest.hexdigest()
    common_dir = Path(os.path.commonpath([path.parent for path in paths]))
    chunk = bytearray(_HASH_CHUNK_BYTES)
    chunk_view = memoryview(chunk)
    for relative_path, path in sorted((path.relative_to(common_dir).as_posix(), path) for path in paths):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with open(path, "rb", buffering=0) as file:
            while num_bytes_read := file.readinto(chunk):
                digest.update(chunk_view[:num_bytes_read])
    return digest.hexdigest()


def read_git_state(source_dir) -> tuple[str | None, bool | None]:
    """(HEAD commit hash, whether tracked files differ from it) of the git checkout holding `source_dir`;
    (None, None) outside a checkout or without git. Takes no optional locks, so it never rewrites the
    index of the checkout it reads."""
    git = ["git", "--no-optional-locks", "-C", str(source_dir)]
    try:
        head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
        if head.returncode != 0:
            return None, None
        status = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    return head.stdout.strip(), None if status.returncode != 0 else bool(status.stdout.strip())


def _find_installed_version(distribution_name: str) -> str | None:
    """From the installed metadata: importing torch only to read its version costs seconds."""
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None
