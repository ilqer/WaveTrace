"""Versioned model runs: each run's files live in <model_dir>/<run_id>/, and the `current` symlink (or the
`CURRENT` file holding the run id, on Windows and wherever symlinks are unavailable) names the run loaders
serve. A run dir is built under a hidden temp name and renamed into place only once complete, and the
pointer moves by one atomic rename, so a crash leaves loaders on the previous run or on none, never on a
partial one. Model dirs trained before runs hold their files flat and still resolve."""

import json
import logging
import os
import re
import secrets
import shutil
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path

from wavetrace.domain.contracts import RunManifest

MODEL_FILE_NAME = "model.joblib"
METRICS_FILE_NAME = "metrics.json"
MANIFEST_FILE_NAME = "run_manifest.json"
HEATMAP_FILE_NAME = "heatmap.joblib"
CURRENT_LINK_NAME = "current"
CURRENT_FILE_NAME = "CURRENT"

# <created UTC, compact ISO 8601>-<6 hex>: sorts by creation second; the suffix keeps same-second runs apart
_RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
_RUN_ID_PATTERN = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{6}")
# Windows cannot rename onto a directory symlink (MoveFileEx refuses a directory target), and on NTFS
# `current` and `CURRENT` are one entry, so there the pointer is always the CURRENT file
_CAN_REPLACE_DIR_SYMLINKS = sys.platform != "win32"
_COPIED_CURRENT_PREFIX = f"{CURRENT_LINK_NAME}.copied-"

_warned_dangling_pointers: set[Path] = set()


def is_run_id(name: str) -> bool:
    return _RUN_ID_PATTERN.fullmatch(name) is not None


def write_model_run(model_dir, manifest: RunManifest, write_files: Callable[[Path], None]) -> Path:
    """Runs `write_files(run_dir)` to fill a new run of `model_dir`, adds `manifest` as run_manifest.json,
    then makes the run current; returns its dir. If `write_files` raises (Ctrl+C included) the partial
    run is deleted and the previous run stays current; a kill leaves only hidden temp entries no loader
    reads. If moving the pointer fails it raises, leaving no temp file, the new run unserved and the
    previous pointer in place, except in the two windows `_point_current_at` names, where it leaves none
    (loaders then serve the flat files, if any). Older runs are kept."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    created_time = time.strftime(_RUN_ID_TIME_FORMAT, time.gmtime(manifest.created_timestamp_s))
    run_id = f"{created_time}-{secrets.token_hex(3)}"
    partial_dir = model_dir / f".{run_id}.partial"
    partial_dir.mkdir()
    try:
        write_files(partial_dir)
        with open(partial_dir / MANIFEST_FILE_NAME, "w") as manifest_file:
            json.dump(manifest.to_dict(), manifest_file, indent=2, sort_keys=True)
        _fsync_files(partial_dir)
        os.rename(partial_dir, model_dir / run_id)
    except BaseException:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    _point_current_at(model_dir, run_id)
    _fsync_dir(model_dir)
    return model_dir / run_id


def find_current_run_dir(model_dir) -> Path | None:
    """The run dir `model_dir`'s pointer names: the `current` symlink's target, else the run id in
    `CURRENT`, with a copied `current` dir served as itself; None without a pointer. O(1): at most two stats
    and one readlink or small read."""
    model_dir = Path(model_dir)
    current_link = model_dir / CURRENT_LINK_NAME
    if current_link.is_symlink():
        return model_dir / os.readlink(current_link)
    if current_link.is_dir():  # a copy that followed the symlink (shutil.copytree, scp, zip) holds the run itself
        return current_link
    try:
        return model_dir / (model_dir / CURRENT_FILE_NAME).read_text().strip()
    except (FileNotFoundError, NotADirectoryError):
        return None


def resolve_run_dir(model_dir, model_file_name: str = MODEL_FILE_NAME) -> Path | None:
    """The dir holding the served `model_file_name`: the current run when it holds that file, else
    `model_dir` itself when the file sits there flat (the layout before runs), else None. A pointer naming
    a missing run is warned about once per process. Read every file of one load from the returned dir, so
    a run swapped in mid-load cannot mix two runs. A retrain that moves a copied `current` dir aside can
    split a load reading it at that moment; that closes once the pointer is a real link or CURRENT file.
    O(1) file system calls."""
    model_dir = Path(model_dir)
    current_run_dir = find_current_run_dir(model_dir)
    if current_run_dir is not None and not current_run_dir.is_dir():
        _warn_dangling_pointer(model_dir, current_run_dir)
    for candidate_dir in (current_run_dir, model_dir):
        if candidate_dir is not None and (candidate_dir / model_file_name).is_file():
            return candidate_dir
    return None


def resolve_model_path(model_path) -> Path:
    """The served file a model path names: `<model_dir>/<file name>`, as written for the flat layout,
    resolves to that file in the current run, so model paths in CLI flags and settings keep working; a
    path into a run dir is itself. Raises FileNotFoundError when no layout holds the file. O(1)."""
    model_path = Path(model_path)
    run_dir = resolve_run_dir(model_path.parent, model_path.name)
    if run_dir is None:
        raise FileNotFoundError(f"no model at {model_path}: neither a current run nor a flat file holds it")
    return run_dir / model_path.name


def find_model_dir(run_or_model_dir) -> Path:
    """The model dir a run dir belongs to (a run id, a copied `current`, or one moved aside), else the dir
    itself (flat)."""
    run_or_model_dir = Path(run_or_model_dir)
    name = run_or_model_dir.name
    if is_run_id(name) or name == CURRENT_LINK_NAME or name.startswith(_COPIED_CURRENT_PREFIX):
        return run_or_model_dir.parent
    return run_or_model_dir


def _warn_dangling_pointer(model_dir: Path, missing_run_dir: Path) -> None:
    warning_key = missing_run_dir.resolve()  # one key however the model dir was spelled or linked
    if warning_key in _warned_dangling_pointers:
        return
    _warned_dangling_pointers.add(warning_key)
    current_link = model_dir / CURRENT_LINK_NAME
    pointer_path = current_link if current_link.is_symlink() else model_dir / CURRENT_FILE_NAME
    logging.getLogger(__name__).warning(
        "%s names the run %s, which does not exist; serving %s's flat files, if any",
        pointer_path, missing_run_dir, model_dir)


def _point_current_at(model_dir: Path, run_id: str) -> None:
    """The new pointer is complete before the old one changes, and a stale pointer of the other kind goes
    only once it is in place, so a crash leaves a pointer on a complete run except in two windows with
    none: between moving a copied `current` dir aside and the final rename, on every platform; and on
    Windows, retraining a model dir that carries a `current` symlink, between removing it and the replace."""
    if _CAN_REPLACE_DIR_SYMLINKS and _swap_current_link(model_dir, run_id):
        _remove_if_kind(model_dir / CURRENT_FILE_NAME, stat.S_ISREG)
    else:
        _write_current_file(model_dir, run_id)


def _move_copied_current_aside(model_dir: Path, run_id: str) -> None:
    """A real `current` dir is a copied run, and no link or file can replace a dir, so it moves to
    `current.copied-<new run_id>`, kept like any older run, never deleted."""
    current_path = model_dir / CURRENT_LINK_NAME
    try:
        is_copied_dir = stat.S_ISDIR(os.lstat(current_path).st_mode)
    except FileNotFoundError:
        return
    if is_copied_dir:
        os.rename(current_path, model_dir / f"{_COPIED_CURRENT_PREFIX}{run_id}")


def _swap_current_link(model_dir: Path, run_id: str) -> bool:
    """False where symlinks cannot be made, e.g. on file systems without them."""
    temporary_link = model_dir / f".{CURRENT_LINK_NAME}.{run_id}"
    try:
        os.symlink(run_id, temporary_link, target_is_directory=True)  # relative: survives moving the data root
        _move_copied_current_aside(model_dir, run_id)
        os.replace(temporary_link, model_dir / CURRENT_LINK_NAME)
    except (OSError, NotImplementedError):
        _remove_if_present(temporary_link)
        return False
    except BaseException:
        _remove_if_present(temporary_link)
        raise
    return True


def _write_current_file(model_dir: Path, run_id: str) -> None:
    """A stale `current` link outranks `CURRENT`, so it goes once `CURRENT` is in place; a crash in between
    leaves it on the previous run. On Windows it goes first: NTFS ignores case, so it is the very entry
    `CURRENT` names, and a directory link refuses replacement."""
    current_link = model_dir / CURRENT_LINK_NAME
    temporary_path = model_dir / f".{CURRENT_FILE_NAME}.{run_id}"
    try:
        with open(temporary_path, "w") as current_file:
            current_file.write(f"{run_id}\n")
            current_file.flush()
            os.fsync(current_file.fileno())
        _move_copied_current_aside(model_dir, run_id)
        if not _CAN_REPLACE_DIR_SYMLINKS:
            _remove_if_kind(current_link, stat.S_ISLNK)
        os.replace(temporary_path, model_dir / CURRENT_FILE_NAME)  # atomic on Windows too
    except BaseException:
        _remove_if_present(temporary_path)
        raise
    _remove_if_kind(current_link, stat.S_ISLNK)  # on macOS the replace already took the link's entry


def _remove_if_present(path: Path) -> None:
    try:
        os.unlink(path)  # on Windows this removes a directory symlink too
    except FileNotFoundError:
        pass


def _remove_if_kind(path: Path, is_kind: Callable[[int], bool]) -> None:
    """Removes `path` only when `is_kind(lstat mode)`: on a case-insensitive file system (macOS, Windows)
    `current` and `CURRENT` are one entry, and it may be the pointer just written."""
    try:
        if is_kind(os.lstat(path).st_mode):
            os.unlink(path)  # on Windows this removes a directory symlink too
    except FileNotFoundError:
        pass


def _fsync_files(directory: Path) -> None:
    """Flushes the run's files to disk before the rename publishes them, so a power cut cannot leave a
    published run with empty files."""
    for path in directory.iterdir():
        if path.is_file():
            file_descriptor = os.open(path, os.O_RDWR)  # Windows can flush only a handle open for writing
            try:
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)


def _fsync_dir(directory: Path) -> None:
    """Persists the renames; POSIX only, as Windows cannot open a directory."""
    if os.name != "posix":
        return
    file_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
