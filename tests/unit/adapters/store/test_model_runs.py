"""Versioned model runs: a run is written whole or not at all, the pointer swap is atomic, and loaders
resolve the `current` symlink, then the `CURRENT` file, then the flat layout every older model uses."""

import logging
import os
import shutil
import signal
import subprocess
import sys
import textwrap

import pytest

from wavetrace.adapters.store import model_runs
from wavetrace.adapters.store.model_runs import (
    CURRENT_FILE_NAME,
    CURRENT_LINK_NAME,
    MANIFEST_FILE_NAME,
    MODEL_FILE_NAME,
    find_current_run_dir,
    find_model_dir,
    is_run_id,
    resolve_model_path,
    resolve_run_dir,
    write_model_run,
)
from wavetrace.domain.contracts import SCHEMA_VERSION, LibraryVersions, PipelineContract, RunManifest

_CREATED_TIMESTAMP_S = 1793028612  # 2026-10-26T15:30:12Z


def _manifest(created_timestamp_s=_CREATED_TIMESTAMP_S):
    return RunManifest(
        origin="trained", git_commit=None, has_uncommitted_changes=None, config_hash="c" * 64,
        dataset_hash="d" * 64, seed=0,
        library_versions=LibraryVersions(python="3.14.0", numpy="2.0.0", scikit_learn="1.5.0", wavetrace=None),
        contract=PipelineContract(target_sample_rate_hz=100.0, window_frames=128, hop_frames=32),
        created_timestamp_s=created_timestamp_s, schema_version=SCHEMA_VERSION)


def _write_model(content):
    def write_files(run_dir):
        (run_dir / MODEL_FILE_NAME).write_text(content)
        (run_dir / "metrics.json").write_text(f'{{"of": "{content}"}}')
    return write_files


def _served_model(model_dir):
    run_dir = resolve_run_dir(model_dir)
    return None if run_dir is None else (run_dir / MODEL_FILE_NAME).read_text()


def _hidden_entries(model_dir):
    return [path.name for path in model_dir.iterdir() if path.name.startswith(".")]


def test_a_run_lands_in_its_own_run_dir_with_its_manifest_and_becomes_current(tmp_path):
    run_dir = write_model_run(tmp_path / "node1", _manifest(), _write_model("first"))

    assert run_dir.parent == tmp_path / "node1"
    assert run_dir.name.startswith("20261026T153012Z-") and is_run_id(run_dir.name)
    assert sorted(path.name for path in run_dir.iterdir()) == ["metrics.json", MODEL_FILE_NAME, MANIFEST_FILE_NAME]
    assert (tmp_path / "node1" / CURRENT_LINK_NAME).is_symlink()
    assert find_current_run_dir(tmp_path / "node1") == run_dir
    assert resolve_run_dir(tmp_path / "node1") == run_dir


def test_a_later_run_replaces_current_and_keeps_every_older_run(tmp_path):
    first = write_model_run(tmp_path, _manifest(), _write_model("first"))
    second = write_model_run(tmp_path, _manifest(_CREATED_TIMESTAMP_S + 48), _write_model("second"))

    assert _served_model(tmp_path) == "second"
    assert first.is_dir() and sorted([first.name, second.name]) == [first.name, second.name]
    assert sorted(path.name for path in tmp_path.iterdir()) == [first.name, second.name, CURRENT_LINK_NAME]


def test_an_exception_while_writing_leaves_the_previous_run_current_and_no_partial_run(tmp_path):
    write_model_run(tmp_path, _manifest(), _write_model("first"))

    def crash_between_files(run_dir):
        (run_dir / MODEL_FILE_NAME).write_text("half-written")
        raise KeyboardInterrupt  # Ctrl+C after model.joblib, before metrics.json

    with pytest.raises(KeyboardInterrupt):
        write_model_run(tmp_path, _manifest(), crash_between_files)

    assert _served_model(tmp_path) == "first"
    assert len([path for path in tmp_path.iterdir() if is_run_id(path.name)]) == 1
    assert not _hidden_entries(tmp_path)


def test_an_interrupt_before_the_pointer_swap_leaves_the_previous_run_current(tmp_path, monkeypatch):
    write_model_run(tmp_path, _manifest(), _write_model("first"))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_model_run(tmp_path, _manifest(), _write_model("second"))
    monkeypatch.undo()

    assert _served_model(tmp_path) == "first"
    assert not _hidden_entries(tmp_path)  # the temporary link is gone too


def test_a_pointer_swap_that_fails_raises_and_leaves_the_previous_run_current(tmp_path, monkeypatch):
    write_model_run(tmp_path, _manifest(), _write_model("first"))

    def rename_fails(*args, **kwargs):
        raise OSError("input/output error")

    monkeypatch.setattr(os, "replace", rename_fails)
    with pytest.raises(OSError, match="input/output error"):
        write_model_run(tmp_path, _manifest(), _write_model("second"))
    monkeypatch.undo()

    assert _served_model(tmp_path) == "first"
    assert not _hidden_entries(tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX")
def test_a_process_killed_mid_write_leaves_loaders_on_the_previous_run(tmp_path):
    model_dir = tmp_path / "node1"
    write_model_run(model_dir, _manifest(), _write_model("first"))
    killed_writer = textwrap.dedent("""
        import sys, time
        from pathlib import Path
        from wavetrace.adapters.store.model_runs import MODEL_FILE_NAME, write_model_run
        from wavetrace.domain.contracts import LibraryVersions, PipelineContract, RunManifest

        def write_files(run_dir):
            (run_dir / MODEL_FILE_NAME).write_text("half-written")
            print("model written", flush=True)
            time.sleep(60)

        manifest = RunManifest(
            origin="trained", git_commit=None, has_uncommitted_changes=None, config_hash="c", dataset_hash="d",
            seed=0, library_versions=LibraryVersions(python="3", numpy="2", scikit_learn="1", wavetrace=None),
            contract=PipelineContract(target_sample_rate_hz=100.0, window_frames=128, hop_frames=32),
            created_timestamp_s=1793028660, schema_version=1)
        write_model_run(Path(sys.argv[1]), manifest, write_files)
    """)
    writer = subprocess.Popen([sys.executable, "-c", killed_writer, str(model_dir)], stdout=subprocess.PIPE, text=True)
    try:
        assert writer.stdout.readline().strip() == "model written"
        writer.send_signal(signal.SIGKILL)
    finally:
        writer.wait(timeout=30)

    assert writer.returncode == -signal.SIGKILL
    assert _served_model(model_dir) == "first"
    assert len([path for path in model_dir.iterdir() if is_run_id(path.name)]) == 1


def test_without_symlinks_the_current_file_names_the_run(tmp_path, monkeypatch):
    def symlinks_unsupported(*args, **kwargs):
        raise OSError("symbolic link privilege not held")  # Windows without Developer Mode

    write_model_run(tmp_path, _manifest(), _write_model("first"))
    monkeypatch.setattr(os, "symlink", symlinks_unsupported)
    second = write_model_run(tmp_path, _manifest(), _write_model("second"))

    assert (tmp_path / CURRENT_FILE_NAME).read_text().strip() == second.name
    assert not (tmp_path / CURRENT_LINK_NAME).is_symlink()  # a stale link would outrank the file
    assert _served_model(tmp_path) == "second"


def _windows_replace(real_replace):
    """os.replace as on NTFS: names match whatever their case, and a directory link refuses replacement."""
    def replace(source, destination):
        destination = os.fspath(destination)
        directory, name = os.path.split(destination)
        for entry in os.scandir(directory):
            if entry.name.casefold() == name.casefold() and entry.is_symlink() and entry.is_dir():
                raise PermissionError(13, "Access is denied", destination)
        real_replace(source, destination)
    return replace


def test_on_windows_the_current_file_replaces_a_link_and_then_names_each_new_run(tmp_path, monkeypatch):
    write_model_run(tmp_path, _manifest(), _write_model("first"))
    assert (tmp_path / CURRENT_LINK_NAME).is_symlink()  # e.g. a model dir trained on Linux, copied with its link

    monkeypatch.setattr(model_runs, "_CAN_REPLACE_DIR_SYMLINKS", False)
    monkeypatch.setattr(os, "replace", _windows_replace(os.replace))
    second = write_model_run(tmp_path, _manifest(), _write_model("second"))
    assert not (tmp_path / CURRENT_LINK_NAME).is_symlink()
    assert (tmp_path / CURRENT_FILE_NAME).read_text().strip() == second.name
    assert _served_model(tmp_path) == "second"

    third = write_model_run(tmp_path, _manifest(), _write_model("third"))
    assert (tmp_path / CURRENT_FILE_NAME).read_text().strip() == third.name
    assert _served_model(tmp_path) == "third"
    assert not _hidden_entries(tmp_path)


def test_a_copied_current_dir_is_served_as_the_run_it_holds(tmp_path):
    copied_run = tmp_path / CURRENT_LINK_NAME
    copied_run.mkdir()
    (copied_run / MODEL_FILE_NAME).write_text("copied")  # e.g. shutil.copytree, which follows symlinks
    assert _served_model(tmp_path) == "copied"


def test_retraining_a_copied_model_dir_moves_its_current_copy_aside_and_keeps_it(tmp_path):
    write_model_run(tmp_path / "original", _manifest(), _write_model("first"))
    shutil.copytree(tmp_path / "original", tmp_path / "copy")  # follows `current`: a real dir in the copy
    copied_model_dir = tmp_path / "copy"
    assert not (copied_model_dir / CURRENT_LINK_NAME).is_symlink()

    second = write_model_run(copied_model_dir, _manifest(), _write_model("second"))

    moved_copy = copied_model_dir / f"{CURRENT_LINK_NAME}.copied-{second.name}"
    assert (moved_copy / MODEL_FILE_NAME).read_text() == "first"
    assert (copied_model_dir / CURRENT_LINK_NAME).is_symlink()
    assert _served_model(copied_model_dir) == "second"
    assert find_model_dir(moved_copy) == copied_model_dir
    assert not _hidden_entries(copied_model_dir)


def test_a_pointer_to_a_missing_run_warns_once_naming_both_and_serves_the_flat_files(tmp_path, caplog):
    (tmp_path / MODEL_FILE_NAME).write_text("legacy")
    run_dir = write_model_run(tmp_path, _manifest(), _write_model("deleted"))
    shutil.rmtree(run_dir)

    with caplog.at_level(logging.WARNING):
        assert _served_model(tmp_path) == "legacy"
        assert _served_model(tmp_path) == "legacy"

    [record] = caplog.records
    assert str(tmp_path / CURRENT_LINK_NAME) in record.getMessage() and str(run_dir) in record.getMessage()


def test_a_flat_model_dir_from_before_runs_is_served_from_its_own_dir(tmp_path):
    (tmp_path / MODEL_FILE_NAME).write_text("legacy")
    assert resolve_run_dir(tmp_path) == tmp_path


def test_a_current_run_outranks_flat_files_left_from_before_runs(tmp_path):
    (tmp_path / MODEL_FILE_NAME).write_text("legacy")
    write_model_run(tmp_path, _manifest(), _write_model("retrained"))
    assert _served_model(tmp_path) == "retrained"


def test_a_dir_with_no_model_resolves_to_none(tmp_path):
    assert resolve_run_dir(tmp_path) is None
    assert resolve_run_dir(tmp_path / "missing") is None
    assert find_current_run_dir(tmp_path) is None


def test_another_artifact_resolves_by_its_own_file_name(tmp_path):
    def write_heatmap(run_dir):
        (run_dir / "heatmap.joblib").write_text("heatmap")

    run_dir = write_model_run(tmp_path, _manifest(), write_heatmap)
    assert resolve_run_dir(tmp_path, "heatmap.joblib") == run_dir
    assert resolve_run_dir(tmp_path) is None


def test_a_flat_model_path_resolves_to_the_current_runs_file(tmp_path):
    run_dir = write_model_run(tmp_path, _manifest(), _write_model("first"))
    assert resolve_model_path(tmp_path / MODEL_FILE_NAME) == run_dir / MODEL_FILE_NAME
    assert resolve_model_path(run_dir / MODEL_FILE_NAME) == run_dir / MODEL_FILE_NAME


def test_a_model_path_neither_layout_holds_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="no model"):
        resolve_model_path(tmp_path / MODEL_FILE_NAME)


def test_only_run_dir_names_are_run_ids():
    assert is_run_id("20261026T153012Z-a1b2c3")
    assert not is_run_id("node1") and not is_run_id(CURRENT_LINK_NAME) and not is_run_id("20261026T153012Z")


def test_the_model_dir_of_a_run_dir_is_its_parent_and_of_a_flat_dir_itself(tmp_path):
    assert find_model_dir(tmp_path / "node1" / "20261026T153012Z-a1b2c3") == tmp_path / "node1"
    assert find_model_dir(tmp_path / "node1" / CURRENT_LINK_NAME) == tmp_path / "node1"
    assert find_model_dir(tmp_path / "node1") == tmp_path / "node1"


def test_a_copied_current_dir_is_served_as_itself_so_a_file_dropped_into_it_wins(tmp_path):
    write_model_run(tmp_path / "original", _manifest(), _write_model("first"))
    shutil.copytree(tmp_path / "original", tmp_path / "copy")
    copied_current_dir = tmp_path / "copy" / CURRENT_LINK_NAME
    (copied_current_dir / MODEL_FILE_NAME).write_text("dropped in")

    served_dir = resolve_run_dir(tmp_path / "copy")

    assert served_dir == copied_current_dir
    assert (served_dir / MODEL_FILE_NAME).read_text() == "dropped in"


def test_an_interrupt_before_the_new_link_exists_leaves_a_copied_current_dir_in_place(tmp_path, monkeypatch):
    write_model_run(tmp_path / "original", _manifest(), _write_model("first"))
    shutil.copytree(tmp_path / "original", tmp_path / "copy")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "symlink", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_model_run(tmp_path / "copy", _manifest(), _write_model("second"))
    monkeypatch.undo()

    assert (tmp_path / "copy" / CURRENT_LINK_NAME).is_dir()
    assert not (tmp_path / "copy" / CURRENT_LINK_NAME).is_symlink()
    assert _served_model(tmp_path / "copy") == "first"
    assert not _hidden_entries(tmp_path / "copy")


def test_a_missing_run_warns_once_however_the_model_dir_is_spelled_or_linked(tmp_path, caplog):
    model_dir = tmp_path / "node1"
    shutil.rmtree(write_model_run(model_dir, _manifest(), _write_model("deleted")))
    (tmp_path / "alias").symlink_to(model_dir, target_is_directory=True)

    with caplog.at_level(logging.WARNING):
        for spelling in (model_dir, tmp_path / "node1" / ".." / "node1", tmp_path / "alias"):
            assert resolve_run_dir(spelling) is None

    assert len(caplog.records) == 1
