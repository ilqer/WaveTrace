"""Head loaders enforce the saved `schema_version`: equal loads, missing loads as legacy version 1 with a
warning once per file, any other value fails naming the file, both versions and the retrain command. They
also follow a flat `<model_dir>/model.joblib` path to the current run."""

import logging
import shutil

import joblib
import numpy as np
import pytest

from wavetrace.adapters.recognition.heads import (
    SchemaVersionMismatchError,
    build_presence_head,
    build_weapon_head,
    load_presence_head,
    load_weapon_head,
)
from wavetrace.adapters.store.model_runs import MODEL_FILE_NAME, write_model_run
from wavetrace.adapters.store.run_manifest import build_run_manifest
from wavetrace.domain.contracts import SCHEMA_VERSION
from wavetrace.domain.model_config import ModelConfig


def _fitted_presence_head():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 18)).astype(np.float32)
    y = (np.arange(40) % 2).astype(np.int64)
    X[y == 1] += 3.0
    return build_presence_head(ModelConfig(stage="presence", k=2)).fit(X, y)


def _fitted_weapon_head():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 27)).astype(np.float32)
    y = (np.arange(40) % 2).astype(np.int64)
    X[y == 1, 9] -= 3.0
    return build_weapon_head(ModelConfig(stage="weapon", k=12, backend="variance")).fit(X, y)


HEADS = [pytest.param(_fitted_presence_head, load_presence_head, "presence", id="presence"),
         pytest.param(_fitted_weapon_head, load_weapon_head, "weapon", id="weapon")]


def _save_with_schema_version(head, path, schema_version):
    head.save(path)
    blob = joblib.load(path)
    if schema_version is None:
        del blob["schema_version"]
    else:
        blob["schema_version"] = schema_version
    joblib.dump(blob, path)
    return path


@pytest.mark.parametrize("fit_head, load_head, stage", HEADS)
def test_the_current_schema_version_loads_without_a_warning(fit_head, load_head, stage, tmp_path, caplog):
    path = fit_head().save(tmp_path / MODEL_FILE_NAME)
    with caplog.at_level(logging.WARNING):
        assert load_head(path).config.stage == stage
    assert not caplog.records


@pytest.mark.parametrize("fit_head, load_head, stage", HEADS)
def test_a_missing_schema_version_loads_as_legacy_version_1_with_a_warning(fit_head, load_head, stage, tmp_path,
                                                                            caplog):
    path = _save_with_schema_version(fit_head(), tmp_path / MODEL_FILE_NAME, None)
    with caplog.at_level(logging.WARNING):
        assert load_head(path).config.stage == stage
    [record] = caplog.records
    assert record.levelno == logging.WARNING and str(path) in record.getMessage()
    assert "legacy" in record.getMessage() and "1" in record.getMessage()


@pytest.mark.parametrize("fit_head, load_head, stage", HEADS)
def test_the_legacy_warning_fires_once_per_file_however_often_it_loads(fit_head, load_head, stage, tmp_path,
                                                                       caplog):
    head = fit_head()
    first_path = _save_with_schema_version(head, tmp_path / "node1" / MODEL_FILE_NAME, None)
    second_path = _save_with_schema_version(head, tmp_path / "node2" / MODEL_FILE_NAME, None)
    with caplog.at_level(logging.WARNING):
        for path in (first_path, first_path, second_path, first_path):
            load_head(path)
    assert [str(first_path) in record.getMessage() for record in caplog.records] == [True, False]


@pytest.mark.parametrize("fit_head, load_head, stage", HEADS)
def test_another_schema_version_fails_naming_file_versions_and_the_retrain_command(fit_head, load_head, stage,
                                                                                    tmp_path):
    found_version = SCHEMA_VERSION + 1
    path = _save_with_schema_version(fit_head(), tmp_path / "node1" / MODEL_FILE_NAME, found_version)
    with pytest.raises(SchemaVersionMismatchError) as error:
        load_head(path)
    message = str(error.value)
    assert str(path) in message
    assert f"schema_version {found_version}" in message and f"expects {SCHEMA_VERSION}" in message
    assert f"wavetrace train <dataset dirs> --stage {stage} --out {tmp_path / 'node1'}" in message


def test_a_flat_model_path_loads_the_current_run_and_retrains_into_its_model_dir(tmp_path):
    head = _fitted_presence_head()
    model_dir = tmp_path / "node1"
    manifest = build_run_manifest(head.config, head.contract, [])
    run_dir = write_model_run(model_dir, manifest, lambda run_dir: head.save(run_dir / MODEL_FILE_NAME))
    _save_with_schema_version(head, run_dir / MODEL_FILE_NAME, SCHEMA_VERSION + 1)

    with pytest.raises(SchemaVersionMismatchError) as error:
        load_presence_head(model_dir / MODEL_FILE_NAME)
    assert str(run_dir / MODEL_FILE_NAME) in str(error.value)  # the file actually read
    assert f"--out {model_dir}" in str(error.value)            # the model dir, not the run dir


def test_a_copied_model_dir_retrains_into_the_copy_not_its_current_dir(tmp_path):
    head = _fitted_presence_head()
    manifest = build_run_manifest(head.config, head.contract, [])
    run_dir = write_model_run(tmp_path / "node1", manifest, lambda run_dir: head.save(run_dir / MODEL_FILE_NAME))
    _save_with_schema_version(head, run_dir / MODEL_FILE_NAME, SCHEMA_VERSION + 1)
    shutil.copytree(tmp_path / "node1", tmp_path / "copy")  # `current` becomes a real dir in the copy

    with pytest.raises(SchemaVersionMismatchError) as error:
        load_presence_head(tmp_path / "copy" / MODEL_FILE_NAME)
    assert str(error.value).endswith(f"--out {tmp_path / 'copy'}")


def test_the_legacy_warning_fires_once_however_the_path_is_spelled_or_linked(tmp_path, caplog):
    path = _save_with_schema_version(_fitted_presence_head(), tmp_path / "node1" / MODEL_FILE_NAME, None)
    (tmp_path / "alias").symlink_to(tmp_path / "node1", target_is_directory=True)
    with caplog.at_level(logging.WARNING):
        for spelling in (path, tmp_path / "node1" / ".." / "node1" / MODEL_FILE_NAME,
                         tmp_path / "alias" / MODEL_FILE_NAME):
            load_presence_head(spelling)
    assert len(caplog.records) == 1
