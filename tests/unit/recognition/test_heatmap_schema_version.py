"""The heatmap head saves `schema_version` and loads it by the head rule: equal loads, missing loads as
legacy version 1 with one warning per file, any other value fails naming the file and how to retrain."""

import logging

import joblib
import numpy as np
import pytest

from wavetrace.adapters.recognition.heads import SchemaVersionMismatchError
from wavetrace.adapters.store.model_runs import HEATMAP_FILE_NAME
from wavetrace.domain.contracts import SCHEMA_VERSION
from wavetrace.domain.model_config import ModelConfig

pytest.importorskip("torch")
from wavetrace.recognition.heatmap import HeatmapHead  # noqa: E402  (needs torch)


def _saved_heatmap(path, schema_version):
    rng = np.random.default_rng(0)
    head = HeatmapHead(ModelConfig(stage="presence", k=4), grid=2)
    head.fit(rng.normal(0, 1, (6, 1, 4, 8)), rng.integers(0, 2, (6, 2, 2)), epochs=1)
    head.save(path)
    blob = joblib.load(path)
    assert blob["schema_version"] == SCHEMA_VERSION
    if schema_version is None:
        del blob["schema_version"]
    else:
        blob["schema_version"] = schema_version
    joblib.dump(blob, path)
    return path


def test_the_current_schema_version_loads_without_a_warning(tmp_path, caplog):
    path = _saved_heatmap(tmp_path / HEATMAP_FILE_NAME, SCHEMA_VERSION)
    with caplog.at_level(logging.WARNING):
        assert HeatmapHead.load(path).grid == 2
    assert not caplog.records


def test_a_heatmap_without_schema_version_loads_as_legacy_with_one_warning(tmp_path, caplog):
    path = _saved_heatmap(tmp_path / HEATMAP_FILE_NAME, None)
    with caplog.at_level(logging.WARNING):
        assert HeatmapHead.load(path).grid == 2
        assert HeatmapHead.load(path).grid == 2
    [record] = caplog.records
    assert str(path) in record.getMessage() and "legacy" in record.getMessage()


def test_another_schema_version_fails_naming_the_file_and_the_model_root_to_retrain(tmp_path):
    path = _saved_heatmap(tmp_path / "model_camera" / HEATMAP_FILE_NAME, SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionMismatchError) as error:
        HeatmapHead.load(path)
    message = str(error.value)
    assert str(path) in message and f"schema_version {SCHEMA_VERSION + 1}" in message
    assert f"into {tmp_path / 'model_camera'}" in message
