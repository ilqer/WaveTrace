"""load_logo_accuracy reads each node's LOGO balanced accuracy."""

import json

import pytest

from wavetrace.recognition.serving import load_logo_accuracy


def _write_metrics(tmp_path, payload):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_prefers_session_axis(tmp_path):
    path = _write_metrics(tmp_path, {"logo": {"session": {"accuracy": 0.9}, "subject": {"accuracy": 0.6}}})
    assert load_logo_accuracy(path) == pytest.approx(0.9)


def test_falls_back_to_subject(tmp_path):
    path = _write_metrics(tmp_path, {"logo": {"subject": {"accuracy": 0.7}}})
    assert load_logo_accuracy(path) == pytest.approx(0.7)


def test_none_when_no_logo(tmp_path):
    """A model trained without foldable groups has no validation accuracy: None (neutral weight 1.0)."""
    assert load_logo_accuracy(_write_metrics(tmp_path, {"logo": {}})) is None
    assert load_logo_accuracy(_write_metrics(tmp_path, {})) is None


def test_none_when_file_missing_or_bad(tmp_path):
    """Missing or corrupt metrics.json returns None to avoid crashing serving."""
    assert load_logo_accuracy(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_logo_accuracy(str(bad)) is None
