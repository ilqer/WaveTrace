"""Per-node vote weight wiring: _logo_accuracy reads each node's honest LOGO balanced accuracy."""

import json

import pytest

from run_live_mesh import _logo_accuracy


def _write(tmp_path, payload):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_prefers_session_axis(tmp_path):
    """Session LOGO accuracy wins over subject when both present."""
    path = _write(tmp_path, {"logo": {"session": {"accuracy": 0.9}, "subject": {"accuracy": 0.6}}})
    assert _logo_accuracy(path) == pytest.approx(0.9)


def test_falls_back_to_subject(tmp_path):
    """No session axis -> use the subject LOGO accuracy."""
    path = _write(tmp_path, {"logo": {"subject": {"accuracy": 0.7}}})
    assert _logo_accuracy(path) == pytest.approx(0.7)


def test_none_when_no_logo(tmp_path):
    """A model trained without a foldable group has no honest number -> None (neutral weight 1.0)."""
    assert _logo_accuracy(_write(tmp_path, {"logo": {}})) is None
    assert _logo_accuracy(_write(tmp_path, {})) is None


def test_none_when_file_missing_or_bad(tmp_path):
    """Missing/corrupt metrics.json must not crash serving -> None."""
    assert _logo_accuracy(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert _logo_accuracy(str(bad)) is None
