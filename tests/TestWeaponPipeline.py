"""Weapon pipeline: span-based labeling + ic27 plan."""

from types import SimpleNamespace

import numpy as np

from wavetrace.Cli import _serving_plan
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler, weapon_label_fn


def test_spans_label_clear_vs_weapon():
    """[] -> class 0, [span] -> class 1."""
    clear = ScriptedLabeler([], label_fn=weapon_label_fn)
    armed = ScriptedLabeler([(0.0, 10.0, True)], label_fn=weapon_label_fn)
    assert clear(5.0).class_id == 0
    assert armed(5.0).class_id == 1
    assert armed(5.0).name == "weapon"


def test_serving_plan_ic27_uses_intercarrier():
    """ic27 head -> no gain-lock, intercarrier ON."""
    head = SimpleNamespace(feature_mode="ic27", config=SimpleNamespace(backend="variance"))
    apply_lock, intercarrier, pick = _serving_plan("weapon", head)
    assert apply_lock is False
    assert intercarrier is True
    assert pick("F", "I", "IC") == "IC"


def test_serving_plan_fusion_concatenates():
    """fusion head -> gain-lock + intercarrier."""
    head = SimpleNamespace(feature_mode="fusion", config=SimpleNamespace(backend="mlp"))
    apply_lock, intercarrier, pick = _serving_plan("weapon", head)
    assert apply_lock is True and intercarrier is True
    out = pick(np.array([1.0, 2.0]), None, np.array([9.0]))
    assert np.allclose(out, [9.0, 1.0, 2.0])
