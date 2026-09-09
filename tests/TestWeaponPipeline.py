"""Weapon pipeline: span-based labeling + ic27 plan."""

from types import SimpleNamespace

import numpy as np

from wavetrace.recognition import planInferenceInput
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler, weaponLabelFn


def test_spans_label_clear_vs_weapon():
    """[] -> class 0, [span] -> class 1."""
    clear = ScriptedLabeler([], label_fn=weaponLabelFn)
    armed = ScriptedLabeler([(0.0, 10.0, True)], label_fn=weaponLabelFn)
    assert clear(5.0).class_id == 0
    assert armed(5.0).class_id == 1
    assert armed(5.0).name == "weapon"


def test_serving_plan_ic27_uses_intercarrier():
    """ic27 head -> no gain-lock, intercarrier ON."""
    head = SimpleNamespace(feature_mode="ic27", config=SimpleNamespace(backend="variance"))
    applyLock, intercarrier, pick = planInferenceInput("weapon", head)
    assert applyLock is False
    assert intercarrier is True
    assert pick("F", "I", "IC") == "IC"


def test_serving_plan_fusion_concatenates():
    """fusion head -> gain-lock + intercarrier."""
    head = SimpleNamespace(feature_mode="fusion", config=SimpleNamespace(backend="mlp"))
    applyLock, intercarrier, pick = planInferenceInput("weapon", head)
    assert applyLock is True and intercarrier is True
    out = pick(np.array([1.0, 2.0]), None, np.array([9.0]))
    assert np.allclose(out, [9.0, 1.0, 2.0])
