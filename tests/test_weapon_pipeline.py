from types import SimpleNamespace

import numpy as np

from wavetrace.domain.recognition.input_plan import plan_input
from wavetrace.domain.groundtruth.labeler import ScriptedLabeler, weapon_label_function


def test_spans_label_clear_vs_weapon():
    """[] -> class 0, [span] -> class 1."""
    clear = ScriptedLabeler([], label_function=weapon_label_function)
    armed = ScriptedLabeler([(0.0, 10.0, True)], label_function=weapon_label_function)
    assert clear(5.0).class_id == 0
    assert armed(5.0).class_id == 1
    assert armed(5.0).name == "weapon"


def test_serving_plan_ic27_uses_intercarrier():
    """ic27 head -> no gain-lock, intercarrier ON."""
    head = SimpleNamespace(feature_mode="ic27", config=SimpleNamespace(backend="variance"))
    should_apply_lock, intercarrier, pick = plan_input("weapon", head)
    assert should_apply_lock is False
    assert intercarrier is True
    assert pick("F", "I", "IC") == "IC"


def test_serving_plan_fusion_concatenates():
    """fusion head -> gain-lock + intercarrier."""
    head = SimpleNamespace(feature_mode="fusion", config=SimpleNamespace(backend="mlp"))
    should_apply_lock, intercarrier, pick = plan_input("weapon", head)
    assert should_apply_lock is True and intercarrier is True
    out = pick(np.array([1.0, 2.0]), None, np.array([9.0]))
    assert np.allclose(out, [9.0, 1.0, 2.0])
