from types import SimpleNamespace

import numpy as np
import pytest

from wavetrace.domain.recognition.input_plan import plan_input

FEATURES = np.arange(3.0)
IMAGE = np.arange(12.0).reshape(3, 4)
INTERCARRIER = np.array([7.0, 8.0])


def test_presence_takes_plain_features_and_applies_lock():
    should_apply_lock, intercarrier, pick = plan_input("presence", SimpleNamespace())
    assert (should_apply_lock, intercarrier) == (True, False)
    assert pick(FEATURES, IMAGE, INTERCARRIER) is FEATURES


@pytest.mark.parametrize(
    "feature_mode, expected_flags, expected_row",
    [
        ("cnn", (False, False), IMAGE.reshape(-1)),
        ("fusion", (True, True), np.hstack([INTERCARRIER, FEATURES])),
        ("ic27", (False, True), INTERCARRIER),
    ],
)
def test_weapon_plan_follows_feature_mode(feature_mode, expected_flags, expected_row):
    head = SimpleNamespace(feature_mode=feature_mode, default_feature_mode="unused")
    should_apply_lock, intercarrier, pick = plan_input("weapon", head)
    assert (should_apply_lock, intercarrier) == expected_flags
    np.testing.assert_array_equal(pick(FEATURES, IMAGE, INTERCARRIER), expected_row)


def test_weapon_head_without_recorded_feature_mode_uses_default():
    head = SimpleNamespace(feature_mode=None, default_feature_mode="ic27")
    _, intercarrier, pick = plan_input("weapon", head)
    assert intercarrier is True
    assert pick(FEATURES, IMAGE, INTERCARRIER) is INTERCARRIER

