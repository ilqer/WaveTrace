from types import SimpleNamespace

import pytest

from wavetrace.domain.rules import (
    BusinessRuleViolationError, HeadsMustShareClassOrdering, HeadsMustShareSampleRate, LinkMustHaveServingHead,
    SampleRateMustBeWithinTolerance, SampleRateMustKeepPaceWithPeers, SubcarrierWidthMustCoverCalibration,
    SubcarrierWidthMustMatchCalibration, SubcarrierWidthMustMatchContract, SubcarrierWidthMustStayConstantPerLink,
    measure_delivered_rate_hz, require,
)


def test_width_must_cover_every_subcarrier_the_calibration_indexes():
    calibration = SimpleNamespace(subcarriers=[1, 2, 3, 4], image_subcarriers=[1, 2, 3, 4, 9])
    rule = SubcarrierWidthMustCoverCalibration.for_calibration(calibration)
    assert rule.required_width == 10
    assert rule.is_satisfied(10)
    assert not rule.is_satisfied(9)


def test_width_must_match_the_trained_capture_width():
    rule = SubcarrierWidthMustMatchContract(192)
    assert rule.is_satisfied(192)
    assert not rule.is_satisfied(128)


def test_a_legacy_contract_without_a_width_accepts_every_width():
    rule = SubcarrierWidthMustMatchContract(None)
    assert rule.is_satisfied(64) and rule.is_satisfied(192)


def test_a_link_keeps_its_pinned_width():
    rule = SubcarrierWidthMustStayConstantPerLink(128)
    assert rule.is_satisfied(128)
    assert not rule.is_satisfied(192)


def test_a_link_without_a_head_breaks_the_serving_head_rule():
    rule = LinkMustHaveServingHead()
    assert rule.is_satisfied(object())
    assert not rule.is_satisfied(None)


def test_heads_must_share_one_class_ordering():
    rule = HeadsMustShareClassOrdering()
    assert rule.is_satisfied([[0, 1], [0, 1]])
    assert rule.is_satisfied([])
    assert not rule.is_satisfied([[0, 1], [1, 0]])


def test_sample_rate_may_fall_short_by_at_most_the_tolerance():
    rule = SampleRateMustBeWithinTolerance(target_sample_rate_hz=100.0, min_sample_rate_fraction=0.8)
    assert rule.is_satisfied(80.0)
    assert rule.is_satisfied(209.5)  # a faster stream is decimated by the resampler
    assert not rule.is_satisfied(79.9)


def test_the_minimum_sample_rate_is_the_exact_fraction_of_the_target():
    assert SampleRateMustBeWithinTolerance(100.0, 0.8).min_sample_rate_hz == 100.0 * 0.8
    assert SampleRateMustBeWithinTolerance(100.0, 0.1).min_sample_rate_hz == 10.0


def test_heads_must_share_one_target_sample_rate():
    rule = HeadsMustShareSampleRate()
    assert rule.is_satisfied([100.0, 100.0])
    assert rule.is_satisfied([])
    assert not rule.is_satisfied([100.0, 50.0])
    assert rule.describe_violation([100.0, 50.0, 100.0]) == (
        "target sample rates [50.0, 100.0] Hz differ; retrain the heads at one rate")


def test_a_stream_must_match_the_capture_width_the_calibration_was_made_from():
    rule = SubcarrierWidthMustMatchCalibration(128)
    assert rule.is_satisfied(128)
    assert not rule.is_satisfied(192)
    assert rule.describe_violation(192) == "192 subcarriers, the calibration was made from 128"


def test_a_legacy_calibration_without_a_width_accepts_every_width():
    rule = SubcarrierWidthMustMatchCalibration(None)
    assert rule.is_satisfied(64) and rule.is_satisfied(192)


def test_require_raises_a_violation_naming_the_rule_and_the_subject():
    with pytest.raises(BusinessRuleViolationError) as raised:
        require(SubcarrierWidthMustMatchContract(192), 128, "node 3")
    assert isinstance(raised.value, ValueError)
    assert str(raised.value) == ("SubcarrierWidthMustMatchContract violated by node 3: "
                                 "128 subcarriers, the model was trained on 192")
    assert isinstance(raised.value.rule, SubcarrierWidthMustMatchContract)


def test_require_passes_a_satisfied_rule():
    require(SubcarrierWidthMustMatchContract(192), 192, "node 3")


def test_sample_rate_may_fall_short_of_the_peer_median_by_at_most_the_tolerance():
    rule = SampleRateMustKeepPaceWithPeers(median_delivered_rate_hz=100.0, min_sample_rate_fraction=0.8)
    assert rule.min_sample_rate_hz == 100.0 * 0.8
    assert rule.is_satisfied(80.0)
    assert rule.is_satisfied(150.0)
    assert not rule.is_satisfied(79.9)
    assert rule.describe_violation(20.0) == "20.0 Hz delivered, at least 80.0 Hz needed to keep pace with the peers"


def test_delivered_rate_is_the_frame_intervals_per_second_across_the_span():
    assert measure_delivered_rate_hz(101, 2.0, 3.0) == pytest.approx(100.0)
    assert measure_delivered_rate_hz(2, 5.0, 5.5) == pytest.approx(2.0)


def test_delivered_rate_of_frames_sharing_one_timestamp_is_zero():
    assert measure_delivered_rate_hz(8, 4.0, 4.0) == 0.0
