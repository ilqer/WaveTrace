import logging
from types import SimpleNamespace

import pytest

from wavetrace.application.behaviours import RuleViolationCounter, iter_frames_satisfying_width
from wavetrace.domain.rules import (
    LinkMustHaveServingHead, SubcarrierWidthMustCoverCalibration, SubcarrierWidthMustMatchContract,
)

LOGGER_NAME = "wavetrace.application.behaviours.rule_violations"


def _counter():
    return RuleViolationCounter((SubcarrierWidthMustMatchContract, LinkMustHaveServingHead), log_interval_s=10.0)


def _warnings(caplog):
    return [record.getMessage() for record in caplog.records
            if record.name == LOGGER_NAME and record.levelno == logging.WARNING]


def test_counts_start_at_zero_for_every_rule():
    assert _counter().counts == {"SubcarrierWidthMustMatchContract": 0, "LinkMustHaveServingHead": 0}


def test_counts_violations_per_rule():
    counter = _counter()
    counter.record(SubcarrierWidthMustMatchContract(192), 128, ("4f:9c", 2), 0.0, num_violations=5)
    counter.record(SubcarrierWidthMustMatchContract(192), 128, ("4f:9c", 2), 1.0, num_violations=3)
    assert counter.counts == {"SubcarrierWidthMustMatchContract": 8, "LinkMustHaveServingHead": 0}


def test_the_first_violation_logs_the_rule_the_subject_and_the_counts(caplog):
    counter = _counter()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        counter.record(SubcarrierWidthMustMatchContract(192), 128, ("4f:9c", 2), 0.0, num_violations=5)
    assert _warnings(caplog) == ["SubcarrierWidthMustMatchContract violated on ('4f:9c', 2): 128 subcarriers, "
                                 "the model was trained on 192 (5 since the last report, 5 in total)"]


def test_logs_at_most_once_per_rule_per_interval(caplog):
    counter = _counter()
    rule = SubcarrierWidthMustMatchContract(192)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        for now_s in (0.0, 4.0, 9.9):
            counter.record(rule, 128, ("4f:9c", 2), now_s)
        counter.record(rule, 64, ("64:b8", 1), 10.0)
    assert _warnings(caplog) == [
        "SubcarrierWidthMustMatchContract violated on ('4f:9c', 2): 128 subcarriers, the model was trained on 192 "
        "(1 since the last report, 1 in total)",
        "SubcarrierWidthMustMatchContract violated on ('64:b8', 1): 64 subcarriers, the model was trained on 192 "
        "(3 since the last report, 4 in total)",
    ]


def test_each_rule_has_its_own_log_interval(caplog):
    counter = _counter()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        counter.record(SubcarrierWidthMustMatchContract(192), 128, ("4f:9c", 2), 0.0)
        counter.record(LinkMustHaveServingHead(), None, ("4f:9c", 7), 1.0)
    assert [message.split(" ")[0] for message in _warnings(caplog)] == [
        "SubcarrierWidthMustMatchContract", "LinkMustHaveServingHead"]


def test_a_quiet_rule_logs_at_info_while_the_others_keep_warning(caplog):
    counter = RuleViolationCounter((SubcarrierWidthMustMatchContract, LinkMustHaveServingHead),
                                   info_rule_types=(LinkMustHaveServingHead,))
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        counter.record(LinkMustHaveServingHead(), None, ("4f:9c", 7), 0.0)
        counter.record(SubcarrierWidthMustMatchContract(192), 128, ("4f:9c", 2), 0.0)
    assert [(record.levelno, record.getMessage().split(" ")[0]) for record in caplog.records
            if record.name == LOGGER_NAME] == [(logging.INFO, "LinkMustHaveServingHead"),
                                               (logging.WARNING, "SubcarrierWidthMustMatchContract")]
    assert counter.counts == {"SubcarrierWidthMustMatchContract": 1, "LinkMustHaveServingHead": 1}


def test_a_rule_it_was_not_built_for_is_a_programming_error():
    with pytest.raises(KeyError):
        _counter().record(SubcarrierWidthMustCoverCalibration(10), 9, "node 3", 0.0)


def test_frames_of_another_width_are_dropped_and_counted():
    frames = [SimpleNamespace(timestamp=float(i), num_subcarriers=width) for i, width in enumerate([4, 6, 4, 6, 6])]
    counter = RuleViolationCounter((SubcarrierWidthMustMatchContract,))
    kept = list(iter_frames_satisfying_width(frames, SubcarrierWidthMustMatchContract(4), counter))
    assert [frame.timestamp for frame in kept] == [0.0, 2.0]
    assert counter.counts == {"SubcarrierWidthMustMatchContract": 3}
