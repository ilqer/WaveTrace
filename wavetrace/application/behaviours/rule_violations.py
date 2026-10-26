import logging
import math
from collections.abc import Iterable, Iterator

from wavetrace.domain.rules import BusinessRule

_logger = logging.getLogger(__name__)


class RuleViolationCounter:
    """Counts each rule's violations and logs at most one WARNING per rule per `log_interval_s`, naming
    the rule, the subject and the counts; a rule in `info_rule_types` logs at INFO instead. Counters exist
    only for the rule types it was built with, so `record` is O(1) and allocates nothing between log
    lines; an unknown rule raises KeyError. `now_s` is the caller's clock, so replays rate-limit on their
    own time."""

    def __init__(self, rule_types: Iterable[type], *, info_rule_types: Iterable[type] = (),
                 log_interval_s: float = 10.0):
        rule_names = [rule_type.__name__ for rule_type in rule_types]
        info_rule_names = {rule_type.__name__ for rule_type in info_rule_types}
        self._log_level_by_rule = {name: logging.INFO if name in info_rule_names else logging.WARNING
                                   for name in rule_names}
        self._total_by_rule = dict.fromkeys(rule_names, 0)
        self._unreported_by_rule = dict.fromkeys(rule_names, 0)
        self._next_log_time_s_by_rule = dict.fromkeys(rule_names, -math.inf)
        self._log_interval_s = log_interval_s

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._total_by_rule)

    def record(self, rule: BusinessRule, observed, subject, now_s: float, num_violations: int = 1) -> None:
        rule_name = type(rule).__name__
        self._total_by_rule[rule_name] += num_violations
        self._unreported_by_rule[rule_name] += num_violations
        if now_s < self._next_log_time_s_by_rule[rule_name]:
            return
        self._next_log_time_s_by_rule[rule_name] = now_s + self._log_interval_s
        _logger.log(self._log_level_by_rule[rule_name],
                    "%s violated on %s: %s (%d since the last report, %d in total)", rule_name, subject,
                    rule.describe_violation(observed), self._unreported_by_rule[rule_name],
                    self._total_by_rule[rule_name])
        self._unreported_by_rule[rule_name] = 0


def iter_frames_satisfying_width(frames: Iterable, width_rule: BusinessRule,
                                 violations: RuleViolationCounter) -> Iterator:
    """Yields each frame whose subcarrier width satisfies `width_rule` and records every other one as a
    violation at its own timestamp. O(1) per frame."""
    for frame in frames:
        if width_rule.is_satisfied(frame.num_subcarriers):
            yield frame
        else:
            violations.record(width_rule, frame.num_subcarriers, "the served stream", frame.timestamp)
