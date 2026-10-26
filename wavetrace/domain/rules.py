"""Named domain invariants. A rule holds its expected value, so `is_satisfied` is O(1) and allocates
nothing unless its docstring says otherwise; the caller decides whether a violation refuses, drops or only
flags. A rule's class name is its name in errors, logs and violation counts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class BusinessRule(Protocol):
    def is_satisfied(self, observed) -> bool: ...

    def describe_violation(self, observed) -> str:
        """What was observed against what the rule expects; built only when a violation is reported."""


class BusinessRuleViolationError(ValueError):
    def __init__(self, rule: BusinessRule, observed, subject: str):
        super().__init__(f"{type(rule).__name__} violated by {subject}: {rule.describe_violation(observed)}")
        self.rule = rule


def require(rule: BusinessRule, observed, subject: str) -> None:
    """Raises `BusinessRuleViolationError` when `observed` breaks `rule`."""
    if not rule.is_satisfied(observed):
        raise BusinessRuleViolationError(rule, observed, subject)


@dataclass(frozen=True, slots=True)
class SubcarrierWidthMustCoverCalibration:
    """A frame must carry every subcarrier its calibration indexes."""
    required_width: int

    @classmethod
    def for_calibration(cls, calibration) -> "SubcarrierWidthMustCoverCalibration":
        """Required width = highest feature or image subcarrier index + 1."""
        subcarrier_indices = [int(i) for i in list(calibration.subcarriers) + list(calibration.image_subcarriers)]
        return cls(1 + max(subcarrier_indices))

    def is_satisfied(self, num_subcarriers: int) -> bool:
        return num_subcarriers >= self.required_width

    def describe_violation(self, num_subcarriers: int) -> str:
        return f"{num_subcarriers} subcarriers, the calibration needs {self.required_width}"


@dataclass(frozen=True, slots=True)
class SubcarrierWidthMustMatchContract:
    """A head scores only frames of the raw capture width it was trained on. `trained_width` None (a
    model saved before the width was recorded) accepts every width."""
    trained_width: int | None

    def is_satisfied(self, num_subcarriers: int) -> bool:
        return self.trained_width is None or num_subcarriers == self.trained_width

    def describe_violation(self, num_subcarriers: int) -> str:
        return f"{num_subcarriers} subcarriers, the model was trained on {self.trained_width}"


@dataclass(frozen=True, slots=True)
class SubcarrierWidthMustMatchCalibration:
    """A stream must carry the raw capture width its calibration was made from, or the calibration's
    subcarrier indices name other subcarriers. `calibrated_width` None (a calibration saved before the
    width was recorded) accepts every width."""
    calibrated_width: int | None

    def is_satisfied(self, num_subcarriers: int) -> bool:
        return self.calibrated_width is None or num_subcarriers == self.calibrated_width

    def describe_violation(self, num_subcarriers: int) -> str:
        return f"{num_subcarriers} subcarriers, the calibration was made from {self.calibrated_width}"


@dataclass(frozen=True, slots=True)
class SubcarrierWidthMustStayConstantPerLink:
    """A window must not mix CSI widths, so a link keeps the width of its first buffered batch."""
    pinned_width: int

    def is_satisfied(self, num_subcarriers: int) -> bool:
        return num_subcarriers == self.pinned_width

    def describe_violation(self, num_subcarriers: int) -> str:
        return f"{num_subcarriers} subcarriers, the link is pinned to {self.pinned_width}"


@dataclass(frozen=True, slots=True)
class LinkMustHaveServingHead:
    """Observed: the head and calibration serving the link, or None when none was loaded for it."""

    def is_satisfied(self, served_node) -> bool:
        return served_node is not None

    def describe_violation(self, served_node) -> str:
        return "no head is loaded for the link's rx node"


@dataclass(frozen=True, slots=True)
class HeadsMustShareClassOrdering:
    """Blending raw proba vectors is valid only when every head orders its classes alike.
    O(heads · classes)."""

    def is_satisfied(self, class_orderings: Sequence[Sequence[int]]) -> bool:
        return len({tuple(ordering) for ordering in class_orderings}) <= 1

    def describe_violation(self, class_orderings: Sequence[Sequence[int]]) -> str:
        orderings = sorted({tuple(ordering) for ordering in class_orderings})
        return f"class orderings {orderings} differ; retrain the heads consistently"


@dataclass(frozen=True, slots=True)
class HeadsMustShareSampleRate:
    """Heads served together blend their votes, so each must have been trained at the same rate, the one
    its link is resampled to. O(heads)."""

    def is_satisfied(self, target_sample_rates_hz: Sequence[float]) -> bool:
        return len(set(target_sample_rates_hz)) <= 1

    def describe_violation(self, target_sample_rates_hz: Sequence[float]) -> str:
        return f"target sample rates {sorted(set(target_sample_rates_hz))} Hz differ; retrain the heads at one rate"


def measure_delivered_rate_hz(num_frames: int, first_timestamp_s: float, last_timestamp_s: float) -> float:
    """Frames per second across the span of `num_frames` frames; 0.0 when the span is not positive."""
    span_s = last_timestamp_s - first_timestamp_s
    return (num_frames - 1) / span_s if span_s > 0 else 0.0


@dataclass(frozen=True, slots=True)
class SampleRateMustBeWithinTolerance:
    """The delivered rate may fall short of the target down to `min_sample_rate_fraction` of it; a faster
    stream is fine, the resampler decimates it."""
    target_sample_rate_hz: float
    min_sample_rate_fraction: float

    @property
    def min_sample_rate_hz(self) -> float:
        return self.target_sample_rate_hz * self.min_sample_rate_fraction

    def is_satisfied(self, delivered_rate_hz: float) -> bool:
        return delivered_rate_hz >= self.min_sample_rate_hz

    def describe_violation(self, delivered_rate_hz: float) -> str:
        return f"{delivered_rate_hz:.1f} Hz delivered, at least {self.min_sample_rate_hz:.1f} Hz needed"


@dataclass(frozen=True, slots=True)
class SampleRateMustKeepPaceWithPeers:
    """A link's delivered rate may fall short of the median delivered rate of the links measured at the
    same fuse down to `min_sample_rate_fraction` of it, so links stay in step with each other whatever
    rate they run at; a faster link is fine."""
    median_delivered_rate_hz: float
    min_sample_rate_fraction: float

    @property
    def min_sample_rate_hz(self) -> float:
        return self.median_delivered_rate_hz * self.min_sample_rate_fraction

    def is_satisfied(self, delivered_rate_hz: float) -> bool:
        return delivered_rate_hz >= self.min_sample_rate_hz

    def describe_violation(self, delivered_rate_hz: float) -> str:
        return (f"{delivered_rate_hz:.1f} Hz delivered, at least {self.min_sample_rate_hz:.1f} Hz needed "
                "to keep pace with the peers")
