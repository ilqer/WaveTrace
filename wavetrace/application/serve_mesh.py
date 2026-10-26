"""Live all-pairs mesh serving: buffer every (tx_short, rx_node_id) link, and on a fixed cadence score
each live link through a `ServeMode` and blend the scores into one verdict."""

import collections
import contextlib
import enum
import statistics
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from wavetrace.application.behaviours.rule_violations import RuleViolationCounter
from wavetrace.application.ports import LinkBatchReceiver
from wavetrace.application.serve import WindowView
from wavetrace.domain.contracts import DEFAULT_MIN_SAMPLE_RATE_FRACTION, require_valid_min_sample_rate_fraction
from wavetrace.domain.link_key import LinkKey
from wavetrace.domain.recognition.link_vote import LinkVoter
from wavetrace.domain.rules import (
    LinkMustHaveServingHead, SampleRateMustBeWithinTolerance, SampleRateMustKeepPaceWithPeers,
    SubcarrierWidthMustCoverCalibration, SubcarrierWidthMustMatchCalibration, SubcarrierWidthMustMatchContract,
    SubcarrierWidthMustStayConstantPerLink, measure_delivered_rate_hz,
)

_LINK_MUST_HAVE_SERVING_HEAD = LinkMustHaveServingHead()
_RULE_TYPES = (LinkMustHaveServingHead, SubcarrierWidthMustCoverCalibration, SubcarrierWidthMustMatchContract,
               SubcarrierWidthMustMatchCalibration, SubcarrierWidthMustStayConstantPerLink,
               SampleRateMustBeWithinTolerance, SampleRateMustKeepPaceWithPeers)


class RateGate(str, enum.Enum):
    """Which delivered rate a link must stay near to be scored at a fuse: its head's target rate, or the
    median rate of the links measured at that fuse."""
    TARGET = "target"
    PEERS = "peers"


class ServedNode(Protocol):
    weight: float  # static vote prior; 0 drops the link
    min_num_subcarriers: int
    trained_subcarrier_width: int | None  # None for a model saved before the width was recorded
    calibrated_subcarrier_width: int | None  # None for a calibration saved before the width was recorded
    target_sample_rate_hz: float


@dataclass(frozen=True, slots=True)
class LinkScore:
    vote_proba: np.ndarray  # in the mode's global class space
    quality: float  # live decision margin, 0 (unsure) .. 1 (confident)
    breakdown_token: str
    window_view: WindowView | None = None  # the window behind `vote_proba`, for display


@dataclass(frozen=True, slots=True)
class LinkReport:
    """One live link at a fuse; `score` is None while its frames cannot fill a window."""
    link_key: LinkKey
    frames: list
    score: LinkScore | None


@dataclass(frozen=True, slots=True)
class FusedVerdict:
    """`blended_proba` is None when no live link scored, or every scored link's static weight is 0."""
    now_s: float
    blended_proba: np.ndarray | None
    link_reports: tuple[LinkReport, ...]  # live links, in link-key order


class ServeMode(Protocol):
    def select_node(self, link_key: LinkKey) -> ServedNode | None:
        """The calibration and head that serve this link; None ignores the link."""

    def score_link(self, link_key: LinkKey, node: ServedNode, frames: list) -> LinkScore | None:
        """None while the frames cannot fill one window."""

    def format_banner(self, port: int) -> str: ...

    def format_verdict(self, blended_proba: np.ndarray) -> str: ...


@dataclass(slots=True)
class _LiveLink:
    link_id: int
    node: ServedNode
    frames: collections.deque
    last_seen_time_s: float
    sample_rate_rule: SampleRateMustBeWithinTolerance


@dataclass(slots=True)
class _LinkAdmission:
    """Built once per link key, so a batch is checked without allocating. `width_rules` run in order and
    the first broken one names the drop; the link's own width pin joins them with its first batch."""
    node: ServedNode | None
    width_rules: tuple
    live_link: _LiveLink | None = None


class ServeMeshUseCase:
    """clock: wall-clock seconds, read once per receive. chunk_s: fuse cadence. link_timeout_s: a link
    unheard this long leaves the vote. buffer_s: per-link frame history kept for windowing. rate_gate:
    which rule drops a slow link from a fuse (`PEERS` drops nothing while fewer than 2 links are measured);
    min_sample_rate_fraction: the share of the gate's reference rate a link must deliver."""

    def __init__(self, mode: ServeMode, *, clock: Callable[[], float], chunk_s: float,
                 link_timeout_s: float, buffer_s: float, rate_gate: RateGate = RateGate.TARGET,
                 min_sample_rate_fraction: float = DEFAULT_MIN_SAMPLE_RATE_FRACTION):
        require_valid_min_sample_rate_fraction(min_sample_rate_fraction)
        self._rate_gate = RateGate(rate_gate)
        self._min_sample_rate_fraction = min_sample_rate_fraction
        self._mode = mode
        self._clock = clock
        self._chunk_s = chunk_s
        self._link_timeout_s = link_timeout_s
        self._buffer_s = buffer_s
        self._links: dict[LinkKey, _LiveLink] = {}
        self._admissions: dict[LinkKey, _LinkAdmission] = {}
        self._violations = RuleViolationCounter(_RULE_TYPES, info_rule_types=(LinkMustHaveServingHead,))

    @property
    def violation_counts(self) -> dict[str, int]:
        """Per broken rule name: the frames dropped before buffering, and for the sample-rate rules the
        times a link was left out of a fuse."""
        return self._violations.counts

    def run(self, receiver: LinkBatchReceiver) -> None:
        """Prints one fused line per `chunk_s` until Ctrl-C."""
        try:
            with contextlib.closing(self.iter_fused(receiver)) as fused_verdicts:
                for fused_verdict in fused_verdicts:
                    if fused_verdict is not None:
                        _print_fused_verdict(self._mode, fused_verdict)
        except KeyboardInterrupt:
            print("\nstopped")

    def iter_fused(self, receiver: LinkBatchReceiver) -> Iterator[FusedVerdict | None]:
        """Yields None after each receive that does not fuse and one `FusedVerdict` per `chunk_s`;
        closing the generator closes `receiver`. A receive is O(frames received), a fuse
        O(links x window inference)."""
        next_fuse_time_s = self._clock() + self._chunk_s
        try:
            while True:
                now_s = self._clock()
                for link_key, frames in receiver.receive().items():
                    self._buffer(link_key, frames, now_s)
                if now_s < next_fuse_time_s:
                    yield None
                    continue
                next_fuse_time_s = now_s + self._chunk_s
                self._trim_history()
                yield self._fuse(now_s)
        finally:
            receiver.close()

    def _buffer(self, link_key: LinkKey, frames: list, now_s: float) -> None:
        admission = self._admissions.get(link_key)
        if admission is None:
            admission = self._admissions[link_key] = self._admit(link_key)
        if not _LINK_MUST_HAVE_SERVING_HEAD.is_satisfied(admission.node):
            self._violations.record(_LINK_MUST_HAVE_SERVING_HEAD, admission.node, link_key, now_s, len(frames))
            return
        num_subcarriers = frames[0].num_subcarriers
        for rule in admission.width_rules:
            if not rule.is_satisfied(num_subcarriers):
                self._violations.record(rule, num_subcarriers, link_key, now_s, len(frames))
                return
        link = admission.live_link
        if link is None:
            link = admission.live_link = self._links[link_key] = _LiveLink(
                len(self._links), admission.node, collections.deque(), now_s,
                SampleRateMustBeWithinTolerance(admission.node.target_sample_rate_hz,
                                                self._min_sample_rate_fraction))
            admission.width_rules += (SubcarrierWidthMustStayConstantPerLink(num_subcarriers),)
        link.frames.extend(frames)
        link.last_seen_time_s = now_s

    def _admit(self, link_key: LinkKey) -> _LinkAdmission:
        node = self._mode.select_node(link_key)
        if node is None:
            return _LinkAdmission(None, ())
        return _LinkAdmission(node, (SubcarrierWidthMustCoverCalibration(node.min_num_subcarriers),
                                     SubcarrierWidthMustMatchContract(node.trained_subcarrier_width),
                                     SubcarrierWidthMustMatchCalibration(node.calibrated_subcarrier_width)))

    def _trim_history(self) -> None:
        for link in self._links.values():
            frames = link.frames
            cutoff_s = frames[-1].timestamp - self._buffer_s
            while frames[0].timestamp < cutoff_s:
                frames.popleft()

    def _fuse(self, now_s: float) -> FusedVerdict:
        static_weights = {link.link_id: link.node.weight for link in self._links.values()}
        # all-zero priors would make every vote weightless, so fall back to a uniform blend
        voter = LinkVoter(static_weights if any(weight > 0 for weight in static_weights.values()) else None)
        link_reports = []
        delivered_rates_hz = self._measure_live_links(now_s)
        peers_rule = self._build_peers_rule(delivered_rates_hz)
        for link_key in sorted(delivered_rates_hz):
            link = self._links[link_key]
            delivered_rate_hz = delivered_rates_hz[link_key]
            sample_rate_rule = peers_rule if self._rate_gate is RateGate.PEERS else link.sample_rate_rule
            if sample_rate_rule is not None and not sample_rate_rule.is_satisfied(delivered_rate_hz):
                self._violations.record(sample_rate_rule, delivered_rate_hz, link_key, now_s)
                link_reports.append(LinkReport(link_key, list(link.frames), None))
                continue
            frames = list(link.frames)
            score = self._mode.score_link(link_key, link.node, frames)
            if score is not None:
                voter.add(link.link_id, score.vote_proba, quality=score.quality)
            link_reports.append(LinkReport(link_key, frames, score))
        try:
            _class_id, blended_proba = voter.finalize()
        except ValueError:
            blended_proba = None
        return FusedVerdict(now_s, blended_proba, tuple(link_reports))

    def _measure_live_links(self, now_s: float) -> dict[LinkKey, float]:
        """Delivered rate of each link heard within the timeout with 2 or more buffered frames. O(links)."""
        return {link_key: _delivered_rate_hz(link.frames) for link_key, link in self._links.items()
                if now_s - link.last_seen_time_s <= self._link_timeout_s and len(link.frames) >= 2}

    def _build_peers_rule(self, delivered_rates_hz: dict[LinkKey, float]) -> SampleRateMustKeepPaceWithPeers | None:
        """None under the `TARGET` gate, or while fewer than 2 links are measured. O(links log links)."""
        if self._rate_gate is not RateGate.PEERS or len(delivered_rates_hz) < 2:
            return None
        return SampleRateMustKeepPaceWithPeers(statistics.median(delivered_rates_hz.values()),
                                               self._min_sample_rate_fraction)


def _delivered_rate_hz(frames: collections.deque) -> float:
    return measure_delivered_rate_hz(len(frames), frames[0].timestamp, frames[-1].timestamp)


def _print_fused_verdict(mode: ServeMode, fused_verdict: FusedVerdict) -> None:
    breakdown_tokens = [report.score.breakdown_token for report in fused_verdict.link_reports
                        if report.score is not None]
    if not breakdown_tokens:
        print("\r(no live links with a full window yet)            ", end="", flush=True)
        return
    if fused_verdict.blended_proba is None:
        print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
        return
    verdict = mode.format_verdict(fused_verdict.blended_proba)
    print(f"{verdict}  [{len(breakdown_tokens)} links] " + " ".join(breakdown_tokens))
