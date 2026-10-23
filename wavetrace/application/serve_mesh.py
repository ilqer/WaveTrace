"""Live all-pairs mesh serving: buffer every (tx_short, rx_node_id) link, and on a fixed cadence score
each live link through a `ServeMode` and blend the scores into one verdict."""

import collections
import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from wavetrace.application.ports import LinkBatchReceiver
from wavetrace.application.serve import WindowView
from wavetrace.domain.link_key import LinkKey
from wavetrace.domain.recognition.link_vote import LinkVoter


class ServedNode(Protocol):
    weight: float  # static vote prior; 0 drops the link
    min_num_subcarriers: int


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
    num_subcarriers: int  # pinned by the first batch: a window must not mix CSI widths
    frames: collections.deque
    last_seen_time_s: float


class ServeMeshUseCase:
    """clock: wall-clock seconds, read once per receive. chunk_s: fuse cadence. link_timeout_s: a link
    unheard this long leaves the vote. buffer_s: per-link frame history kept for windowing."""

    def __init__(self, mode: ServeMode, *, clock: Callable[[], float], chunk_s: float,
                 link_timeout_s: float, buffer_s: float):
        self._mode = mode
        self._clock = clock
        self._chunk_s = chunk_s
        self._link_timeout_s = link_timeout_s
        self._buffer_s = buffer_s
        self._links: dict[LinkKey, _LiveLink] = {}

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
        link = self._links.get(link_key)
        node = self._mode.select_node(link_key) if link is None else link.node
        num_subcarriers = frames[0].num_subcarriers
        if node is None or num_subcarriers < node.min_num_subcarriers:
            return
        if link is None:
            link = self._links[link_key] = _LiveLink(len(self._links), node, num_subcarriers,
                                                     collections.deque(), now_s)
        elif num_subcarriers != link.num_subcarriers:
            return
        link.frames.extend(frames)
        link.last_seen_time_s = now_s

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
        for link_key in sorted(self._links):
            link = self._links[link_key]
            if now_s - link.last_seen_time_s > self._link_timeout_s or len(link.frames) < 2:
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
