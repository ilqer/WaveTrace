"""Labelled mesh collection: capture each condition of a session over every (tx_short, rx_node_id)
link, write one dataset per usable link through a `CollectPlan`, then train every node's heads through
a `TrainingPolicy`."""

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from wavetrace import Label
from wavetrace.application.ports import LinkCapture
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.domain.groundtruth.labeler import ScriptedLabeler
from wavetrace.domain.link_key import LinkKey


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureCondition:
    prompt: str
    countdown_s: int
    hint: str
    class_id: int
    class_name: str
    num_beeps_after_capture: int


@dataclass(frozen=True, slots=True, kw_only=True)
class LinkDatasetRequest:
    """frames: the link's resampled recording; spans and labeler are `collect_dataset`'s (labeler None
    labels by `stage` over `spans`; a list of camera Labels is aligned to the windows by timestamp)."""

    frames: list
    rx_node_id: int
    recording_dir: str
    dataset_dir: str
    spans: list[tuple[float, float]]
    stage: str
    session_id: str
    subject_id: str
    labeler: ScriptedLabeler | list[Label] | None
    should_subtract_ic_baseline: bool


@dataclass(frozen=True, slots=True)
class HeadTrainingReport:
    line: str  # may span several lines
    trained_head_key: object | None  # None when the head was skipped
    metrics: dict | None = None  # the trained head's metrics.json content


class CollectPlan(Protocol):
    def format_banner(self, node_ids: list[int]) -> str: ...

    def plan_session(self, session_index: int,
                     num_sessions: int) -> list[tuple[tuple[CaptureCondition, ...], ...]]:
        """The session's capture groups in order. Every condition of a group is captured before any of
        its datasets is written; each inner tuple is one dataset's conditions, labelled together from
        one resampled segment each."""

    def describe_link_dataset(self, session_index: int, conditions: tuple[CaptureCondition, ...],
                              link_key: LinkKey, segments: list[list]) -> LinkDatasetRequest:
        """segments: one resampled frame list per condition, each at least one window long."""


class TrainingPolicy(Protocol):
    def format_training_banner(self) -> str: ...

    def train_node(self, node_id: int, dataset_dirs: list) -> Iterable[HeadTrainingReport]:
        """dataset_dirs: the node's datasets, in write order; a collect mode may train on a wider pool
        (weapon trains on the cumulative weapon_ds pool). A head may train only when its report is
        consumed."""

    def format_trained_summary(self, trained_head_keys: list) -> str: ...

    def format_untrained_error(self) -> str: ...


class CollectMode(CollectPlan, TrainingPolicy, Protocol):
    model_root: str  # created before the first capture


class CollectMeshUseCase:
    """write_link_dataset: saves one request's recording and dataset, returns the dataset dir.
    min_window_frames: a resampled segment shorter than this emits no window, so its link is dropped."""

    def __init__(self, plan: CollectPlan, training_policy: TrainingPolicy, *, capture: LinkCapture,
                 write_link_dataset: Callable[[LinkDatasetRequest], str], target_sample_rate_hz: float,
                 min_window_frames: int):
        self._plan = plan
        self._training_policy = training_policy
        self._capture = capture
        self._write_link_dataset = write_link_dataset
        self._target_sample_rate_hz = target_sample_rate_hz
        self._min_window_frames = min_window_frames

    def run(self, node_ids: list[int], num_sessions: int) -> None:
        """Peak memory is one group's captures, O(conditions x links x frames per link)."""
        print(self._plan.format_banner(node_ids))
        dataset_dirs_by_node = {node_id: [] for node_id in node_ids}
        for session_index in range(num_sessions):
            num_written_by_node = dict.fromkeys(node_ids, 0)
            for group in self._plan.plan_session(session_index, num_sessions):
                captures_by_dataset = [[self._capture_condition(condition) for condition in conditions]
                                       for conditions in group]
                for node_id in node_ids:
                    for conditions, captures in zip(group, captures_by_dataset):
                        dataset_dirs = list(self._write_link_datasets(session_index, conditions, captures,
                                                                      node_id))
                        dataset_dirs_by_node[node_id].extend(dataset_dirs)
                        num_written_by_node[node_id] += len(dataset_dirs)
            for node_id in node_ids:
                if num_written_by_node[node_id] == 0:
                    print(f"   [SKIP] node {node_id} session {session_index}: "
                          f"no link had >= {self._min_window_frames} frames/class.")
        self._train(node_ids, dataset_dirs_by_node)

    def _capture_condition(self, condition: CaptureCondition) -> dict:
        capture = self._capture.capture(condition.prompt, countdown_s=condition.countdown_s,
                                        hint=condition.hint)
        if condition.num_beeps_after_capture:
            print("\a" * condition.num_beeps_after_capture, end="", flush=True)
        return capture

    def _write_link_datasets(self, session_index, conditions, captures, node_id):
        link_keys = {link_key for capture in captures for link_key in capture if link_key[1] == node_id}
        for link_key in sorted(link_keys):
            # resampled apart: one resample across conditions would interpolate frames over the gap
            segments = [resample_uniform(capture.get(link_key, []), self._target_sample_rate_hz)
                        for capture in captures]
            if any(len(segment) < self._min_window_frames for segment in segments):
                continue
            yield self._write_link_dataset(
                self._plan.describe_link_dataset(session_index, conditions, link_key, segments))

    def _train(self, node_ids, dataset_dirs_by_node):
        if train_every_node(self._training_policy, node_ids, dataset_dirs_by_node, log=print,
                            log_error=lambda message: print(message, file=sys.stderr), should_stop=lambda: False):
            print("\a", end="", flush=True)


def train_every_node(training_policy: TrainingPolicy, node_ids: list[int], dataset_dirs_by_node: dict[int, list],
                     *, log: Callable[[str], None], log_error: Callable[[str], None],
                     should_stop: Callable[[], bool]) -> list[HeadTrainingReport]:
    """Logs the policy's banner, each head's report line, then its summary, or its error through `log_error`
    when no head trained. `should_stop` is asked after every head; once true no further head trains and the
    heads trained so far are logged instead of the summary. Returns the trained heads' reports in training
    order."""
    log(training_policy.format_training_banner())
    trained_reports = []
    reports = (report for node_id in node_ids
               for report in training_policy.train_node(node_id, dataset_dirs_by_node[node_id]))
    for report in reports:
        log(report.line)
        if report.trained_head_key is not None:
            trained_reports.append(report)
        if should_stop():
            log(f"stopped: trained heads {[report.trained_head_key for report in trained_reports]}; "
                "every other head keeps its previous model.")
            return trained_reports
    if not trained_reports:
        log_error(training_policy.format_untrained_error())
        return trained_reports
    log(training_policy.format_trained_summary([report.trained_head_key for report in trained_reports]))
    return trained_reports
