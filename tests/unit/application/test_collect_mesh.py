from collections import deque

import pytest

from wavetrace import CsiFrame
from wavetrace.application.collect_mesh import (
    CaptureCondition, CollectMeshUseCase, HeadTrainingReport, train_every_node,
)

SAMPLE_RATE_HZ = 128.0  # binary-exact frame spacing, so resampling keeps the frame count
WINDOW_FRAMES = 4


def _frames(num_frames, start_s=0.0):
    frames = []
    for i in range(num_frames):
        frame = CsiFrame(1, 4)
        frame.timestamp = start_s + i / SAMPLE_RATE_HZ
        frames.append(frame)
    return frames


def _condition(name, *, num_beeps=0):
    return CaptureCondition(prompt=f"prompt {name}", countdown_s=3, hint=f"hint {name}", class_id=0,
                            class_name=name, num_beeps_after_capture=num_beeps)


class _ScriptedCapture:
    def __init__(self, captures):
        self._captures = deque(captures)
        self.calls = []

    def capture(self, prompt, *, countdown_s, hint):
        self.calls.append((prompt, countdown_s, hint))
        return self._captures.popleft()


class _FakePlan:
    """Sessions of `groups_per_session` groups; a request records which segments it was built from."""

    def __init__(self, groups_per_session):
        self._groups_per_session = groups_per_session

    def format_banner(self, node_ids):
        return f"banner {node_ids}"

    def plan_session(self, session_index, num_sessions):
        return self._groups_per_session

    def describe_link_dataset(self, session_index, conditions, link_key, segments):
        names = "+".join(condition.class_name for condition in conditions)
        return (session_index, names, link_key, [len(segment) for segment in segments])


class _FakeTraining:
    def __init__(self, *, is_trained=True):
        self._is_trained = is_trained
        self.dataset_dirs_by_node = {}

    def format_training_banner(self):
        return "training..."

    def train_node(self, node_id, dataset_dirs):
        self.dataset_dirs_by_node[node_id] = dataset_dirs
        return [HeadTrainingReport(f"node {node_id}: {len(dataset_dirs)} datasets",
                                   node_id if self._is_trained else None)]

    def format_trained_summary(self, trained_head_keys):
        return f"trained {trained_head_keys}"

    def format_untrained_error(self):
        return "nothing trained"


def _collect(plan, training, captures, node_ids, num_sessions):
    capture = _ScriptedCapture(captures)
    written = []

    def write_link_dataset(request):
        written.append(request)
        return f"ds{len(written)}"

    CollectMeshUseCase(plan, training, capture=capture, write_link_dataset=write_link_dataset,
                       target_sample_rate_hz=SAMPLE_RATE_HZ, min_window_frames=WINDOW_FRAMES,
                       ).run(node_ids, num_sessions)
    return capture, written


def test_a_link_is_written_only_when_every_condition_of_its_group_fills_a_window(capsys):
    plan = _FakePlan([((_condition("empty"), _condition("present")),)])
    empty = {("bb", 1): _frames(6), ("aa", 1): _frames(5), ("aa", 2): _frames(6), ("cc", 1): _frames(9)}
    present = {("aa", 1): _frames(7, 10.0), ("bb", 1): _frames(3, 10.0), ("aa", 2): _frames(4, 10.0),
               ("dd", 1): _frames(8, 10.0)}
    _, written = _collect(plan, _FakeTraining(), [empty, present], [1, 2], 1)
    assert written == [(0, "empty+present", ("aa", 1), [5, 7]), (0, "empty+present", ("aa", 2), [6, 4])]


def test_each_condition_is_resampled_on_its_own_so_the_gap_between_them_is_not_filled(capsys):
    plan = _FakePlan([((_condition("empty"), _condition("present")),)])
    _, written = _collect(plan, _FakeTraining(), [{("aa", 1): _frames(5)}, {("aa", 1): _frames(5, 60.0)}],
                          [1], 1)
    assert written[0][3] == [5, 5]


class _LoggingCapture:
    def __init__(self, events):
        self._events = events

    def capture(self, prompt, *, countdown_s, hint):
        self._events.append(f"capture {prompt}")
        return {("aa", 1): _frames(5), ("aa", 2): _frames(5)}


def _log_capture_and_write_order(groups_per_session):
    events = []

    def write_link_dataset(request):
        events.append(f"write {request[1]} {request[2]}")
        return "ds"

    CollectMeshUseCase(_FakePlan(groups_per_session), _FakeTraining(), capture=_LoggingCapture(events),
                       write_link_dataset=write_link_dataset, target_sample_rate_hz=SAMPLE_RATE_HZ,
                       min_window_frames=WINDOW_FRAMES).run([1, 2], 1)
    return events


def test_each_group_is_written_before_the_next_capture_starts(capsys):
    events = _log_capture_and_write_order([((_condition("zero"),),), ((_condition("one"),),)])
    assert events == ["capture prompt zero", "write zero ('aa', 1)", "write zero ('aa', 2)",
                      "capture prompt one", "write one ('aa', 1)", "write one ('aa', 2)"]


def test_a_group_captures_all_its_datasets_then_writes_them_node_by_node_in_dataset_order(capsys):
    events = _log_capture_and_write_order([((_condition("clear"),), (_condition("armed"),))])
    assert events == ["capture prompt clear", "capture prompt armed",
                      "write clear ('aa', 1)", "write armed ('aa', 1)",
                      "write clear ('aa', 2)", "write armed ('aa', 2)"]


def test_each_dataset_of_a_group_keeps_or_drops_a_link_on_its_own_conditions(capsys):
    plan = _FakePlan([((_condition("clear"),), (_condition("armed"),))])
    _, written = _collect(plan, _FakeTraining(), [{("aa", 1): _frames(5)}, {("aa", 1): _frames(3)}], [1], 1)
    assert written == [(0, "clear", ("aa", 1), [5])]


def test_capture_gets_each_conditions_prompt_countdown_and_hint_and_beeps_after_it(capsys):
    plan = _FakePlan([((_condition("one", num_beeps=3),),)])
    capture, _ = _collect(plan, _FakeTraining(), [{}, {}], [1], 2)
    assert capture.calls == [("prompt one", 3, "hint one")] * 2
    assert capsys.readouterr().out.startswith(
        "banner [1]\n\a\a\a   [SKIP] node 1 session 0: no link had >= 4 frames/class.\n\a\a\a")


def test_training_gets_each_nodes_dataset_dirs_in_write_order_and_prints_its_reports(capsys):
    plan = _FakePlan([((_condition("one"),),)])
    training = _FakeTraining()
    _collect(plan, training, [{("aa", 1): _frames(5), ("aa", 2): _frames(5), ("bb", 1): _frames(5)}],
             [1, 2], 1)
    assert training.dataset_dirs_by_node == {1: ["ds1", "ds2"], 2: ["ds3"]}
    assert capsys.readouterr().out == ("banner [1, 2]\ntraining...\nnode 1: 2 datasets\nnode 2: 1 datasets\n"
                                       "trained [1, 2]\n\a")


def test_a_node_with_no_written_link_in_a_session_is_reported_once_for_that_session(capsys):
    plan = _FakePlan([((_condition("a"),),), ((_condition("b"),),)])
    _collect(plan, _FakeTraining(), [{("aa", 1): _frames(5)}, {}, {}, {}], [1, 2], 2)
    output = capsys.readouterr().out
    assert output.split("training...")[0] == (
        "banner [1, 2]\n"
        "   [SKIP] node 2 session 0: no link had >= 4 frames/class.\n"
        "   [SKIP] node 1 session 1: no link had >= 4 frames/class.\n"
        "   [SKIP] node 2 session 1: no link had >= 4 frames/class.\n")


def test_nothing_trained_is_an_error_on_stderr_without_the_closing_bell(capsys):
    plan = _FakePlan([((_condition("one"),),)])
    _collect(plan, _FakeTraining(is_trained=False), [{("aa", 1): _frames(5)}], [1], 1)
    captured = capsys.readouterr()
    assert captured.out == "banner [1]\ntraining...\nnode 1: 1 datasets\n"
    assert captured.err == "nothing trained\n"


@pytest.mark.parametrize("num_frames", [WINDOW_FRAMES - 1, 0])
def test_a_link_under_one_window_is_never_written(capsys, num_frames):
    plan = _FakePlan([((_condition("one"),),)])
    _, written = _collect(plan, _FakeTraining(), [{("aa", 1): _frames(num_frames)}], [1], 1)
    assert written == []


class _TrainedThenSkippedTraining(_FakeTraining):
    def train_node(self, node_id, dataset_dirs):
        return [HeadTrainingReport(f"node {node_id} trained", node_id, {"node": node_id}),
                HeadTrainingReport(f"node {node_id} skipped", None)]


def test_train_every_node_returns_the_trained_heads_reports_in_training_order():
    logged, errors = [], []
    reports = train_every_node(_TrainedThenSkippedTraining(), [2, 1], {1: [], 2: []}, log=logged.append,
                               log_error=errors.append, should_stop=lambda: False)
    assert [(report.trained_head_key, report.metrics) for report in reports] == [(2, {"node": 2}), (1, {"node": 1})]
    assert logged == ["training...", "node 2 trained", "node 2 skipped", "node 1 trained", "node 1 skipped",
                      "trained [2, 1]"]
    assert errors == []


def test_train_every_node_returns_no_report_and_logs_the_error_when_nothing_trained():
    logged, errors = [], []
    reports = train_every_node(_FakeTraining(is_trained=False), [1], {1: ["d"]}, log=logged.append,
                               log_error=errors.append, should_stop=lambda: False)
    assert reports == [] and errors == ["nothing trained"]


def test_a_skipped_report_carries_no_metrics():
    assert HeadTrainingReport("   [SKIP] node 1", None).metrics is None


class _LazyTwoHeadTraining(_FakeTraining):
    """Trains each node's two heads only as its reports are consumed."""

    def __init__(self):
        super().__init__()
        self.trained_head_keys = []

    def train_node(self, node_id, dataset_dirs):
        for head in ("a", "b"):
            self.trained_head_keys.append((node_id, head))
            yield HeadTrainingReport(f"node {node_id} {head}", (node_id, head), {})


def test_a_stop_ends_training_before_the_next_head_and_names_the_heads_trained():
    training = _LazyTwoHeadTraining()
    logged = []
    reports = train_every_node(training, [1, 2], {1: [], 2: []}, log=logged.append, log_error=logged.append,
                               should_stop=lambda: bool(training.trained_head_keys))
    assert training.trained_head_keys == [(1, "a")]
    assert [report.trained_head_key for report in reports] == [(1, "a")]
    assert logged == ["training...", "node 1 a",
                      "stopped: trained heads [(1, 'a')]; every other head keeps its previous model."]
