from collections import deque
from types import SimpleNamespace

import numpy as np

from wavetrace.application.serve_mesh import LinkScore, ServeMeshUseCase


class _FakeClock:
    def __init__(self):
        self.now_s = 0.0

    def __call__(self):
        return self.now_s


class _ScriptedReceiver:
    """Each receive advances the clock by step_s; Ctrl-C once the batches run out."""

    def __init__(self, clock, batches, step_s):
        self._clock = clock
        self._batches = deque(batches)
        self._step_s = step_s
        self.is_closed = False

    def receive(self):
        if not self._batches:
            raise KeyboardInterrupt
        self._clock.now_s += self._step_s
        return self._batches.popleft()

    def close(self):
        self.is_closed = True


class _FakeMode:
    def __init__(self, nodes, proba=(0.25, 0.75)):
        self.nodes = nodes
        self._proba = None if proba is None else np.array(proba)
        self.scored_timestamps = {}

    def select_node(self, link_key):
        return self.nodes.get(link_key[1])

    def score_link(self, link_key, node, frames):
        self.scored_timestamps[link_key] = [frame.timestamp for frame in frames]
        if self._proba is None:
            return None
        return LinkScore(self._proba, 1.0, f"{link_key[0]}->{link_key[1]}")

    def format_banner(self, port):
        return ""

    def format_verdict(self, blended_proba):
        return f"P {blended_proba[1]:.2f}"


def _node(weight=1.0):
    return SimpleNamespace(weight=weight, min_num_subcarriers=2)


def _frames(*timestamps, num_subcarriers=4):
    return [SimpleNamespace(timestamp=timestamp, num_subcarriers=num_subcarriers) for timestamp in timestamps]


def _serve(mode, batches, capsys, *, step_s=0.6, link_timeout_s=3.0, buffer_s=3.0):
    clock = _FakeClock()
    receiver = _ScriptedReceiver(clock, batches, step_s)
    ServeMeshUseCase(mode, clock=clock, chunk_s=1.0, link_timeout_s=link_timeout_s,
                     buffer_s=buffer_s).run(receiver)
    assert receiver.is_closed
    return capsys.readouterr().out


def test_fuses_served_links_in_link_key_order(capsys):
    mode = _FakeMode({1: _node(), 2: _node()})
    batch = {("b", 1): _frames(0.0, 0.1), ("a", 1): _frames(0.0, 0.1),
             ("a", 9): _frames(0.0, 0.1), ("c", 2): _frames(0.0, 0.1, num_subcarriers=1)}
    output = _serve(mode, [batch, {}, {}], capsys)
    assert output == "P 0.75  [2 links] a->1 b->1\n\nstopped\n"


def test_waits_until_a_link_fills_a_window(capsys):
    mode = _FakeMode({1: _node()}, proba=None)
    output = _serve(mode, [{("a", 1): _frames(0.0, 0.1)}, {}, {}], capsys)
    assert output == "\r(no live links with a full window yet)            \nstopped\n"


def test_single_frame_link_is_not_scored(capsys):
    mode = _FakeMode({1: _node()})
    _serve(mode, [{("a", 1): _frames(0.0)}, {}, {}], capsys)
    assert mode.scored_timestamps == {}


def test_all_zero_priors_fall_back_to_a_uniform_vote(capsys):
    mode = _FakeMode({1: _node(weight=0.0)})
    output = _serve(mode, [{("a", 1): _frames(0.0, 0.1)}, {}, {}], capsys)
    assert output.startswith("P 0.75  [1 links]")


def test_only_zero_weight_links_live_reports_chance_level(capsys):
    mode = _FakeMode({1: _node(weight=0.0), 2: _node(weight=1.0)})
    batches = [{("a", 2): _frames(0.0, 0.1)}, {}, {}, {("a", 1): _frames(0.0, 0.1)}, {}]
    output = _serve(mode, batches, capsys, step_s=0.3, link_timeout_s=0.5)
    assert output == "\r(live links present, but all from chance-level nodes)   \nstopped\n"


def test_history_keeps_only_the_last_buffer_seconds(capsys):
    mode = _FakeMode({1: _node()})
    _serve(mode, [{("a", 1): _frames(0.0, 1.0, 2.0, 3.0, 4.0)}, {}, {}], capsys, buffer_s=1.0)
    assert mode.scored_timestamps[("a", 1)] == [3.0, 4.0]


def _iter_fused(mode, batches, *, step_s=0.6):
    clock = _FakeClock()
    receiver = _ScriptedReceiver(clock, batches, step_s)
    use_case = ServeMeshUseCase(mode, clock=clock, chunk_s=1.0, link_timeout_s=3.0, buffer_s=3.0)
    return use_case.iter_fused(receiver), receiver


def test_iter_fused_yields_none_between_fuses_and_reports_links_in_link_key_order():
    mode = _FakeMode({1: _node(), 2: _node()})
    batch = {("b", 1): _frames(0.0, 0.1), ("a", 1): _frames(0.0, 0.1),
             ("a", 9): _frames(0.0, 0.1), ("c", 2): _frames(0.0, 0.1, num_subcarriers=1)}
    fused_verdicts, _receiver = _iter_fused(mode, [batch, {}, {}])
    assert next(fused_verdicts) is None
    assert next(fused_verdicts) is None
    fused_verdict = next(fused_verdicts)
    assert fused_verdict.now_s == 1.2
    assert list(fused_verdict.blended_proba) == [0.25, 0.75]
    assert [report.link_key for report in fused_verdict.link_reports] == [("a", 1), ("b", 1)]
    assert [[frame.timestamp for frame in report.frames] for report in fused_verdict.link_reports] == [
        [0.0, 0.1], [0.0, 0.1]]
    assert [report.score.breakdown_token for report in fused_verdict.link_reports] == ["a->1", "b->1"]


def test_iter_fused_reports_an_unscored_live_link_without_a_blend():
    fused_verdicts, _receiver = _iter_fused(_FakeMode({1: _node()}, proba=None), [{("a", 1): _frames(0.0, 0.1)}, {}, {}])
    fused_verdict = [next(fused_verdicts) for _ in range(3)][-1]
    assert fused_verdict.blended_proba is None
    assert [(report.link_key, report.score) for report in fused_verdict.link_reports] == [(("a", 1), None)]


def test_closing_iter_fused_closes_the_receiver():
    fused_verdicts, receiver = _iter_fused(_FakeMode({1: _node()}), [{}, {}])
    next(fused_verdicts)
    assert not receiver.is_closed
    fused_verdicts.close()
    assert receiver.is_closed


def test_a_link_keeps_the_width_of_its_first_batch(capsys):
    mode = _FakeMode({1: _node()})
    batches = [{("a", 1): _frames(0.0, 0.1)}, {("a", 1): _frames(0.2, 0.3, num_subcarriers=6)}, {}]
    output = _serve(mode, batches, capsys)
    assert mode.scored_timestamps[("a", 1)] == [0.0, 0.1]
    assert output.startswith("P 0.75  [1 links]")
