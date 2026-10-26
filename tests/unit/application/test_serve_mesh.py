import logging
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from wavetrace.application.serve_mesh import LinkScore, RateGate, ServeMeshUseCase


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


def _node(weight=1.0, trained_subcarrier_width=None, calibrated_subcarrier_width=None, target_sample_rate_hz=10.0):
    return SimpleNamespace(weight=weight, min_num_subcarriers=2, trained_subcarrier_width=trained_subcarrier_width,
                           calibrated_subcarrier_width=calibrated_subcarrier_width,
                           target_sample_rate_hz=target_sample_rate_hz)


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
    mode = _FakeMode({1: _node(target_sample_rate_hz=1.0)})
    _serve(mode, [{("a", 1): _frames(0.0, 1.0, 2.0, 3.0, 4.0)}, {}, {}], capsys, buffer_s=1.0)
    assert mode.scored_timestamps[("a", 1)] == [3.0, 4.0]


def _iter_fused(mode, batches, *, step_s=0.6):
    fused_verdicts, receiver, _use_case = _iter_fused_with_use_case(mode, batches, step_s=step_s)
    return fused_verdicts, receiver


def _iter_fused_with_use_case(mode, batches, *, step_s=0.6, **use_case_options):
    clock = _FakeClock()
    receiver = _ScriptedReceiver(clock, batches, step_s)
    use_case = ServeMeshUseCase(mode, clock=clock, chunk_s=1.0, link_timeout_s=3.0, buffer_s=3.0,
                                **use_case_options)
    return use_case.iter_fused(receiver), receiver, use_case


def _drain(fused_verdicts):
    """Every fuse until the receiver runs out (it raises Ctrl-C then)."""
    verdicts = []
    try:
        for fused_verdict in fused_verdicts:
            verdicts.append(fused_verdict)
    except KeyboardInterrupt:
        pass
    return [verdict for verdict in verdicts if verdict is not None]


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


MESH_LOGGER = "wavetrace.application.behaviours.rule_violations"


def _messages_at(caplog, level):
    return [record.getMessage() for record in caplog.records
            if record.name == MESH_LOGGER and record.levelno == level]


def _warnings(caplog):
    return _messages_at(caplog, logging.WARNING)


def test_every_dropped_batch_is_counted_under_the_rule_it_broke(caplog):
    mode = _FakeMode({1: _node(), 2: _node()})
    batches = [{("a", 1): _frames(0.0, 0.1), ("a", 9): _frames(0.0, 0.1, 0.2),
                ("c", 2): _frames(0.0, 0.1, num_subcarriers=1)},
               {("a", 1): _frames(0.2, 0.3, num_subcarriers=6)}, {}]
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, batches)
    with caplog.at_level(logging.INFO, logger=MESH_LOGGER):
        _drain(fused_verdicts)
    assert use_case.violation_counts == {
        "LinkMustHaveServingHead": 3, "SubcarrierWidthMustCoverCalibration": 2,
        "SubcarrierWidthMustMatchContract": 0, "SubcarrierWidthMustMatchCalibration": 0,
        "SubcarrierWidthMustStayConstantPerLink": 2, "SampleRateMustBeWithinTolerance": 0,
        "SampleRateMustKeepPaceWithPeers": 0}
    assert _messages_at(caplog, logging.INFO) == [
        "LinkMustHaveServingHead violated on ('a', 9): no head is loaded for the link's rx node "
        "(3 since the last report, 3 in total)"]
    assert sorted(_warnings(caplog)) == [
        "SubcarrierWidthMustCoverCalibration violated on ('c', 2): 1 subcarriers, the calibration needs 2 "
        "(2 since the last report, 2 in total)",
        "SubcarrierWidthMustStayConstantPerLink violated on ('a', 1): 6 subcarriers, the link is pinned to 4 "
        "(2 since the last report, 2 in total)",
    ]


def test_a_model_scores_only_frames_of_its_trained_capture_width():
    mode = _FakeMode({1: _node(trained_subcarrier_width=6)})
    batches = [{("a", 1): _frames(0.0, 0.1)}, {("a", 1): _frames(0.2, 0.3, num_subcarriers=6)}, {}, {}]
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, batches)
    fused = _drain(fused_verdicts)
    assert [[frame.timestamp for frame in report.frames] for report in fused[0].link_reports] == [[0.2, 0.3]]
    assert use_case.violation_counts["SubcarrierWidthMustMatchContract"] == 2
    assert use_case.violation_counts["SubcarrierWidthMustStayConstantPerLink"] == 0


def test_a_legacy_model_keeps_pinning_the_first_batch_width():
    mode = _FakeMode({1: _node(trained_subcarrier_width=None)})
    batches = [{("a", 1): _frames(0.0, 0.1)}, {("a", 1): _frames(0.2, 0.3, num_subcarriers=6)}, {}, {}]
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, batches)
    fused = _drain(fused_verdicts)
    assert [[frame.timestamp for frame in report.frames] for report in fused[0].link_reports] == [[0.0, 0.1]]
    assert use_case.violation_counts["SubcarrierWidthMustMatchContract"] == 0


def test_a_clean_stream_counts_no_violation(caplog):
    mode = _FakeMode({1: _node(trained_subcarrier_width=4)})
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, [{("a", 1): _frames(0.0, 0.1)}, {}, {}])
    with caplog.at_level(logging.WARNING, logger=MESH_LOGGER):
        _drain(fused_verdicts)
    assert set(use_case.violation_counts.values()) == {0}
    assert _warnings(caplog) == []


def test_a_stream_of_another_width_than_its_calibration_is_dropped_and_counted():
    mode = _FakeMode({1: _node(calibrated_subcarrier_width=6)})
    batches = [{("a", 1): _frames(0.0, 0.1)}, {("a", 1): _frames(0.2, 0.3, num_subcarriers=6)}, {}, {}]
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, batches)
    fused = _drain(fused_verdicts)
    assert [[frame.timestamp for frame in report.frames] for report in fused[0].link_reports] == [[0.2, 0.3]]
    assert use_case.violation_counts["SubcarrierWidthMustMatchCalibration"] == 2


def test_a_legacy_calibration_without_a_width_does_not_restrict_the_stream():
    mode = _FakeMode({1: _node(calibrated_subcarrier_width=None)})
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, [{("a", 1): _frames(0.0, 0.1)}, {}, {}])
    _drain(fused_verdicts)
    assert use_case.violation_counts["SubcarrierWidthMustMatchCalibration"] == 0


@pytest.mark.parametrize("step_s, is_scored", [(0.125, True), (0.1, True), (0.2, False)])
def test_a_link_is_scored_only_while_it_delivers_eighty_percent_of_the_target_rate(step_s, is_scored):
    mode = _FakeMode({1: _node(target_sample_rate_hz=10.0)})
    frames = _frames(*(index * step_s for index in range(8)))
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, [{("a", 1): frames}, {}, {}])
    fused = _drain(fused_verdicts)
    assert (fused[0].link_reports[0].score is not None) is is_scored
    assert use_case.violation_counts["SampleRateMustBeWithinTolerance"] == (0 if is_scored else 1)
    assert fused[0].link_reports[0].frames == frames


def test_a_slow_link_logs_a_warning_naming_the_delivered_and_needed_rate(caplog):
    mode = _FakeMode({1: _node(target_sample_rate_hz=10.0)})
    fused_verdicts, _receiver, _use_case = _iter_fused_with_use_case(
        mode, [{("a", 1): _frames(0.0, 0.5, 1.0)}, {}, {}])
    with caplog.at_level(logging.WARNING, logger=MESH_LOGGER):
        _drain(fused_verdicts)
    assert _warnings(caplog) == ["SampleRateMustBeWithinTolerance violated on ('a', 1): 2.0 Hz delivered, "
                                 "at least 8.0 Hz needed (1 since the last report, 1 in total)"]


def _frames_at(rate_hz, num_frames=8):
    return _frames(*(index / rate_hz for index in range(num_frames)))


def _scored_link_keys(fused):
    return [report.link_key for report in fused[0].link_reports if report.score is not None]


def _fuse_once(mode, batch, **use_case_options):
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, [batch, {}, {}], **use_case_options)
    return _drain(fused_verdicts), use_case


def test_the_peers_gate_drops_a_link_far_slower_than_the_median_of_its_peers():
    mode = _FakeMode({1: _node(target_sample_rate_hz=10.0)})
    batch = {(f"t{index}", 1): _frames_at(100.0) for index in range(5)} | {("slow", 1): _frames_at(20.0)}
    fused, use_case = _fuse_once(mode, batch, rate_gate=RateGate.PEERS)
    assert _scored_link_keys(fused) == [(f"t{index}", 1) for index in range(5)]
    assert ("slow", 1) in [report.link_key for report in fused[0].link_reports]
    assert use_case.violation_counts["SampleRateMustKeepPaceWithPeers"] == 1
    assert use_case.violation_counts["SampleRateMustBeWithinTolerance"] == 0


def test_the_peers_gate_drops_nothing_when_every_link_runs_at_the_same_slow_rate():
    mode = _FakeMode({1: _node(target_sample_rate_hz=100.0)})
    batch = {(f"t{index}", 1): _frames_at(30.0) for index in range(5)}
    fused, use_case = _fuse_once(mode, batch, rate_gate=RateGate.PEERS)
    assert len(_scored_link_keys(fused)) == 5
    assert set(use_case.violation_counts.values()) == {0}


def test_the_target_gate_drops_links_running_below_their_heads_target():
    mode = _FakeMode({1: _node(target_sample_rate_hz=100.0)})
    batch = {(f"t{index}", 1): _frames_at(30.0) for index in range(5)}
    fused, use_case = _fuse_once(mode, batch)
    assert _scored_link_keys(fused) == []
    assert use_case.violation_counts["SampleRateMustBeWithinTolerance"] == 5
    assert use_case.violation_counts["SampleRateMustKeepPaceWithPeers"] == 0


def test_the_peers_gate_drops_nothing_with_a_single_measurable_link():
    mode = _FakeMode({1: _node(target_sample_rate_hz=100.0)})
    batch = {("a", 1): _frames_at(5.0), ("b", 1): _frames(0.0)}
    fused, use_case = _fuse_once(mode, batch, rate_gate=RateGate.PEERS)
    assert _scored_link_keys(fused) == [("a", 1)]
    assert set(use_case.violation_counts.values()) == {0}


def test_the_peers_gate_ignores_a_link_that_timed_out_of_the_vote():
    mode = _FakeMode({1: _node()})
    batches = [{("old", 1): _frames_at(1000.0)}, {}, {}, {}, {}, {("a", 1): _frames_at(100.0), ("b", 1): _frames_at(70.0)}, {}]
    fused_verdicts, _receiver, use_case = _iter_fused_with_use_case(mode, batches, step_s=0.7,
                                                                    rate_gate=RateGate.PEERS)
    fused = _drain(fused_verdicts)
    assert _scored_link_keys(fused[-1:]) == [("a", 1), ("b", 1)]
    assert use_case.violation_counts["SampleRateMustKeepPaceWithPeers"] == 0


def test_the_peers_gate_logs_a_warning_naming_the_delivered_and_needed_rate(caplog):
    mode = _FakeMode({1: _node()})
    batch = {("a", 1): _frames_at(100.0), ("b", 1): _frames_at(100.0), ("slow", 1): _frames_at(20.0)}
    fused_verdicts, _receiver, _use_case = _iter_fused_with_use_case(mode, [batch, {}, {}],
                                                                    rate_gate=RateGate.PEERS)
    with caplog.at_level(logging.WARNING, logger=MESH_LOGGER):
        _drain(fused_verdicts)
    assert _warnings(caplog) == ["SampleRateMustKeepPaceWithPeers violated on ('slow', 1): 20.0 Hz delivered, "
                                 "at least 80.0 Hz needed to keep pace with the peers "
                                 "(1 since the last report, 1 in total)"]


@pytest.mark.parametrize("rate_gate", [RateGate.TARGET, RateGate.PEERS])
def test_both_gates_share_the_minimum_rate_fraction(rate_gate):
    mode = _FakeMode({1: _node(target_sample_rate_hz=100.0)})
    batch = {("a", 1): _frames_at(100.0), ("b", 1): _frames_at(100.0), ("slow", 1): _frames_at(60.0)}
    dropped, _ = _fuse_once(mode, batch, rate_gate=rate_gate)
    kept, _ = _fuse_once(mode, batch, rate_gate=rate_gate, min_sample_rate_fraction=0.5)
    assert len(_scored_link_keys(dropped)) == 2
    assert len(_scored_link_keys(kept)) == 3


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_a_minimum_rate_fraction_outside_zero_to_one_is_refused(fraction):
    with pytest.raises(ValueError, match="min_sample_rate_fraction must be in"):
        ServeMeshUseCase(_FakeMode({}), clock=_FakeClock(), chunk_s=1.0, link_timeout_s=3.0, buffer_s=3.0,
                         min_sample_rate_fraction=fraction)


def test_an_unknown_rate_gate_is_refused():
    with pytest.raises(ValueError):
        ServeMeshUseCase(_FakeMode({}), clock=_FakeClock(), chunk_s=1.0, link_timeout_s=3.0, buffer_s=3.0,
                         rate_gate="median")
