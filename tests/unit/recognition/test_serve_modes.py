from types import SimpleNamespace

import numpy as np
import pytest

from wavetrace.application.serve import WindowView
from wavetrace.recognition import serve_modes
from wavetrace.recognition.serve_modes import CountServeMode, PresenceServeMode, WeaponServeMode
from wavetrace.recognition.serving import discover_node_dirs

WINDOW_VIEW = WindowView(features=np.zeros(9), image=np.zeros((1, 4)), intercarrier_features=None)


def _node(**fields):
    session = SimpleNamespace(head=SimpleNamespace(contract=SimpleNamespace(target_sample_rate_hz=100.0)))
    return SimpleNamespace(**{"session": session, "calibration": None, "gain_lock": None, "config": None,
                              "should_use_intercarrier": False, "pick_input": None, "weight": 1.0, **fields})


def _frames(*timestamps):
    return [SimpleNamespace(timestamp=timestamp) for timestamp in timestamps]


def _serve_last_window_as(monkeypatch, proba):
    monkeypatch.setattr(serve_modes, "last_window_proba", lambda *args: (np.array(proba), WINDOW_VIEW))


def _serve_dwell_as(monkeypatch, proba):
    monkeypatch.setattr(serve_modes, "dwell_proba", lambda *args: (np.array(proba), WINDOW_VIEW))


def test_discover_node_dirs_needs_a_numbered_model_dir_and_its_calibration(tmp_path):
    for name in ("node1", "node2", "node10", "node3", "nodeX"):
        (tmp_path / "model" / name).mkdir(parents=True)
    for name in ("node1", "node2", "node10", "nodeX"):
        (tmp_path / "cal" / name).mkdir(parents=True)
    found = list(discover_node_dirs(str(tmp_path / "cal"), str(tmp_path / "model")))
    assert [(node_id, model_dir.endswith(f"node{node_id}"), calibration_dir.endswith(f"node{node_id}"))
            for node_id, model_dir, calibration_dir in found] == [(1, True, True), (10, True, True),
                                                                  (2, True, True)]


def test_presence_scores_the_binary_margin(monkeypatch):
    _serve_last_window_as(monkeypatch, [0.2, 0.8])
    node = _node(positive_class_index=1)
    score = PresenceServeMode({1: node}).score_link(("4f:9c", 1), node, _frames(0.0, 0.01))
    assert score.quality == pytest.approx(0.6)
    assert score.breakdown_token == "4f:9c->1:0.80"
    assert list(score.vote_proba) == [0.2, 0.8]


def test_presence_verdict_labels_and_bars_the_present_proba():
    mode = PresenceServeMode({1: _node(positive_class_index=1)})
    assert mode.format_verdict(np.array([0.25, 0.75], dtype=np.float32)) == "PRESENT  P 0.75  " + "#" * 15 + " " * 5
    assert mode.format_verdict(np.array([0.75, 0.25], dtype=np.float32)) == "absent   P 0.25  " + "#" * 5 + " " * 15


def test_presence_without_a_positive_class_reads_zero():
    mode = PresenceServeMode({1: _node(positive_class_index=-1)})
    assert mode.format_verdict(np.array([0.1, 0.9])) == "absent   P 0.00  " + " " * 20


def test_count_expands_into_global_classes_and_scores_the_top_two_margin(monkeypatch):
    _serve_last_window_as(monkeypatch, [0.3, 0.7])
    node = _node(global_class_columns=[0, 2])
    score = CountServeMode({1: node}, [0, 1, 2], max_count=2).score_link(("4f:9c", 1), node, _frames(0.0, 0.01))
    assert list(score.vote_proba) == [0.3, 0.0, 0.7]
    assert score.quality == pytest.approx(0.4)
    assert score.breakdown_token == "4f:9c->1:2"


def test_count_single_class_head_scores_its_only_proba(monkeypatch):
    _serve_last_window_as(monkeypatch, [1.0])
    node = _node(global_class_columns=[1])
    score = CountServeMode({1: node}, [0, 1], max_count=3).score_link(("4f:9c", 1), node, _frames(0.0, 0.01))
    assert score.quality == 1.0 and score.breakdown_token == "4f:9c->1:1"


def test_count_verdict_is_the_expected_count_with_the_top_bin_open_ended():
    mode = CountServeMode({1: _node()}, [0, 1, 2], max_count=2)
    assert mode.format_verdict(np.array([0.0, 0.5, 0.5], dtype=np.float32)) == "PEOPLE ~1.5  (2+)"
    assert mode.format_verdict(np.array([0.75, 0.25, 0.0], dtype=np.float32)) == "PEOPLE ~0.2  (0)"


def test_weapon_prefers_the_per_link_head_then_the_node_head():
    per_link, per_node = _node(), _node()
    mode = WeaponServeMode({("4f9c", 1): per_link, (None, 1): per_node})
    assert mode.select_node(("4f:9c", 1)) is per_link
    assert mode.select_node(("64:b8", 1)) is per_node
    assert mode.select_node(("64:b8", 2)) is None


def test_weapon_token_carries_the_link_health(monkeypatch):
    _serve_dwell_as(monkeypatch, [0.1, 0.9])
    node = _node(positive_class_index=1)
    mode = WeaponServeMode({("4f9c", 1): node})
    score = mode.score_link(("4f:9c", 1), node, _frames(0.0, 0.01, 0.02, 0.05, 0.06))
    assert score.quality == pytest.approx(0.8)
    assert score.breakdown_token == "4f:9c->1:0.90@67Hz!33%drop"
    assert mode.score_link(("4f:9c", 1), node, _frames(0.0, 0.01, 0.02)).breakdown_token == "4f:9c->1:0.90@100Hz"


def test_weapon_verdict_and_per_node_banner():
    mode = WeaponServeMode({(None, 1): _node(positive_class_index=1)})
    mode.nodes[(None, 1)].weight = 0.5
    assert mode.format_verdict(np.array([0.75, 0.25], dtype=np.float32)) == "clear   P 0.25  " + "#" * 5 + " " * 15
    assert mode.format_banner(5599) == ("WEAPON detection on udp/5599 (fs=100Hz, per-node heads; "
                                        "vote weights *->1:w=0.50). Ctrl+C to stop.\n")


def test_weapon_banner_renders_a_root_mixing_per_link_and_per_node_heads():
    mode = WeaponServeMode({("4f9c", 1): _node(), (None, 2): _node()})
    banner = mode.format_banner(5599)
    assert "per-link heads" in banner
    assert banner.index("*->2") < banner.index("4f9c")


def test_presence_exposes_its_class_ids_calibrations_and_the_scored_window(monkeypatch):
    _serve_last_window_as(monkeypatch, [0.2, 0.8])
    nodes = {1: _node(classes=[0, 1], calibration="cal1", positive_class_index=1),
             2: _node(classes=[0, 1], calibration="cal2", positive_class_index=1)}
    mode = PresenceServeMode(nodes)
    assert mode.class_ids == [0, 1]
    assert mode.calibration_by_rx_node == {1: "cal1", 2: "cal2"}
    assert mode.score_link(("4f:9c", 1), nodes[1], _frames(0.0, 0.01)).window_view is WINDOW_VIEW


def test_count_exposes_the_global_class_ids_and_calibrations(monkeypatch):
    _serve_last_window_as(monkeypatch, [0.3, 0.7])
    node = _node(global_class_columns=[0, 2], calibration="cal1")
    mode = CountServeMode({1: node}, [0, 1, 2], max_count=2)
    assert mode.class_ids == [0, 1, 2]
    assert mode.calibration_by_rx_node == {1: "cal1"}
    assert mode.score_link(("4f:9c", 1), node, _frames(0.0, 0.01)).window_view is WINDOW_VIEW


def test_count_without_max_count_opens_the_top_learned_class():
    mode = CountServeMode({1: _node()}, [0, 1, 2])
    assert mode.format_verdict(np.array([0.0, 0.0, 1.0], dtype=np.float32)) == "PEOPLE ~2.0  (2+)"


def test_weapon_exposes_class_ids_and_one_calibration_per_rx_node(monkeypatch):
    _serve_dwell_as(monkeypatch, [0.1, 0.9])
    per_link = _node(classes=[0, 1], calibration="cal1", positive_class_index=1)
    mode = WeaponServeMode({("4f9c", 1): per_link, ("64b8", 1): _node(classes=[0, 1], calibration="cal1"),
                            (None, 2): _node(classes=[0, 1], calibration="cal2")})
    assert mode.class_ids == [0, 1]
    assert mode.calibration_by_rx_node == {1: "cal1", 2: "cal2"}
    assert mode.score_link(("4f:9c", 1), per_link, _frames(0.0, 0.01, 0.02)).window_view is WINDOW_VIEW
