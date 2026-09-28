"""Per-link weapon heads and link health metrics."""

from types import SimpleNamespace

import numpy as np

from wavetrace.recognition import find_weapon_entry, link_health, make_link_tag, parse_link_tag


def test_link_tag_parses_tx_from_dataset_dir():
    """Groups per direction by link suffix."""
    assert parse_link_tag("data/weapon_ds/node2/p0_na_s0_clear_link4f9c") == "4f9c"
    assert parse_link_tag("p1_chest_s3_weapon_link64b8") == "64b8"
    assert parse_link_tag("legacy_node_pool_dir") is None  # pre-per-link dirs -> no tag


def test_link_tag_written_at_collection_is_the_key_live_serving_looks_up():
    tag = make_link_tag("4f:9c")
    assert parse_link_tag(f"data/weapon_ds/node2/p0_na_s0_clear_link{tag}") == tag
    assert find_weapon_entry({(tag, 2): "LINK"}, "4f:9c", 2) == "LINK"


def test_find_weapon_entry_prefers_per_link_then_falls_back_to_node():
    entries_by_link = {("4f9c", 2): "LINK", (None, 2): "NODE"}
    assert find_weapon_entry(entries_by_link, "4f:9c", 2) == "LINK"   # per-link wins
    assert find_weapon_entry(entries_by_link, "64:b8", 2) == "NODE"   # unknown direction -> node fallback
    assert find_weapon_entry({(None, 3): "NODE3"}, "aa:bb", 3) == "NODE3"  # pure per-node layout
    assert find_weapon_entry(entries_by_link, "4f:9c", 9) is None     # unknown RX node


def _frames(timestamps):
    return [SimpleNamespace(timestamp=float(t)) for t in timestamps]


def test_link_health_clean_stream_no_missing():
    """100 Hz stream reports ~100 Hz delivered and ~0 missing."""
    hz, miss = link_health(_frames(np.arange(50) * 0.01))
    assert abs(hz - 100.0) < 1.0
    assert miss == 0.0


def test_link_health_detects_dropped_frames():
    ts = list(np.arange(20) * 0.01)
    del ts[10]; del ts[5]  # two single-frame drops
    hz, miss = link_health(_frames(ts))
    assert miss > 0.0
    assert hz > 0.0


def test_link_health_too_few_frames():
    assert link_health(_frames([0.0, 0.01])) == (0.0, 0.0)
