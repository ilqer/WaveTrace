"""All-pairs link splitting: parse_batch_links buckets one batch by (tx_short, rx_node)."""

import json

import numpy as np
import pytest

from wavetrace.Source import parse_batch_links, mac_short


def _csi_line(csi_ints, mac, local_ts):
    """Well-formed 25-column esp-csi CSV line (mirrors TestUdpSource._make_csi_line)."""
    cols = [
        "CSI_DATA", "1", mac, "-60", "11", "0", "7", "0",
        "1", "0", "0", "0", "0", "0",
        "-95", "0", "6", "0", str(local_ts), "0", "100", "0",
        str(len(csi_ints) // 2), "1", '"' + json.dumps(csi_ints, separators=(",", ":")) + '"',
    ]
    assert len(cols) == 25
    return ",".join(cols)


def _batch(rows, node_id=7, ntp_ms=5000):
    """rows = [(csi_ints, mac, local_ts), ...] -> one UDP batch payload (header + lines)."""
    header = json.dumps({"v": 1, "node": node_id, "ntp_ms": ntp_ms, "n": len(rows)})
    return ("\n".join([header] + [_csi_line(*r) for r in rows])).encode("utf-8")


def test_mac_short():
    assert mac_short("aa:bb:cc:dd:ee:ff") == "ee:ff"
    assert mac_short("no_colons") == "no_colons"


def test_splits_two_transmitters_into_two_links():
    """Two interleaved TX MACs to one RX -> two (tx,rx) buckets, frames routed by MAC."""
    a, b = "aa:aa:aa:aa:00:01", "bb:bb:bb:bb:00:02"
    S = 4
    rows = [
        ([1, 2] * S, a, 1000),
        ([3, 4] * S, b, 1010),
        ([5, 6] * S, a, 2000),
        ([7, 8] * S, b, 2010),
        ([9, 9] * S, a, 3000),
    ]
    links = parse_batch_links(_batch(rows, node_id=7))
    assert set(links) == {("00:01", 7), ("00:02", 7)}
    assert len(links[("00:01", 7)]) == 3   # three frames from TX a
    assert len(links[("00:02", 7)]) == 2   # two frames from TX b
    for frames in links.values():
        assert all(fr.node_id == 7 and fr.num_subcarriers == S for fr in frames)


def test_timestamps_match_parse_batch_scheme():
    """Per-frame t = ntp_ms/1000 - (last_us - local_ts)/1e6, last_us = last parsed line overall."""
    a = "aa:aa:aa:aa:00:01"
    ntp_ms = 5000
    rows = [([1, 2, 3, 4], a, 1000), ([5, 6, 7, 8], a, 1010), ([9, 9, 9, 9], a, 1020)]
    links = parse_batch_links(_batch(rows, node_id=3, ntp_ms=ntp_ms))
    frames = links[("00:01", 3)]
    last_us = 1020
    for fr, ts in zip(frames, [1000, 1010, 1020]):
        assert fr.timestamp == pytest.approx(ntp_ms / 1000.0 - (last_us - ts) / 1e6, abs=1e-9)


def test_tx_mac_filter_keeps_one_link():
    a, b = "aa:aa:aa:aa:00:01", "bb:bb:bb:bb:00:02"
    rows = [([1, 2, 3, 4], a, 1000), ([5, 6, 7, 8], b, 1010)]
    links = parse_batch_links(_batch(rows), tx_mac=a)
    assert set(links) == {("00:01", 7)}


def test_per_link_width_guard():
    """A link's first width sets its reference; off-width lines in THAT link are dropped — but a
    different link may legitimately carry a different width (e.g. a future 5 GHz arm)."""
    a, b = "aa:aa:aa:aa:00:01", "bb:bb:bb:bb:00:02"
    rows = [
        ([1, 2, 3, 4], a, 1000),          # link a: S=2 (reference)
        ([1, 2, 3, 4, 5, 6], a, 1010),    # link a: S=3 -> dropped
        ([1, 2, 3, 4, 5, 6], b, 1020),    # link b: S=3 (its own reference) -> kept
    ]
    links = parse_batch_links(_batch(rows))
    assert len(links[("00:01", 7)]) == 1 and links[("00:01", 7)][0].num_subcarriers == 2
    assert len(links[("00:02", 7)]) == 1 and links[("00:02", 7)][0].num_subcarriers == 3


def test_bad_header_raises_empty_returns_dict():
    with pytest.raises(ValueError, match="bad batch header"):
        parse_batch_links(b"not_json\nfoo")
    with pytest.raises(ValueError, match="bad batch header"):
        parse_batch_links(b"")
    header = json.dumps({"v": 1, "node": 0, "ntp_ms": 1000, "n": 0})
    assert parse_batch_links((header + "\nbad\nlines").encode()) == {}
