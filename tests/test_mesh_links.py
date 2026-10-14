"""parse_batch_by_link buckets batches by (tx_short, rx_node)."""

import struct

import pytest

from wavetrace.adapters.source import parse_batch_by_link, short_mac


def _record(csi_values, mac, local_timestamp_us):
    """One binary v2 record: mac[6] | ts_us(u32 LE) | len(u16 LE) | int8 raw CSI."""
    mac_bytes = bytes(int(x, 16) for x in mac.split(":"))
    data = bytes(value & 0xFF for value in csi_values)
    return mac_bytes + struct.pack("<IH", local_timestamp_us, len(csi_values)) + data


def _batch(rows, node_id=7, ntp_ms=5000):
    """Converts [(csi_values, mac, local_timestamp_us)] into a binary v2 UDP batch payload with header."""
    header = struct.pack("<BBBQH", 0x57, 2, node_id, ntp_ms, len(rows))
    return header + b"".join(_record(*row) for row in rows)


def test_mac_short():
    assert short_mac("aa:bb:cc:dd:ee:ff") == "ee:ff"
    assert short_mac("no_colons") == "no_colons"


def test_splits_two_transmitters_into_two_links():
    """Interleaved frames from two TX MACs to one RX are routed into two distinct (tx,rx) buckets."""
    tx_mac_a, tx_mac_b = "aa:aa:aa:aa:00:01", "bb:bb:bb:bb:00:02"
    subcarrier_count = 4
    rows = [
        ([1, 2] * subcarrier_count, tx_mac_a, 1000),
        ([3, 4] * subcarrier_count, tx_mac_b, 1010),
        ([5, 6] * subcarrier_count, tx_mac_a, 2000),
        ([7, 8] * subcarrier_count, tx_mac_b, 2010),
        ([9, 9] * subcarrier_count, tx_mac_a, 3000),
    ]
    links = parse_batch_by_link(_batch(rows, node_id=7))
    assert set(links) == {("00:01", 7), ("00:02", 7)}
    assert len(links[("00:01", 7)]) == 3
    assert len(links[("00:02", 7)]) == 2
    for frames in links.values():
        assert all(frame.node_id == 7 and frame.num_subcarriers == subcarrier_count for frame in frames)


def test_timestamps_match_parse_batch_scheme():
    """Per-frame t = ntp_ms/1000 - (last_us - local_timestamp_us)/1e6, last_us = last parsed line overall."""
    tx_mac_a = "aa:aa:aa:aa:00:01"
    ntp_ms = 5000
    rows = [([1, 2, 3, 4], tx_mac_a, 1000), ([5, 6, 7, 8], tx_mac_a, 1010), ([9, 9, 9, 9], tx_mac_a, 1020)]
    links = parse_batch_by_link(_batch(rows, node_id=3, ntp_ms=ntp_ms))
    frames = links[("00:01", 3)]
    last_local_timestamp_us = 1020
    for frame, local_timestamp_us in zip(frames, [1000, 1010, 1020]):
        assert frame.timestamp == pytest.approx(
            ntp_ms / 1000.0 - (last_local_timestamp_us - local_timestamp_us) / 1e6, abs=1e-9)


def test_tx_mac_filter_keeps_one_link():
    tx_mac_a, tx_mac_b = "aa:aa:aa:aa:00:01", "bb:bb:bb:bb:00:02"
    rows = [([1, 2, 3, 4], tx_mac_a, 1000), ([5, 6, 7, 8], tx_mac_b, 1010)]
    links = parse_batch_by_link(_batch(rows), tx_mac=tx_mac_a)
    assert set(links) == {("00:01", 7)}


def test_per_link_width_guard():
    """First frame sets the expected subcarrier width for a link.
    Frames with mismatched widths in the same link are dropped.
    Different links can operate on different widths (e.g. 5GHz vs 2.4GHz)."""
    tx_mac_a, tx_mac_b = "aa:aa:aa:aa:00:01", "bb:bb:bb:bb:00:02"
    rows = [
        ([1, 2, 3, 4], tx_mac_a, 1000),          # link a: S=2 (reference)
        ([1, 2, 3, 4, 5, 6], tx_mac_a, 1010),    # link a: S=3 -> dropped
        ([1, 2, 3, 4, 5, 6], tx_mac_b, 1020),    # link b: S=3 (its own reference) -> kept
    ]
    links = parse_batch_by_link(_batch(rows))
    assert len(links[("00:01", 7)]) == 1 and links[("00:01", 7)][0].num_subcarriers == 2
    assert len(links[("00:02", 7)]) == 1 and links[("00:02", 7)][0].num_subcarriers == 3


def test_bad_header_raises_empty_returns_dict():
    with pytest.raises(ValueError, match="bad batch header"):
        parse_batch_by_link(b"not_json\nfoo")
    with pytest.raises(ValueError, match="bad batch header"):
        parse_batch_by_link(b"")
    # Valid header, no complete records (trailing garbage) -> empty dict, no error
    valid_header = struct.pack("<BBBQH", 0x57, 2, 0, 1000, 0)
    assert parse_batch_by_link(valid_header + b"\x01\x02bad") == {}
