"""Phase 2 — hardware ingest: FrameParser decode + NodeAggregator multi-node tagging/sync."""

import numpy as np
import pytest

from wavetrace import CsiFrame, FrameError, FrameParser, NodeAggregator
from fixtures.SyntheticCsi import encodeFrame, generateRawFrames


# --- FrameParser: decode correctness --------------------------------------------------------

def test_parse_round_trip_recovers_complex_grid():
    # Random int8 I/Q (incl. negatives) must decode exactly, exercising the v-=256 sign fixup.
    frames = generateRawFrames(numAntennas=3, numSubcarriers=64, numFrames=8, seed=11)
    parser = FrameParser(num_antennas=3, num_subcarriers=64)
    for raw, expected in frames:
        frame = parser.parse(raw, timestamp=1.0, node_id=0)
        assert frame.grid.dtype == np.complex64
        assert np.array_equal(frame.grid, expected)


def test_parse_sign_fixup_boundaries():
    # Wire byte 128 -> -128, 255 -> -1, 127 -> 127, 0 -> 0 (imag first, then real).
    raw = encodeFrame(realIQ=[[127, 0]], imagIQ=[[-128, -1]])
    parser = FrameParser(num_antennas=1, num_subcarriers=2)
    grid = parser.parse(raw).grid
    assert grid[0, 0] == 127.0 - 128.0j
    assert grid[0, 1] == 0.0 - 1.0j


def test_parse_stamps_timestamp_and_node_id():
    raw, _ = generateRawFrames(numAntennas=2, numSubcarriers=8, numFrames=1, seed=1)[0]
    parser = FrameParser(num_antennas=2, num_subcarriers=8)
    frame = parser.parse(raw, timestamp=3.5, node_id=4)
    assert frame.timestamp == 3.5
    assert frame.node_id == 4
    assert frame.num_antennas == 2 and frame.num_subcarriers == 8


def test_parse_length_mismatch_raises_frameerror():
    parser = FrameParser(num_antennas=1, num_subcarriers=4)  # expects 2*1*4 = 8 bytes
    with pytest.raises(FrameError):
        parser.parse(np.zeros(6, dtype=np.uint8))  # truncated packet -> error, never a panic


def test_parse_reuses_buffer_across_calls():
    # Same reused CsiFrame is returned each call (zero per-frame alloc) and overwritten in place.
    frames = generateRawFrames(numAntennas=1, numSubcarriers=4, numFrames=2, seed=2)
    parser = FrameParser(num_antennas=1, num_subcarriers=4)
    first = parser.parse(frames[0][0])
    second = parser.parse(frames[1][0])
    assert first is second  # reference_internal -> identical Python object
    assert np.array_equal(second.grid, frames[1][1])  # now holds the second frame's data


@pytest.mark.parametrize("numAntennas,numSubcarriers", [(1, 1), (1, 64), (64, 1), (2, 30)])
def test_parse_arbitrary_geometry(numAntennas, numSubcarriers):
    # 1x1, 1xN, Nx1, MxN — link count stays open; the parser must handle any geometry.
    raw, expected = generateRawFrames(
        numAntennas=numAntennas, numSubcarriers=numSubcarriers, numFrames=1, seed=3
    )[0]
    parser = FrameParser(num_antennas=numAntennas, num_subcarriers=numSubcarriers)
    frame = parser.parse(raw)
    assert frame.grid.shape == (numAntennas, numSubcarriers)
    assert np.array_equal(frame.grid, expected)


# --- NodeAggregator: multi-node tagging + time-sync -----------------------------------------

def _frameWith(node_id, timestamp, value):
    frame = CsiFrame(num_antennas=1, num_subcarriers=2)
    frame.node_id = node_id
    frame.timestamp = timestamp
    frame.grid[:, :] = value
    return frame


def test_aggregator_tags_and_keeps_latest_per_node():
    agg = NodeAggregator()
    agg.submit(_frameWith(0, 1.00, 1 + 0j))
    agg.submit(_frameWith(1, 1.01, 2 + 0j))
    agg.submit(_frameWith(0, 1.02, 9 + 0j))  # overwrites node 0's latest
    assert agg.num_nodes == 2
    synced = {int(f.node_id): f for f in agg.synced(tolerance=0.1)}
    assert synced[0].grid[0, 0] == 9 + 0j  # newest for node 0
    assert synced[1].grid[0, 0] == 2 + 0j


def test_aggregator_synced_drops_stale_nodes():
    agg = NodeAggregator()
    agg.submit(_frameWith(0, 10.00, 1))
    agg.submit(_frameWith(1, 10.01, 1))
    agg.submit(_frameWith(2, 10.50, 1))  # newest; node 0/1 are 0.49-0.50 s behind
    in_sync = agg.synced(tolerance=0.05)
    assert {int(f.node_id) for f in in_sync} == {2}
    assert {int(f.node_id) for f in agg.synced(tolerance=1.0)} == {0, 1, 2}
