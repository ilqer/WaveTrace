import numpy as np
import pytest

from wavetrace import CsiFrame, FrameError, FrameParser, NodeAggregator
from fixtures.synthetic_csi import encode_frame, generate_raw_frames


def test_parse_round_trip_recovers_complex_grid():
    # Random int8 I/Q must decode exactly (exercises sign fixup).
    frames = generate_raw_frames(num_antennas=3, num_subcarriers=64, num_frames=8, seed=11)
    parser = FrameParser(num_antennas=3, num_subcarriers=64)
    for raw, expected in frames:
        frame = parser.parse(raw, timestamp=1.0, node_id=0)
        assert frame.grid.dtype == np.complex64
        assert np.array_equal(frame.grid, expected)


def test_parse_sign_fixup_boundaries():
    # Wire byte boundaries: 128->-128, 255->-1, 127->127, 0->0.
    raw = encode_frame(real_grid=[[127, 0]], imaginary_grid=[[-128, -1]])
    parser = FrameParser(num_antennas=1, num_subcarriers=2)
    grid = parser.parse(raw).grid
    assert grid[0, 0] == 127.0 - 128.0j
    assert grid[0, 1] == 0.0 - 1.0j


def test_parse_stamps_timestamp_and_node_id():
    raw, _ = generate_raw_frames(num_antennas=2, num_subcarriers=8, num_frames=1, seed=1)[0]
    parser = FrameParser(num_antennas=2, num_subcarriers=8)
    frame = parser.parse(raw, timestamp=3.5, node_id=4)
    assert frame.timestamp == 3.5
    assert frame.node_id == 4
    assert frame.num_antennas == 2 and frame.num_subcarriers == 8


def test_parse_length_mismatch_raises_frameerror():
    parser = FrameParser(num_antennas=1, num_subcarriers=4)  # Expects 8 bytes.
    with pytest.raises(FrameError):
        parser.parse(np.zeros(6, dtype=np.uint8))


def test_parse_reuses_buffer_across_calls():
    # The same CsiFrame is reused each call: no per-frame allocation.
    frames = generate_raw_frames(num_antennas=1, num_subcarriers=4, num_frames=2, seed=2)
    parser = FrameParser(num_antennas=1, num_subcarriers=4)
    first = parser.parse(frames[0][0])
    second = parser.parse(frames[1][0])
    assert first is second
    assert np.array_equal(second.grid, frames[1][1])


@pytest.mark.parametrize("num_antennas,num_subcarriers", [(1, 1), (1, 64), (64, 1), (2, 30)])
def test_parse_arbitrary_geometry(num_antennas, num_subcarriers):
    raw, expected = generate_raw_frames(
        num_antennas=num_antennas, num_subcarriers=num_subcarriers, num_frames=1, seed=3
    )[0]
    parser = FrameParser(num_antennas=num_antennas, num_subcarriers=num_subcarriers)
    frame = parser.parse(raw)
    assert frame.grid.shape == (num_antennas, num_subcarriers)
    assert np.array_equal(frame.grid, expected)


def _make_frame(node_id, timestamp, value):
    frame = CsiFrame(num_antennas=1, num_subcarriers=2)
    frame.node_id = node_id
    frame.timestamp = timestamp
    frame.grid[:, :] = value
    return frame


def test_aggregator_tags_and_keeps_latest_per_node():
    aggregator = NodeAggregator()
    aggregator.submit(_make_frame(0, 1.00, 1 + 0j))
    aggregator.submit(_make_frame(1, 1.01, 2 + 0j))
    aggregator.submit(_make_frame(0, 1.02, 9 + 0j))
    assert aggregator.num_nodes == 2
    latest_by_node_id = {int(frame.node_id): frame for frame in aggregator.synced(tolerance=0.1)}
    assert latest_by_node_id[0].grid[0, 0] == 9 + 0j
    assert latest_by_node_id[1].grid[0, 0] == 2 + 0j


def test_aggregator_synced_drops_stale_nodes():
    aggregator = NodeAggregator()
    aggregator.submit(_make_frame(0, 10.00, 1))
    aggregator.submit(_make_frame(1, 10.01, 1))
    aggregator.submit(_make_frame(2, 10.50, 1))  # Newest.
    synced_frames = aggregator.synced(tolerance=0.05)
    assert {int(frame.node_id) for frame in synced_frames} == {2}
    assert {int(frame.node_id) for frame in aggregator.synced(tolerance=1.0)} == {0, 1, 2}
