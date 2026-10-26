import struct

import numpy as np
import pytest

from wavetrace.adapters.source.uwb_wire_format import parse_cir_batch


def _packet_record(*, peer=0x1234, timestamp_us=4_000_000_000, tap_spacing_fs=1_000_000, window_start=-3,
                   first_path=(10, 12), distance_cm=250, azimuth_q=-6400, elevation_q=64, pdoa_q=192,
                   pdoa_index=7, snr=(33, 28), iq=None, num_taps=4):
    """One v4 record; `iq` is the (R, N, 2) int16 tap array, real first."""
    num_rx = len(first_path)
    if iq is None:
        iq = np.arange(num_rx * num_taps * 2, dtype=np.int16).reshape(num_rx, num_taps, 2) - 5
    return (struct.pack("<HIBHIh", peer, timestamp_us, num_rx, iq.shape[1], tap_spacing_fs, window_start)
            + struct.pack(f"<{num_rx}h", *first_path)
            + struct.pack("<HhhhH", distance_cm, azimuth_q, elevation_q, pdoa_q, pdoa_index)
            + bytes(snr) + iq.astype("<i2").tobytes())


def _packet(records, *, node=5, ntp_ms=1_700_000_000_123, magic=0x57, version=4):
    return struct.pack("<BBBQH", magic, version, node, ntp_ms, len(records)) + b"".join(records)


def test_single_record_exact_fields():
    iq = np.array([[[1, -2], [3, 4], [-5, 6], [32767, -32768]],
                   [[7, 8], [-9, -10], [11, 12], [0, 1]]], dtype=np.int16)
    [record] = parse_cir_batch(_packet([_packet_record(iq=iq)]))

    assert record.node_id == 5
    assert record.ntp_ms == 1_700_000_000_123
    assert record.peer_short_addr == 0x1234
    assert record.timestamp_us == 4_000_000_000
    assert record.tap_spacing_s == pytest.approx(1e-9)
    assert record.window_start_index == -3
    assert record.first_path_indices.dtype == np.int16
    np.testing.assert_array_equal(record.first_path_indices, [10, 12])
    assert record.distance_m == pytest.approx(2.5)
    assert record.aoa_azimuth_deg == -50.0       # -6400 / 128
    assert record.aoa_elevation_deg == 0.5       # 64 / 128
    assert record.pdoa_deg == 1.5                # 192 / 128
    assert record.pdoa_index == 7
    assert record.first_path_snr_db.dtype == np.uint8
    np.testing.assert_array_equal(record.first_path_snr_db, [33, 28])
    assert record.taps.dtype == np.complex64
    assert record.taps.shape == (2, 4)
    # real first, imag second
    assert record.taps[0, 0] == complex(1, -2)
    assert record.taps[0, 3] == complex(32767, -32768)
    assert record.taps[1, 2] == complex(11, 12)
    np.testing.assert_array_equal(record.first_path_columns, [13, 15])
    assert record.first_path_columns.dtype == np.int32


def test_two_records_in_arrival_order():
    first = _packet_record(peer=1, timestamp_us=10, first_path=(4,), snr=(9,), num_taps=3)
    second = _packet_record(peer=2, timestamp_us=20, tap_spacing_fs=2_000_000, num_taps=5)
    records = parse_cir_batch(_packet([first, second], node=9))

    assert [record.peer_short_addr for record in records] == [1, 2]
    assert [record.node_id for record in records] == [9, 9]
    assert records[0].taps.shape == (1, 3)
    assert records[1].taps.shape == (2, 5)
    assert records[1].tap_spacing_s == pytest.approx(2e-9)


def test_empty_batch_is_no_records():
    assert parse_cir_batch(_packet([])) == []


def test_first_path_columns_do_not_wrap_int16():
    [record] = parse_cir_batch(_packet([_packet_record(window_start=-32768, first_path=(32767, 0))]))
    np.testing.assert_array_equal(record.first_path_columns, [65535, 32768])


@pytest.mark.parametrize(("packet", "field"), [
    (b"\x57\x04\x01", "header"),
    (_packet([], magic=0x58), "magic"),
    (_packet([], version=3), "version"),
])
def test_bad_header_names_field(packet, field):
    with pytest.raises(ValueError, match=field):
        parse_cir_batch(packet)


def test_zero_num_rx_rejected():
    record = struct.pack("<HIBHIh", 1, 2, 0, 4, 1_000_000, 0)
    with pytest.raises(ValueError, match="num_rx"):
        parse_cir_batch(_packet([record]))


def test_zero_num_taps_rejected():
    record = struct.pack("<HIBHIh", 1, 2, 2, 0, 1_000_000, 0)
    with pytest.raises(ValueError, match="num_taps"):
        parse_cir_batch(_packet([record]))


# record bytes for the default 2-RX, 4-tap record: head 15, first_path_index 4, angles 10, snr 2, taps 32
@pytest.mark.parametrize(("kept_bytes", "field"), [
    (5, "head"),
    (15 + 1, "first_path_index"),
    (15 + 4 + 3, "distance_cm"),
    (15 + 4 + 10 + 1, "snr_first_path_db"),
    (15 + 4 + 10 + 2 + 31, "taps"),
])
def test_truncated_record_names_field(kept_bytes, field):
    with pytest.raises(ValueError, match=field):
        parse_cir_batch(_packet([_packet_record()[:kept_bytes]]))


def test_header_count_beyond_data_rejected():
    packet = _packet([_packet_record()])
    packet = packet[:12] + struct.pack("<H", 2) + packet[14:]
    with pytest.raises(ValueError, match="record 1"):
        parse_cir_batch(packet)


def test_trailing_bytes_rejected():
    with pytest.raises(ValueError, match="trailing"):
        parse_cir_batch(_packet([_packet_record()]) + b"\x00")
