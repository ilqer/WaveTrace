"""Binary UDP wire format v4: UWB CIR records from the Portenta C33 + SR150 shield (little-endian)."""

import struct

import numpy as np

from wavetrace.domain.cir_record import CirRecord

_HEADER = struct.Struct("<BBBQH")   # magic, ver, node, ntp_ms, num_records -> 13 bytes, same as v2/v3
_MAGIC = 0x57
_VERSION = 4
# peer_short_addr u16, ts_us u32, num_rx u8, num_taps u16, tap_spacing_fs u32, window_start_index i16
_RECORD_HEAD = struct.Struct("<HIBHIh")
# distance_cm u16, aoa_azimuth_q9_7 i16, aoa_elevation_q9_7 i16, pdoa_q9_7 i16, pdoa_index u16
_RECORD_ANGLES = struct.Struct("<HhhhH")
_FEMTOSECONDS_PER_SECOND = 1e15
_Q9_7_SCALE = 128.0                  # signed fixed point degrees: value / 2^7
_CENTIMETRES_PER_METRE = 100.0
_INT16_LE = np.dtype("<i2")


def parse_cir_batch(packet: bytes) -> list[CirRecord]:
    """One v4 datagram -> its CirRecords. ValueError naming the field on a bad magic or version, a
    truncated header or record, num_rx == 0, num_taps == 0, or trailing bytes. O(total taps): one
    np.frombuffer view per record, no per-tap Python loop.

    Per record: peer_short_addr u16, ts_us u32, num_rx u8, num_taps u16, tap_spacing_fs u32 (fs),
    window_start_index i16, first_path_index i16[num_rx], distance_cm u16, aoa_azimuth/aoa_elevation/
    pdoa i16 Q9.7 degrees, pdoa_index u16, snr_first_path_db u8[num_rx], then taps i16[num_rx][num_taps]
    [2]. Each tap is REAL first, then imag; the ESP32 CSI buffer is imag first."""
    if len(packet) < _HEADER.size:
        raise ValueError(f"v4 header: {len(packet)} bytes < {_HEADER.size}")
    magic, version, node_id, ntp_ms, num_records = _HEADER.unpack_from(packet, 0)
    if magic != _MAGIC:
        raise ValueError(f"v4 header magic: {magic:#x} != {_MAGIC:#x}")
    if version != _VERSION:
        raise ValueError(f"v4 header version: {version} != {_VERSION}")
    offset = _HEADER.size
    records = []
    for record_number in range(num_records):
        record, offset = _parse_record(packet, offset, node_id, ntp_ms, record_number)
        records.append(record)
    if offset != len(packet):
        raise ValueError(f"v4 packet: {len(packet) - offset} trailing bytes after {num_records} records")
    return records


def _require(packet: bytes, offset: int, size: int, field: str, record_number: int) -> None:
    if offset + size > len(packet):
        raise ValueError(f"v4 record {record_number} {field}: needs {size} bytes at offset {offset}, "
                         f"packet has {len(packet) - offset}")


def _parse_record(packet, offset, node_id, ntp_ms, record_number):
    """-> (CirRecord, offset after it)."""
    _require(packet, offset, _RECORD_HEAD.size, "head", record_number)
    (peer_short_addr, timestamp_us, num_rx, num_taps, tap_spacing_fs,
     window_start_index) = _RECORD_HEAD.unpack_from(packet, offset)
    offset += _RECORD_HEAD.size
    if num_rx == 0:
        raise ValueError(f"v4 record {record_number} num_rx: must be > 0")
    if num_taps == 0:
        raise ValueError(f"v4 record {record_number} num_taps: must be > 0")

    _require(packet, offset, 2 * num_rx, "first_path_index", record_number)
    first_path_indices = np.frombuffer(packet, _INT16_LE, num_rx, offset).astype(np.int16)
    offset += 2 * num_rx

    _require(packet, offset, _RECORD_ANGLES.size, "distance_cm/aoa/pdoa", record_number)
    distance_cm, azimuth_q9_7, elevation_q9_7, pdoa_q9_7, pdoa_index = _RECORD_ANGLES.unpack_from(packet, offset)
    offset += _RECORD_ANGLES.size

    _require(packet, offset, num_rx, "snr_first_path_db", record_number)
    first_path_snr_db = np.frombuffer(packet, np.uint8, num_rx, offset).copy()
    offset += num_rx

    tap_bytes = 4 * num_rx * num_taps
    _require(packet, offset, tap_bytes, "taps", record_number)
    iq = np.frombuffer(packet, _INT16_LE, 2 * num_rx * num_taps, offset).reshape(num_rx, num_taps, 2)
    taps = np.empty((num_rx, num_taps), dtype=np.complex64)
    taps.real = iq[..., 0]
    taps.imag = iq[..., 1]
    offset += tap_bytes

    return CirRecord(
        node_id=node_id, ntp_ms=ntp_ms, peer_short_addr=peer_short_addr, timestamp_us=timestamp_us,
        tap_spacing_s=tap_spacing_fs / _FEMTOSECONDS_PER_SECOND, window_start_index=window_start_index,
        first_path_indices=first_path_indices, distance_m=distance_cm / _CENTIMETRES_PER_METRE,
        aoa_azimuth_deg=azimuth_q9_7 / _Q9_7_SCALE, aoa_elevation_deg=elevation_q9_7 / _Q9_7_SCALE,
        pdoa_deg=pdoa_q9_7 / _Q9_7_SCALE, pdoa_index=pdoa_index, first_path_snr_db=first_path_snr_db,
        taps=taps), offset
