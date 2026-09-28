"""Decoders for the two CSI wire formats: esp-csi CSV lines (serial) and binary UDP batches."""

import json
import struct

import numpy as np

from wavetrace import CsiFrame
from wavetrace.domain.link_key import LinkKey

# esp-csi 25-column CSV header (data = last column, JSON array of ints).
ESP_CSI_COLUMNS = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
    "ant", "sig_len", "rx_state", "len", "first_word", "data",
]  # 25 columns; data is a JSON array at index 24


def parse_csi_line(line, *, tx_mac=None):
    """One esp-csi CSV line -> (csi (S,) complex64, local_timestamp_us int, mac str), or None if
    malformed/filtered. Never raises on a bad line. O(S).

    I/Q pairing: esp-csi stores [imag, real] pairs — csi[k] = complex(data[2k+1], data[2k]).
    tx_mac filter: if set, lines whose MAC != tx_mac (case-insensitive) are silently dropped (removes
    beacons / foreign traffic — CSI must come from the dedicated TX only)."""
    try:
        # maxsplit=24 protects the JSON data array's internal commas
        parts = line.strip().split(",", 24)
        if len(parts) != 25 or parts[0] != "CSI_DATA":
            return None
        mac = parts[2]
        if tx_mac is not None and mac.lower() != tx_mac.lower():
            return None
        local_timestamp_us = int(parts[18])
        # real esp-csi wraps the array in CSV double-quotes ("[...]"); strip them (unquoted also works)
        data = json.loads(parts[24].strip().strip('"'))
        if len(data) % 2 != 0:
            return None
        num_subcarriers = len(data) // 2
        csi = np.empty(num_subcarriers, dtype=np.complex64)
        for k in range(num_subcarriers):
            csi[k] = complex(data[2 * k + 1], data[2 * k])  # [imag, real] pairing
        return csi, local_timestamp_us, mac
    except Exception:
        return None


# Binary UDP wire format (little-endian). Header then packed records. ver 2 = int8 I/Q (ESP32 nodes),
# ver 3 = int16 I/Q (Pi 5 GHz node: keeps the full amplitude range the weapon σ² feature needs).
_BIN_HDR = struct.Struct("<BBBQH")   # magic, ver, node, ntp_ms, n  -> 13 bytes
_BIN_MAGIC = 0x57
_BIN_VERS = (2, 3)


def _parse_batch_header(payload: bytes):
    """(node_id, ntp_ms, num_records, version, body_offset) from a binary batch header; ValueError on a
    bad header. `num_records` is the authoritative record count — the parser trusts it over the raw
    byte length."""
    if len(payload) < _BIN_HDR.size:
        raise ValueError(f"UdpSource: bad batch header: {len(payload)} bytes < {_BIN_HDR.size}")
    magic, version, node_id, ntp_ms, num_records = _BIN_HDR.unpack_from(payload, 0)
    if magic != _BIN_MAGIC or version not in _BIN_VERS:
        raise ValueError(f"UdpSource: bad batch header: magic={magic:#x} ver={version}")
    return node_id, ntp_ms, num_records, version, _BIN_HDR.size


def _mac_to_bytes(mac: str) -> bytes:
    """'aa:bb:cc:dd:ee:ff' -> 6 raw bytes (case-insensitive). Normalize a tx_mac filter once."""
    return bytes(int(octet, 16) for octet in mac.split(":"))


def _iter_batch_records(payload: bytes, offset: int, num_records: int, version: int = 2):
    """Yield (mac_bytes, csi complex64, local_timestamp_us) for up to `num_records` packed records — the
    header count bounds the loop, so trailing bytes past `num_records` records are never interpreted as
    CSI. Stops early on a structurally corrupt record (truncated, or wrong byte length) since the stream
    can't be resynced.
    Record = mac[6] | ts_us(u32 LE) | len(u16 LE) | CSI bytes, laid out [imag0,real0,...]. CSI is
    int8 per component (ver 2, csi_length_bytes=2*S) or int16 (ver 3, csi_length_bytes=4*S).
    Yields the raw 6 MAC bytes (not a formatted string) so the hot path skips per-record formatting;
    callers compare bytes for the tx_mac filter and only stringify when bucketing/logging."""
    end = len(payload)
    bytes_per_subcarrier = 2 if version == 2 else 4   # bytes per subcarrier: int8 I/Q (2) vs int16 I/Q (4)
    for _ in range(num_records):
        if offset + 12 > end:                     # 6 (mac) + 4 (ts) + 2 (len): batch shorter than num_records
            break
        mac = payload[offset:offset + 6]
        local_timestamp_us, csi_length_bytes = struct.unpack_from("<IH", payload, offset + 6)
        offset += 12
        if csi_length_bytes % bytes_per_subcarrier != 0 or offset + csi_length_bytes > end:
            break
        if version == 2:
            iq_components = np.frombuffer(payload, dtype=np.int8, count=csi_length_bytes,
                                          offset=offset).astype(np.float32)
        else:
            iq_components = np.frombuffer(payload, dtype="<i2", count=csi_length_bytes // 2,
                                          offset=offset).astype(np.float32)
        offset += csi_length_bytes
        # csi[k] = complex(real=iq_components[2k+1], imag=iq_components[2k])
        csi = (iq_components[1::2] + 1j * iq_components[0::2]).astype(np.complex64)
        yield mac, csi, local_timestamp_us


def parse_batch(payload: bytes, *, tx_mac=None) -> list:
    """One UDP batch payload -> list[CsiFrame] (node_id + wall-clock timestamps from the header).
    O(n·S).

    Binary v2 wire format: header csi_hdr_t {magic,ver=2,node,ntp_ms,n} then exactly n packed records
    (see _iter_batch_records). A bad/missing header raises ValueError (wiring error). The header's n
    bounds parsing, so trailing bytes are ignored; a structurally corrupt record stops parsing. Records
    whose subcarrier count S differs from the first kept one are skipped (width guard).

    Timestamp scheme: ntp_ms/1000.0 is the batch SEND time ≈ the LAST frame's wall time; each
    frame's absolute time is reconstructed from its local_timestamp offset relative to the last
    record's local_ts:  t_i = ntp_ms/1000 - (last_us - local_ts_i) / 1e6."""
    node_id, ntp_ms, num_records, version, body_offset = _parse_batch_header(payload)
    tx_mac_bytes = _mac_to_bytes(tx_mac) if tx_mac is not None else None

    parsed = []
    expected_num_subcarriers = None
    for mac_bytes, csi, local_timestamp_us in _iter_batch_records(payload, body_offset, num_records, version):
        if tx_mac_bytes is not None and mac_bytes != tx_mac_bytes:
            continue
        num_subcarriers = int(csi.size)
        if expected_num_subcarriers is None:
            expected_num_subcarriers = num_subcarriers
        elif num_subcarriers != expected_num_subcarriers:
            continue  # mixed-S record: skip
        parsed.append((csi, local_timestamp_us))

    if not parsed:
        return []

    last_local_timestamp_us = parsed[-1][1]
    frames = []
    for csi, local_timestamp_us in parsed:
        # & 0xFFFFFFFF: masks the firmware's low-32-bit esp_timer wrap (~71.6 min), safe within a batch.
        timestamp_s = ntp_ms / 1000.0 - ((last_local_timestamp_us - local_timestamp_us) & 0xFFFFFFFF) / 1e6
        frame = CsiFrame(1, expected_num_subcarriers)
        frame.grid[0, :] = csi
        frame.timestamp = timestamp_s
        frame.node_id = node_id
        frames.append(frame)
    return frames


def short_mac(mac: str) -> str:
    """Last two MAC octets — the compact transmitter label for a link key (e.g. 'ee:ff')."""
    return ":".join(mac.split(":")[-2:]) if ":" in mac else mac


def parse_batch_by_link(payload: bytes, *, tx_mac=None) -> dict:
    """One UDP batch -> dict[LinkKey -> list[CsiFrame]], keeping TX identity so each directed
    (tx->rx) link is its OWN stream (the all-pairs fusion input). O(n·S).

    Same binary v2 format + timestamp scheme as parse_batch (rx_node = header node; ntp_ms ≈ the last
    frame's wall time; per-frame t reconstructed from local_timestamp; header n bounds parsing).
    Difference: frames are bucketed by the per-record MAC (the transmitter) instead of merged, and the
    subcarrier-width guard is applied PER LINK (a future 5 GHz arm can carry a different width)."""
    node_id, ntp_ms, num_records, version, body_offset = _parse_batch_header(payload)
    tx_mac_bytes = _mac_to_bytes(tx_mac) if tx_mac is not None else None

    parsed = []  # (tx_short_mac, csi, local_timestamp_us)
    for mac_bytes, csi, local_timestamp_us in _iter_batch_records(payload, body_offset, num_records, version):
        if tx_mac_bytes is not None and mac_bytes != tx_mac_bytes:
            continue
        tx_short_mac = f"{mac_bytes[4]:02x}:{mac_bytes[5]:02x}"   # last two octets == short_mac(full mac)
        parsed.append((tx_short_mac, csi, local_timestamp_us))
    if not parsed:
        return {}

    last_local_timestamp_us = parsed[-1][2]
    # Bare-tuple keys through the per-record loop (this runs O(records) per UDP batch, ~3000/s);
    # LinkKey construction is deferred to one pass over the O(links) result below.
    links: dict = {}
    expected_num_subcarriers_by_link: dict = {}  # per-link width guard
    for tx_short_mac, csi, local_timestamp_us in parsed:
        key = (tx_short_mac, node_id)
        num_subcarriers = int(csi.size)
        if key not in expected_num_subcarriers_by_link:
            expected_num_subcarriers_by_link[key] = num_subcarriers
        elif num_subcarriers != expected_num_subcarriers_by_link[key]:
            continue
        frame = CsiFrame(1, num_subcarriers)
        frame.grid[0, :] = csi
        # & 0xFFFFFFFF handles the firmware ts_us low-32-bit wrap (see parse_batch).
        frame.timestamp = ntp_ms / 1000.0 - ((last_local_timestamp_us - local_timestamp_us)
                                             & 0xFFFFFFFF) / 1e6
        frame.node_id = node_id
        links.setdefault(key, []).append(frame)
    return {LinkKey(*key): frames for key, frames in links.items()}
