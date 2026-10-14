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
]


def parse_csi_line(line, *, tx_mac=None):
    """-> (csi (S,) complex64, local_timestamp_us, mac), or None if malformed/filtered; never raises. O(S).

    esp-csi stores [imag, real] pairs: csi[k] = complex(data[2k+1], data[2k]). `tx_mac` silently drops
    other transmitters (beacons, foreign traffic), case-insensitive."""
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
            csi[k] = complex(data[2 * k + 1], data[2 * k])
        return csi, local_timestamp_us, mac
    except Exception:
        return None


# Binary UDP wire format (little-endian). Header then packed records. ver 2 = int8 I/Q (ESP32 nodes),
# ver 3 = int16 I/Q (Pi 5 GHz node: keeps the full amplitude range the weapon σ² feature needs).
_BIN_HDR = struct.Struct("<BBBQH")   # magic, ver, node, ntp_ms, n  -> 13 bytes
_BIN_MAGIC = 0x57
_BIN_VERS = (2, 3)


def _parse_batch_header(payload: bytes):
    """ValueError on a bad header. `num_records` is authoritative: the parser trusts it over byte length."""
    if len(payload) < _BIN_HDR.size:
        raise ValueError(f"UdpSource: bad batch header: {len(payload)} bytes < {_BIN_HDR.size}")
    magic, version, node_id, ntp_ms, num_records = _BIN_HDR.unpack_from(payload, 0)
    if magic != _BIN_MAGIC or version not in _BIN_VERS:
        raise ValueError(f"UdpSource: bad batch header: magic={magic:#x} ver={version}")
    return node_id, ntp_ms, num_records, version, _BIN_HDR.size


def _mac_to_bytes(mac: str) -> bytes:
    return bytes(int(octet, 16) for octet in mac.split(":"))


def _iter_batch_records(payload: bytes, offset: int, num_records: int, version: int = 2):
    """The header count bounds the loop: trailing bytes past `num_records` records are never read as CSI.
    Stops early on a corrupt record (truncated or wrong byte length) since the stream can't be resynced.
    Record = mac[6] | ts_us(u32 LE) | len(u16 LE) | CSI bytes laid out [imag0,real0,...], int8 per
    component (ver 2, len=2*S) or int16 (ver 3, len=4*S).
    Yields the raw 6 MAC bytes so the hot path skips per-record string formatting."""
    end = len(payload)
    bytes_per_subcarrier = 2 if version == 2 else 4
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
    """O(n·S). A bad or missing header raises ValueError (wiring error). Records whose subcarrier
    count differs from the first kept one are skipped (width guard).

    ntp_ms/1000 is the batch SEND time ≈ the LAST frame's wall time; the others are reconstructed from
    their local_timestamp offsets: t_i = ntp_ms/1000 - (last_us - local_ts_i) / 1e6."""
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
            continue
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
    """Last two MAC octets, e.g. 'ee:ff'."""
    return ":".join(mac.split(":")[-2:]) if ":" in mac else mac


def parse_batch_by_link(payload: bytes, *, tx_mac=None) -> dict:
    """Each directed (tx->rx) link is its OWN stream. O(n·S).

    Same format and timestamp scheme as parse_batch, but frames are bucketed by the per-record MAC
    instead of merged, and the subcarrier-width guard is applied PER LINK, since links can carry
    different widths."""
    node_id, ntp_ms, num_records, version, body_offset = _parse_batch_header(payload)
    tx_mac_bytes = _mac_to_bytes(tx_mac) if tx_mac is not None else None

    parsed = []
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
    expected_num_subcarriers_by_link: dict = {}
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
