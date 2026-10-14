"""Pack CSI into binary UDP wire format v2 and send as node 5.
Mirrors the wavetrace/adapters/source/wire_format.py parser. Format matches ESP firmware.
Input: complex64 csi[S] + microsecond timestamp.
Output: datagrams = 13-byte header + records.
"""
import socket
import struct
import time
import numpy as np

# Header matches wavetrace/adapters/source/wire_format.py _BIN_HDR.
# ver 2: int8 I/Q. ver 3: int16 I/Q (retains amplitude for weapon feature).
_BIN_HDR = struct.Struct("<BBBQH")
_BIN_MAGIC = 0x57
# Record header: 12 bytes (mac[6], ts_us LE, len LE) then CSI.
_REC_HDR = struct.Struct("<6sIH")

_DEFAULT_MAX_DATAGRAM_BYTES = 1450  # Keep datagram under Ethernet MTU.


def mac_to_bytes(mac: str) -> bytes:
    octets = mac.split(":")
    if len(octets) != 6:
        raise ValueError(f"MAC must be 6 octets, got {mac!r}")
    return bytes(int(octet, 16) for octet in octets)


def quantize_csi(csi: np.ndarray, scale=None) -> bytes:
    """Convert to 2*S int8 bytes (ver 2), [imag, real, ...] interleaved.
    scale=None auto-scales frame to 127 peak.
    Warning: auto-scale erases amplitude. Fine for presence, but breaks weapon feature. O(S)."""
    real_part = np.real(csi)
    imaginary_part = np.imag(csi)
    if scale is None:
        peak = float(max(np.abs(real_part).max(initial=0.0), np.abs(imaginary_part).max(initial=0.0), 1e-9))
        scale = 127.0 / peak
    interleaved = np.empty(2 * csi.size, dtype=np.float32)
    interleaved[0::2] = imaginary_part * scale
    interleaved[1::2] = real_part * scale
    np.clip(np.rint(interleaved), -128, 127, out=interleaved)
    return interleaved.astype(np.int8).tobytes()


def quantize_csi_i16(csi: np.ndarray, scale: float = 1.0) -> bytes:
    """Convert to 4*S int16 bytes (ver 3), [imag, real, ...] LE.
    Uses fixed scale to keep absolute amplitude comparable across frames.
    Preserves amplitude/variance for weapon σ² feature. O(S)."""
    interleaved = np.empty(2 * csi.size, dtype=np.float32)
    interleaved[0::2] = np.imag(csi) * scale
    interleaved[1::2] = np.real(csi) * scale
    np.clip(np.rint(interleaved), -32768, 32767, out=interleaved)
    return interleaved.astype("<i2").tobytes()


class BatchPublisher:
    """Accumulate records and flush datagram before MTU limit.
    ntp_ms is stamped at flush, matching host reconstruction scheme."""

    def __init__(self, host_address: str, port: int, node_id: int, access_point_mac: str,
                 wire_version: int = 2, max_datagram_bytes: int = _DEFAULT_MAX_DATAGRAM_BYTES):
        self._destination = (host_address, port)
        self._node_id = node_id
        self._access_point_mac = mac_to_bytes(access_point_mac)
        # 2 = int8 payload, 3 = int16 payload (must match the quantizer used)
        self._wire_version = wire_version
        self._max_datagram_bytes = max_datagram_bytes
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._records: list[bytes] = []
        self._pending_datagram_bytes = _BIN_HDR.size

    def add(self, csi_bytes: bytes, timestamp_us: int) -> None:
        """Sends the pending datagram first when this record would push it past max_datagram_bytes."""
        if len(csi_bytes) % 2 != 0:
            raise ValueError(f"CSI byte length must be even (2*S), got {len(csi_bytes)}")
        record = _REC_HDR.pack(self._access_point_mac, timestamp_us & 0xFFFFFFFF, len(csi_bytes)) + csi_bytes
        if self._records and self._pending_datagram_bytes + len(record) > self._max_datagram_bytes:
            self.flush()
        self._records.append(record)
        self._pending_datagram_bytes += len(record)

    def flush(self) -> None:
        if not self._records:
            return
        ntp_ms = int(time.time() * 1000)
        header = _BIN_HDR.pack(_BIN_MAGIC, self._wire_version, self._node_id, ntp_ms, len(self._records))
        self._sock.sendto(header + b"".join(self._records), self._destination)
        self._records.clear()
        self._pending_datagram_bytes = _BIN_HDR.size

    def close(self) -> None:
        self.flush()
        self._sock.close()
