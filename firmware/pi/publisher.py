"""Pack CSI into binary UDP wire format v2 and send as node 5.
Mirrors wavetrace/Source.py parser. Format matches ESP firmware.
Input: complex64 csi[S] + microsecond timestamp.
Output: datagrams = 13-byte header + records.
"""
import socket
import struct
import time
import numpy as np

# Header matches wavetrace/Source.py _BIN_HDR.
# ver 2: int8 I/Q. ver 3: int16 I/Q (retains amplitude for weapon feature).
_BIN_HDR = struct.Struct("<BBBQH")
_BIN_MAGIC = 0x57
# Record header: 12 bytes (mac[6], ts_us LE, len LE) then CSI.
_REC_HDR = struct.Struct("<6sIH")

_DEFAULT_MTU = 1450  # Keep datagram under Ethernet MTU.


def mac_to_bytes(mac: str) -> bytes:
    """Convert MAC string to 6 bytes."""
    parts = mac.split(":")
    if len(parts) != 6:
        raise ValueError(f"MAC must be 6 octets, got {mac!r}")
    return bytes(int(p, 16) for p in parts)


def quantize_csi(csi: np.ndarray, scale=None) -> bytes:
    """Convert to 2*S int8 bytes (ver 2), [imag, real, ...] interleaved.
    scale=None auto-scales frame to 127 peak.
    Warning: auto-scale erases amplitude. Fine for presence, but breaks weapon feature. O(S)."""
    re = np.real(csi)
    im = np.imag(csi)
    if scale is None:
        peak = float(max(np.abs(re).max(initial=0.0), np.abs(im).max(initial=0.0), 1e-9))
        scale = 127.0 / peak
    out = np.empty(2 * csi.size, dtype=np.float32)
    out[0::2] = im * scale
    out[1::2] = re * scale
    np.clip(np.rint(out), -128, 127, out=out)
    return out.astype(np.int8).tobytes()


def quantize_csi_i16(csi: np.ndarray, scale: float = 1.0) -> bytes:
    """Convert to 4*S int16 bytes (ver 3), [imag, real, ...] LE.
    Uses fixed scale to keep absolute amplitude comparable across frames.
    Preserves amplitude/variance for weapon σ² feature. O(S)."""
    out = np.empty(2 * csi.size, dtype=np.float32)
    out[0::2] = np.imag(csi) * scale
    out[1::2] = np.real(csi) * scale
    np.clip(np.rint(out), -32768, 32767, out=out)
    return out.astype("<i2").tobytes()


class BatchPublisher:
    """Accumulate records and flush datagram before MTU limit.
    ntp_ms is stamped at flush, matching host reconstruction scheme."""

    def __init__(self, pc_ip: str, port: int, node_id: int, ap_mac: str, ver: int = 2,
                 mtu: int = _DEFAULT_MTU):
        self._addr = (pc_ip, port)
        self._node = node_id
        self._mac = mac_to_bytes(ap_mac)
        self._ver = ver          # 2 = int8 payload, 3 = int16 payload (must match the quantizer used)
        self._mtu = mtu
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._records: list[bytes] = []
        self._size = _BIN_HDR.size

    def add(self, csi_bytes: bytes, ts_us: int) -> None:
        """Queue record. Flush if MTU exceeded."""
        if len(csi_bytes) % 2 != 0:
            raise ValueError(f"CSI byte length must be even (2*S), got {len(csi_bytes)}")
        rec = _REC_HDR.pack(self._mac, ts_us & 0xFFFFFFFF, len(csi_bytes)) + csi_bytes
        if self._records and self._size + len(rec) > self._mtu:
            self.flush()
        self._records.append(rec)
        self._size += len(rec)

    def flush(self) -> None:
        """Send queued records as datagram."""
        if not self._records:
            return
        ntp_ms = int(time.time() * 1000)
        hdr = _BIN_HDR.pack(_BIN_MAGIC, self._ver, self._node, ntp_ms, len(self._records))
        self._sock.sendto(hdr + b"".join(self._records), self._addr)
        self._records.clear()
        self._size = _BIN_HDR.size

    def close(self) -> None:
        self.flush()
        self._sock.close()
