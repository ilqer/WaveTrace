"""Pack complex CSI into the host's binary UDP wire format v2 and send it as node 5.

Byte-exact mirror of the parser in wavetrace/Source.py (_BIN_HDR / _iter_bin_records), so the
host picks the Pi up with zero code changes. The ESP firmware emits this same format.

  Input:  complex64 csi[S] + a microsecond timestamp, repeatedly.
  Output: UDP datagrams to (PC_IP, port); one datagram = 13-byte header + n packed records.
  Errors: ValueError on a non-6-byte MAC or odd CSI byte length; socket errors propagate.
"""
import socket
import struct
import time
import numpy as np

# Header: magic, ver, node, ntp_ms, n  -> 13 bytes (must match wavetrace/Source.py _BIN_HDR).
_BIN_HDR = struct.Struct("<BBBQH")
_BIN_MAGIC, _BIN_VER = 0x57, 2
# Record header: mac[6], ts_us(u32 LE), len(u16 LE) -> 12 bytes; CSI bytes follow.
_REC_HDR = struct.Struct("<6sIH")

_DEFAULT_MTU = 1450  # keep each datagram comfortably under the Ethernet MTU


def mac_to_bytes(mac: str) -> bytes:
    """'aa:bb:cc:dd:ee:ff' -> 6 raw bytes. Raises ValueError if not 6 octets."""
    parts = mac.split(":")
    if len(parts) != 6:
        raise ValueError(f"MAC must be 6 octets, got {mac!r}")
    return bytes(int(p, 16) for p in parts)


def quantize_csi(csi: np.ndarray, scale=None) -> bytes:
    """complex64 csi[S] -> 2*S int8 bytes, interleaved [imag0, real0, imag1, real1, ...].

    The host rebuilds csi[k] = real=d[2k+1] + 1j*d[2k], so imag goes in even slots, real in odd.
    scale=None auto-scales each frame so its peak |component| maps to 127 (lossy but fine for
    presence; calibration normalizes amplitude). O(S), one temporary buffer."""
    re = np.real(csi)
    im = np.imag(csi)
    if scale is None:
        peak = float(max(np.abs(re).max(initial=0.0), np.abs(im).max(initial=0.0), 1e-9))
        scale = 127.0 / peak
    out = np.empty(2 * csi.size, dtype=np.float32)
    out[0::2] = im * scale  # imag in even byte slots
    out[1::2] = re * scale  # real in odd byte slots
    np.clip(np.rint(out), -128, 127, out=out)
    return out.astype(np.int8).tobytes()


class BatchPublisher:
    """Accumulate records and flush a v2 datagram whenever the next record would exceed the MTU.

    Call add() per CSI frame and flush() to force-send a partial batch (e.g. at shutdown or on a
    timer). ntp_ms is stamped at flush time (~ the last frame's wall clock), matching the host's
    timestamp-reconstruction scheme."""

    def __init__(self, pc_ip: str, port: int, node_id: int, ap_mac: str, mtu: int = _DEFAULT_MTU):
        self._addr = (pc_ip, port)
        self._node = node_id
        self._mac = mac_to_bytes(ap_mac)
        self._mtu = mtu
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._records: list[bytes] = []
        self._size = _BIN_HDR.size

    def add(self, csi_bytes: bytes, ts_us: int) -> None:
        """Queue one record; flush first if appending it would overflow the MTU."""
        if len(csi_bytes) % 2 != 0:
            raise ValueError(f"CSI byte length must be even (2*S), got {len(csi_bytes)}")
        rec = _REC_HDR.pack(self._mac, ts_us & 0xFFFFFFFF, len(csi_bytes)) + csi_bytes
        if self._records and self._size + len(rec) > self._mtu:
            self.flush()
        self._records.append(rec)
        self._size += len(rec)

    def flush(self) -> None:
        """Send the queued records as one datagram (no-op if empty)."""
        if not self._records:
            return
        ntp_ms = int(time.time() * 1000)
        hdr = _BIN_HDR.pack(_BIN_MAGIC, _BIN_VER, self._node, ntp_ms, len(self._records))
        self._sock.sendto(hdr + b"".join(self._records), self._addr)
        self._records.clear()
        self._size = _BIN_HDR.size

    def close(self) -> None:
        self.flush()
        self._sock.close()
