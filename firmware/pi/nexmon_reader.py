"""Read the live Nexmon CSI stream on the Pi and yield complex CSI vectors.

The Nexmon CSI firmware (CYW43455) emits one UDP datagram per received frame to the local
broadcast on port 5500. We bind a UDP socket to that port and decode each datagram.

Payload layout (Nexmon CSI, bcm43455c0 — same as the `nexcsi` decoder):
    18-byte header:
        u16 magic | i8 rssi | u8 frame_control | u8[6] source_mac | u16 seq
        u16 core_spatial | u16 chanspec | u16 chip_version
    then NFFT complex subcarriers as int16 I/Q pairs (4 bytes each) -> NFFT = (len-18)//4.

  Input:  UDP datagrams on NEXMON_PORT.
  Output: iterator of (timestamp_s, source_mac_bytes, complex64 csi[S]).
  Errors: malformed/short datagrams are skipped (never raises mid-stream).

NOTE (verify on hardware, Part A step 2): the int16 I/Q ordering below assumes
[real, imag] per subcarrier. For PRESENCE this is irrelevant (|csi| is identical either
way and the host normalizes); confirm only if you later need phase. NFFT for 5 GHz HT40 = 128.
"""
import socket
import struct
import time
from typing import Iterator, Optional, Tuple
import numpy as np

_HDR = struct.Struct("<H b B 6s H H H H")  # 18 bytes; see module docstring
_HDR_LEN = _HDR.size
_MAC_OFF = 4  # source_mac starts at byte 4


def parse_nexmon_csi(payload: bytes) -> Optional[Tuple[bytes, np.ndarray]]:
    """One Nexmon UDP payload -> (source_mac_bytes, complex64 csi[NFFT]) or None if malformed."""
    if len(payload) <= _HDR_LEN or (len(payload) - _HDR_LEN) % 4 != 0:
        return None
    src_mac = payload[_MAC_OFF:_MAC_OFF + 6]
    iq = np.frombuffer(payload, dtype="<i2", offset=_HDR_LEN)  # interleaved real, imag
    csi = (iq[0::2].astype(np.float32) + 1j * iq[1::2].astype(np.float32)).astype(np.complex64)
    return src_mac, csi


class NexmonReader:
    """Yield (ts, mac, csi) from the local Nexmon CSI UDP stream, pinned to one subcarrier width.

    expect_s pins the width (off-width frames dropped); None locks to the first frame's width.
    ap_mac (optional) keeps only frames from that transmitter — a cheap software backstop to the
    firmware's makecsiparams MAC filter."""

    def __init__(self, port: int, expect_s: Optional[int] = None, ap_mac: Optional[bytes] = None):
        self._port = port
        self._s = expect_s
        self._ap_mac = ap_mac

    def frames(self) -> Iterator[Tuple[float, bytes, np.ndarray]]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self._port))
        try:
            while True:
                payload, _ = sock.recvfrom(4096)
                parsed = parse_nexmon_csi(payload)
                if parsed is None:
                    continue
                mac, csi = parsed
                if self._ap_mac is not None and mac != self._ap_mac:
                    continue
                if self._s is None:
                    self._s = csi.size  # lock width on first frame
                elif csi.size != self._s:
                    continue  # drop off-width frame (pinned width)
                yield time.time(), mac, csi
        finally:
            sock.close()
