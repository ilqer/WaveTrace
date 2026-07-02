"""Read live Nexmon CSI stream on Pi and yield complex CSI vectors.
Decodes UDP datagrams on port 5500 from Nexmon firmware (CYW43455).

Payload layout: 18-byte header then NFFT complex subcarriers (int16 I/Q).
NFFT = (len-18)//4. 5GHz HT80 = 256.

Note: Assumes [real, imag] ordering. Irrelevant for magnitude features like presence/weapon.
Only matters for phase/CIR methods.
"""
import socket
import struct
import time
from typing import Iterator, Optional, Tuple
import numpy as np

_HDR = struct.Struct("<H b B 6s H H H H")  # 18 bytes; see module docstring
_HDR_LEN = _HDR.size
_MAC_OFF = 4


def parse_nexmon_csi(payload: bytes) -> Optional[Tuple[bytes, np.ndarray]]:
    """Parse UDP payload to (mac, csi) or None."""
    if len(payload) <= _HDR_LEN or (len(payload) - _HDR_LEN) % 4 != 0:
        return None
    srcMac = payload[_MAC_OFF:_MAC_OFF + 6]
    iq = np.frombuffer(payload, dtype="<i2", offset=_HDR_LEN)  # interleaved real, imag
    csi = (iq[0::2].astype(np.float32) + 1j * iq[1::2].astype(np.float32)).astype(np.complex64)
    return srcMac, csi


class NexmonReader:
    """Yield (ts, mac, csi) from local UDP stream.
    expect_s pins subcarrier width. ap_mac filters transmitter."""

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
