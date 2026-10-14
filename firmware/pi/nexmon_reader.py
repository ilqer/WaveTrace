"""Decodes Nexmon (CYW43455) CSI UDP datagrams on port 5500.

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

_HEADER = struct.Struct("<H b B 6s H H H H")  # 18 bytes
_HEADER_BYTES = _HEADER.size
_SOURCE_MAC_OFFSET = 4


def parse_nexmon_csi(payload: bytes) -> Optional[Tuple[bytes, np.ndarray]]:
    if len(payload) <= _HEADER_BYTES or (len(payload) - _HEADER_BYTES) % 4 != 0:
        return None
    source_mac = payload[_SOURCE_MAC_OFFSET:_SOURCE_MAC_OFFSET + 6]
    iq = np.frombuffer(payload, dtype="<i2", offset=_HEADER_BYTES)  # interleaved real, imag
    csi = (iq[0::2].astype(np.float32) + 1j * iq[1::2].astype(np.float32)).astype(np.complex64)
    return source_mac, csi


class NexmonReader:
    """expected_subcarrier_count pins the subcarrier width (None: lock to the first frame's) and drops
    off-width frames. access_point_mac filters the transmitter."""

    def __init__(self, port: int, expected_subcarrier_count: Optional[int] = None,
                 access_point_mac: Optional[bytes] = None):
        self._port = port
        self._expected_subcarrier_count = expected_subcarrier_count
        self._access_point_mac = access_point_mac

    def frames(self) -> Iterator[Tuple[float, bytes, np.ndarray]]:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_socket.bind(("", self._port))
        try:
            while True:
                payload, _ = udp_socket.recvfrom(4096)
                parsed = parse_nexmon_csi(payload)
                if parsed is None:
                    continue
                mac, csi = parsed
                if self._access_point_mac is not None and mac != self._access_point_mac:
                    continue
                if self._expected_subcarrier_count is None:
                    self._expected_subcarrier_count = csi.size
                elif csi.size != self._expected_subcarrier_count:
                    continue
                yield time.time(), mac, csi
        finally:
            udp_socket.close()
