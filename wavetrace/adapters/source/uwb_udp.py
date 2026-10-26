"""Batched-UDP UWB CIR backhaul from the Portenta C33 (wire format v4)."""

from dataclasses import dataclass
import socket

from wavetrace.adapters.source.options import SourceOptions
from wavetrace.adapters.source.udp import bind_udp_socket
from wavetrace.adapters.source.uwb_wire_format import parse_cir_batch
from wavetrace.application.ports import CirSource


@dataclass(frozen=True, slots=True, kw_only=True)
class UdpCirSourceOptions(SourceOptions):
    """`max_frames` caps the number of CirRecords yielded."""

    port: int = 9879

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.port < 65536:
            raise ValueError("port must be in (0, 65536)")


class UdpCirSource(CirSource):
    """Binds 0.0.0.0:port. records() stops after `timeout_seconds` with no packet (an idle timeout,
    not a capture duration) or at `max_frames`. A malformed packet raises ValueError."""

    def __init__(self, options: UdpCirSourceOptions = UdpCirSourceOptions()):
        self._port = options.port
        self._timeout_seconds = options.timeout_seconds
        self._max_records = options.max_frames

    def records(self):
        udp_socket = bind_udp_socket(self._port, timeout_s=self._timeout_seconds)
        try:
            count = 0
            while self._max_records is None or count < self._max_records:
                try:
                    payload, _ = udp_socket.recvfrom(65535)
                except socket.timeout:
                    return
                for record in parse_cir_batch(payload):
                    yield record
                    count += 1
                    if self._max_records is not None and count >= self._max_records:
                        return
        finally:
            udp_socket.close()
