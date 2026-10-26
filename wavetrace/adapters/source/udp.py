"""Batched-UDP CSI backhaul, the transport the ESP32 mesh and the Pi node send on."""

from dataclasses import dataclass
import socket

from wavetrace.adapters.source.options import SourceOptions
from wavetrace.adapters.source.wire_format import parse_batch, parse_batch_by_link
from wavetrace.application.ports import CsiSource, LinkBatchReceiver


def bind_udp_socket(port, *, timeout_s=None):
    """A UDP socket bound to `port` with a large (8 MB) kernel receive buffer. The big buffer absorbs
    bursts from N nodes while the main thread is busy in inference — without it the kernel drops
    datagrams before the single-threaded receiver reads them. The OS may cap the size (on macOS raise
    kern.ipc.maxsockbuf); the setsockopt failing is non-fatal."""
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        except OSError:
            pass
        udp_socket.bind(("0.0.0.0", port))
        if timeout_s is not None:
            udp_socket.settimeout(timeout_s)
    except BaseException:
        udp_socket.close()
        raise
    return udp_socket


class UdpLinkBatchReceiver(LinkBatchReceiver):
    """Binds 0.0.0.0:port; `receive` blocks at most `timeout_s`. Leaving its `with` block closes the socket."""

    def __init__(self, port, *, timeout_s):
        self._socket = bind_udp_socket(port, timeout_s=timeout_s)

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        self.close()

    def receive(self):
        try:
            payload, _ = self._socket.recvfrom(65535)
        except socket.timeout:
            return {}
        return parse_batch_by_link(payload)

    def close(self):
        self._socket.close()


@dataclass(frozen=True, slots=True, kw_only=True)
class UdpSourceOptions(SourceOptions):
    port: int = 5566
    tx_mac: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.port < 65536:
            raise ValueError("port must be in (0, 65536)")


class UdpSource(CsiSource):
    """Binds 0.0.0.0:port. frames() stops after `timeout_seconds` with no packet (an idle timeout,
    not a capture duration) or at `max_frames`."""

    def __init__(self, options: UdpSourceOptions = UdpSourceOptions()):
        self._port = options.port
        self._tx_mac = options.tx_mac
        self._timeout_seconds = options.timeout_seconds
        self._max_frames = options.max_frames

    def frames(self):
        udp_socket = bind_udp_socket(self._port, timeout_s=self._timeout_seconds)
        try:
            count = 0
            while self._max_frames is None or count < self._max_frames:
                try:
                    payload, _ = udp_socket.recvfrom(65535)
                except socket.timeout:
                    return
                for frame in parse_batch(payload, tx_mac=self._tx_mac):
                    yield frame
                    count += 1
                    if self._max_frames is not None and count >= self._max_frames:
                        return
        finally:
            udp_socket.close()
