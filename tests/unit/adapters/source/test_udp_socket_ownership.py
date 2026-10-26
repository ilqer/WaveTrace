import socket

import pytest

from wavetrace.adapters.source.udp import UdpLinkBatchReceiver, bind_udp_socket


class _RecordingSocket:
    def __init__(self, bind_error=None):
        self._bind_error = bind_error
        self.is_closed = False

    def setsockopt(self, *option):
        pass

    def bind(self, address):
        if self._bind_error is not None:
            raise self._bind_error

    def settimeout(self, timeout_s):
        pass

    def recvfrom(self, buffer_size):
        raise socket.timeout

    def close(self):
        self.is_closed = True


def _patch_socket(monkeypatch, udp_socket):
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: udp_socket)
    return udp_socket


def test_a_failed_bind_closes_the_socket_and_raises_the_bind_error(monkeypatch):
    udp_socket = _patch_socket(monkeypatch, _RecordingSocket(OSError(48, "Address already in use")))
    with pytest.raises(OSError, match="Address already in use"):
        bind_udp_socket(5599, timeout_s=0.5)
    assert udp_socket.is_closed


def test_the_receiver_closes_its_socket_when_its_block_raises(monkeypatch):
    udp_socket = _patch_socket(monkeypatch, _RecordingSocket())
    with pytest.raises(KeyboardInterrupt):
        with UdpLinkBatchReceiver(5599, timeout_s=0.5) as receiver:
            assert receiver.receive() == {}
            raise KeyboardInterrupt
    assert udp_socket.is_closed
