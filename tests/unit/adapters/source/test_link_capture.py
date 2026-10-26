import builtins
import collections
import socket
import threading

import pytest

from wavetrace import CsiFrame
from wavetrace.adapters.source import link_capture, udp
from wavetrace.adapters.source.link_capture import UdpLinkCapture, UdpTimedLinkCapture, detect_node_ids


def _frames(num_frames, num_subcarriers=4, node_id=1):
    frames = []
    for i in range(num_frames):
        frame = CsiFrame(1, num_subcarriers)
        frame.timestamp = float(i)
        frame.node_id = node_id
        frames.append(frame)
    return frames


class _FakeUdpSocket:
    """Each recvfrom returns the next queued payload; an empty queue is the silence timeout."""

    def __init__(self, payloads):
        self.pending = collections.deque(payloads)
        self.is_closed = False

    def setsockopt(self, *args):
        pass

    def bind(self, address):
        pass

    def settimeout(self, timeout_s):
        pass

    def recvfrom(self, buffer_size):
        if not self.pending:
            raise socket.timeout
        return self.pending.popleft(), ("127.0.0.1", 0)

    def close(self):
        self.is_closed = True


@pytest.fixture
def fake_socket(monkeypatch):
    """Payloads are the parsed frames themselves, so the tests drive the loop, not the wire format."""
    udp_socket = _FakeUdpSocket([])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: udp_socket)
    monkeypatch.setattr(link_capture, "parse_batch_by_link", lambda payload: payload)
    monkeypatch.setattr(udp, "parse_batch", lambda payload, tx_mac=None: payload)
    monkeypatch.setattr(builtins, "input", lambda *prompt: "")
    monkeypatch.setattr(link_capture.time, "sleep", lambda seconds: None)
    return udp_socket


def test_capture_stops_on_silence_and_keeps_each_links_dominant_width(fake_socket, capsys):
    fake_socket.pending.extend([
        {("aa", 1): _frames(2, num_subcarriers=4), ("bb", 1): _frames(1, num_subcarriers=8)},
        {("aa", 1): _frames(1, num_subcarriers=8) + _frames(1, num_subcarriers=4)},
    ])
    links = UdpLinkCapture(5599, [1], max_frames_per_link=10).capture("prompt", countdown_s=2, hint="hint")
    assert {link_key: [frame.num_subcarriers for frame in frames] for link_key, frames in links.items()} == {
        ("aa", 1): [4, 4, 4], ("bb", 1): [8]}
    assert fake_socket.is_closed
    output = capsys.readouterr().out
    assert output.startswith("\n>> prompt\n   press Enter to start...\n")
    assert "   [CAPTURING] hint\n" in output


def test_capture_stops_once_every_link_of_every_wanted_node_is_full(fake_socket, capsys):
    fake_socket.pending.extend([
        {("aa", 1): _frames(3)},
        {("aa", 2): _frames(5, node_id=2)},
        {("aa", 1): _frames(5)},
        {("aa", 1): _frames(5)},
    ])
    links = UdpLinkCapture(5599, [1, 2], max_frames_per_link=4).capture("prompt", countdown_s=0, hint="hint")
    assert {link_key: len(frames) for link_key, frames in links.items()} == {("aa", 1): 4, ("aa", 2): 4}
    assert len(fake_socket.pending) == 1


def test_detect_node_ids_returns_the_sorted_distinct_nodes_heard_until_silence(fake_socket, capsys):
    fake_socket.pending.extend([_frames(2, node_id=3) + _frames(1, node_id=1), _frames(1, node_id=3)])
    assert detect_node_ids(5599, timeout_s=0.1) == [1, 3]
    assert capsys.readouterr().out == "listening for active nodes...\n"


def test_detect_node_ids_stops_listening_after_150_frames(fake_socket, capsys):
    fake_socket.pending.extend([_frames(150, node_id=1), _frames(1, node_id=2)])
    assert detect_node_ids(5599) == [1]


class _SteppedClock:
    """Each reading is one second after the last."""

    def __init__(self):
        self.now_s = 0.0

    def __call__(self):
        self.now_s += 1.0
        return self.now_s


def test_timed_capture_records_until_its_deadline_keeping_the_wanted_nodes_at_their_dominant_width(
        fake_socket, monkeypatch):
    monkeypatch.setattr(link_capture.time, "monotonic", _SteppedClock())
    fake_socket.pending.extend([
        {("aa", 1): _frames(2, num_subcarriers=4), ("aa", 9): _frames(2, node_id=9)},
        {("aa", 1): _frames(1, num_subcarriers=8), ("bb", 1): _frames(1)},
        {("aa", 1): _frames(5)},
    ])
    links = UdpTimedLinkCapture(5599, [1]).capture_for(2.5, stop=threading.Event())
    assert {link_key: [frame.num_subcarriers for frame in frames] for link_key, frames in links.items()} == {
        ("aa", 1): [4, 4], ("bb", 1): [4]}
    assert len(fake_socket.pending) == 1 and fake_socket.is_closed


def test_timed_capture_rides_out_receive_timeouts_until_its_deadline(fake_socket, monkeypatch):
    clock = _SteppedClock()
    monkeypatch.setattr(link_capture.time, "monotonic", clock)
    links = UdpTimedLinkCapture(5599, [1]).capture_for(4.5, stop=threading.Event())
    assert links == {} and clock.now_s == 6.0


def test_timed_capture_ends_once_stop_is_set(fake_socket, monkeypatch):
    monkeypatch.setattr(link_capture.time, "monotonic", _SteppedClock())
    stop = threading.Event()
    fake_socket.pending.extend([{("aa", 1): _frames(1)}, {("aa", 1): _frames(1)}])
    original_recvfrom = fake_socket.recvfrom

    def receive_then_stop(buffer_size):
        stop.set()
        return original_recvfrom(buffer_size)

    fake_socket.recvfrom = receive_then_stop
    links = UdpTimedLinkCapture(5599, [1]).capture_for(100.0, stop=stop)
    assert {link_key: len(frames) for link_key, frames in links.items()} == {("aa", 1): 1}


def test_timed_capture_raises_the_bind_error_itself(monkeypatch):
    class _PortInUseSocket:
        def setsockopt(self, *option):
            pass

        def bind(self, address):
            raise OSError(48, "Address already in use")

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: _PortInUseSocket())
    with pytest.raises(OSError, match="Address already in use"):
        UdpTimedLinkCapture(5599, [1]).capture_for(1.0, stop=threading.Event())
