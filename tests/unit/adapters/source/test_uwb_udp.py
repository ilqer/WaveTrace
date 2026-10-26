import socket
import struct
import threading

import numpy as np
import pytest

from wavetrace.adapters.source.uwb_udp import UdpCirSource, UdpCirSourceOptions
from wavetrace.adapters.source.uwb_wire_format import parse_cir_batch


def _packet(num_records=1):
    iq = np.arange(16, dtype="<i2").reshape(2, 4, 2)
    record = (struct.pack("<HIBHIh", 0xBEEF, 77, 2, 4, 1_000_000, 0) + struct.pack("<2h", 1, 2)
              + struct.pack("<HhhhH", 100, 0, 0, 0, 0) + bytes((20, 21)) + iq.tobytes())
    return struct.pack("<BBBQH", 0x57, 4, 3, 99, num_records) + record * num_records


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_default_port_avoids_other_wavetrace_ports():
    assert UdpCirSourceOptions().port == 9879


@pytest.mark.parametrize("port", [0, 65536])
def test_options_reject_bad_port(port):
    with pytest.raises(ValueError, match="port"):
        UdpCirSourceOptions(port=port)


def test_loopback_yields_records_then_idle_timeout_ends():
    port = _free_port()
    source = UdpCirSource(UdpCirSourceOptions(port=port, timeout_seconds=0.3))
    records = source.records()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet = _packet(num_records=2)

    # the generator binds on its first next(), inside list(); the timer sends after the bind
    timer = threading.Timer(0.05, lambda: sender.sendto(packet, ("127.0.0.1", port)))
    timer.start()
    try:
        received = list(records)
    finally:
        timer.join()
        sender.close()

    assert len(received) == 2
    assert received[0].peer_short_addr == 0xBEEF
    assert received[0].node_id == 3
    assert received[0].taps.shape == (2, 4)
    expected = parse_cir_batch(packet)
    np.testing.assert_array_equal(received[1].taps, expected[1].taps)


def test_idle_timeout_with_no_packet_ends_immediately():
    port = _free_port()
    assert list(UdpCirSource(UdpCirSourceOptions(port=port, timeout_seconds=0.1)).records()) == []


def test_max_frames_caps_records():
    port = _free_port()
    source = UdpCirSource(UdpCirSourceOptions(port=port, timeout_seconds=1.0, max_frames=1))
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    timer = threading.Timer(0.05, lambda: sender.sendto(_packet(num_records=2), ("127.0.0.1", port)))
    timer.start()
    try:
        received = list(source.records())
    finally:
        timer.join()
        sender.close()
    assert len(received) == 1
