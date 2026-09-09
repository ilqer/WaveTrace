"""Unit tests for wavetrace.Source's transport options classes (§1.5 tests/unit/, mirroring
wavetrace/Source.py). Split out of tests/unit/test_config.py when SourceOptions/UdpSourceOptions/
SerialSourceOptions/NexmonSourceOptions moved out of wavetrace.Config to live beside the sources
they configure — their only real consumer is wavetrace.Source (plus a handful of scripts and
web/streamer.py that construct them)."""

import pytest

from wavetrace.Source import (
    NexmonSourceOptions,
    SerialSourceOptions,
    SourceOptions,
    UdpSourceOptions,
)


# ----- SourceOptions (base) -------------------------------------------------------------------------

def test_source_options_defaults():
    options = SourceOptions()
    assert options.timeout_seconds == 5.0 and options.max_frames is None


def test_source_options_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout_seconds"):
        SourceOptions(timeout_seconds=0)


def test_source_options_rejects_non_positive_max_frames_when_given():
    with pytest.raises(ValueError, match="max_frames"):
        SourceOptions(max_frames=0)


def test_source_options_are_frozen():
    options = SourceOptions()
    with pytest.raises(AttributeError):
        options.max_frames = 1


# ----- UdpSourceOptions ------------------------------------------------------------------------------

def test_udp_source_options_is_a_source_options():
    assert issubclass(UdpSourceOptions, SourceOptions)


def test_udp_source_options_defaults_match_the_previous_udpsource_literals():
    options = UdpSourceOptions()
    assert options.port == 5566 and options.tx_mac is None
    assert options.timeout_seconds == 5.0 and options.max_frames is None


def test_udp_source_options_rejects_out_of_range_port():
    with pytest.raises(ValueError, match="port"):
        UdpSourceOptions(port=0)


def test_udp_source_options_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout_seconds"):
        UdpSourceOptions(timeout_seconds=0)


def test_udp_source_options_rejects_non_positive_max_frames_when_given():
    with pytest.raises(ValueError, match="max_frames"):
        UdpSourceOptions(max_frames=0)


# ----- SerialSourceOptions ---------------------------------------------------------------------------

def test_serial_source_options_is_a_source_options():
    assert issubclass(SerialSourceOptions, SourceOptions)


def test_serial_source_options_defaults_match_the_previous_serialreader_literals():
    options = SerialSourceOptions(device="/dev/tty.usbserial-0")
    assert options.baud == 921600 and options.node_id == 1
    assert options.tx_mac is None and options.timeout_seconds == 5.0 and options.max_frames is None


def test_serial_source_options_requires_non_empty_device():
    with pytest.raises(ValueError, match="device"):
        SerialSourceOptions(device="")


def test_serial_source_options_rejects_non_positive_baud():
    with pytest.raises(ValueError, match="baud"):
        SerialSourceOptions(device="/dev/tty.x", baud=0)


def test_serial_source_options_rejects_negative_node_id():
    with pytest.raises(ValueError, match="node_id"):
        SerialSourceOptions(device="/dev/tty.x", node_id=-1)


def test_serial_source_options_rejects_non_positive_timeout_via_inherited_rule():
    with pytest.raises(ValueError, match="timeout_seconds"):
        SerialSourceOptions(device="/dev/tty.x", timeout_seconds=0)


# ----- NexmonSourceOptions ---------------------------------------------------------------------------

def test_nexmon_source_options_is_a_source_options():
    assert issubclass(NexmonSourceOptions, SourceOptions)


def test_nexmon_source_options_defaults_match_the_previous_nexmonsource_literals():
    options = NexmonSourceOptions(live=True)
    assert options.iface == "wlan0" and options.node_id == 100
    assert options.bandwidth_mhz == 80
    assert options.timeout_seconds == 5.0 and options.max_frames is None


def test_nexmon_source_options_accepts_pcap_path_without_live():
    options = NexmonSourceOptions(pcap_path="/tmp/capture.pcap")
    assert options.pcap_path == "/tmp/capture.pcap" and options.live is False


def test_nexmon_source_options_rejects_neither_pcap_path_nor_live():
    with pytest.raises(ValueError, match="pcap_path or live"):
        NexmonSourceOptions()


def test_nexmon_source_options_rejects_negative_node_id():
    with pytest.raises(ValueError, match="node_id"):
        NexmonSourceOptions(live=True, node_id=-1)


def test_nexmon_source_options_rejects_non_positive_bandwidth():
    with pytest.raises(ValueError, match="bandwidth_mhz"):
        NexmonSourceOptions(live=True, bandwidth_mhz=0)


def test_nexmon_source_options_rejects_non_positive_timeout_via_inherited_rule():
    with pytest.raises(ValueError, match="timeout_seconds"):
        NexmonSourceOptions(live=True, timeout_seconds=0)
