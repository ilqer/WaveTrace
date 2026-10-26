"""UDP datagrams must stay byte-exact for the wire-format parser; round-tripping catches format drift."""
import os
import sys

import numpy as np
import pytest

# firmware/pi/ is a standalone runtime dir (not a package); add it to the path for import.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "firmware", "pi"))

import struct  # noqa: E402

from nexmon_reader import parse_nexmon_csi  # noqa: E402
from publisher import BatchPublisher, mac_to_bytes, quantize_csi, quantize_csi_i16  # noqa: E402
from wavetrace.adapters.source import parse_batch, parse_batch_by_link

ACCESS_POINT_MAC = "aa:bb:cc:dd:ee:ff"
NODE = 5


class _FakeSock:
    """Capture sendto payloads instead of hitting the network."""

    def __init__(self):
        self.sent = []

    def sendto(self, data, address):
        self.sent.append(data)

    def close(self):
        pass


def _publisher(max_datagram_bytes=1450):
    publisher = BatchPublisher("127.0.0.1", 9876, NODE, ACCESS_POINT_MAC,
                               max_datagram_bytes=max_datagram_bytes)
    publisher._sock = _FakeSock()
    return publisher


def test_mac_to_bytes_roundtrip():
    assert mac_to_bytes(ACCESS_POINT_MAC) == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
    with pytest.raises(ValueError):
        mac_to_bytes("aa:bb:cc")


def test_quantize_preserves_amplitude_shape():
    subcarrier_count = 64
    subcarrier_index = np.arange(subcarrier_count)
    csi = ((1.0 + subcarrier_index / subcarrier_count)
           * np.exp(1j * subcarrier_index * 0.1)).astype(np.complex64)
    raw = np.frombuffer(quantize_csi(csi), dtype=np.int8).astype(np.float32)
    assert raw.size == 2 * subcarrier_count
    # Host reconstruction: csi[k] = real=d[2k+1] + 1j*d[2k].
    reconstructed = raw[1::2] + 1j * raw[0::2]
    # Quantization is a global scale: |reconstructed| should track |csi| (correlation ~1).
    correlation = np.corrcoef(np.abs(reconstructed), np.abs(csi))[0, 1]
    assert correlation > 0.99
    assert raw.max() <= 127 and raw.min() >= -128


def test_batch_parses_through_host():
    subcarrier_count = 128
    publisher = _publisher()
    csi_vectors = []
    for j in range(4):
        csi = ((1.0 + 0.01 * j) * np.exp(1j * np.linspace(0, np.pi, subcarrier_count))).astype(np.complex64)
        csi_vectors.append(csi)
        publisher.add(quantize_csi(csi), timestamp_us=1000 + j * 100)
    publisher.flush()

    assert len(publisher._sock.sent) == 1
    frames = parse_batch(publisher._sock.sent[0])
    assert len(frames) == 4
    for frame in frames:
        assert frame.node_id == NODE
        assert frame.grid.shape == (1, subcarrier_count)

    # Per-link split: the link key uses the AP's last two octets and rx node.
    links = parse_batch_by_link(publisher._sock.sent[0])
    assert ("ee:ff", NODE) in links
    assert len(links[("ee:ff", NODE)]) == 4


def test_v3_int16_preserves_exact_amplitude():
    # v3 int16 with fixed scale perfectly round-trips CSI integers. (v2 int8 only rescales).
    subcarrier_count = 256
    publisher = BatchPublisher("127.0.0.1", 9876, NODE, ACCESS_POINT_MAC, wire_version=3)
    publisher._sock = _FakeSock()
    real = (np.arange(subcarrier_count) % 4000 - 2000).astype(np.float32)
    imaginary = (np.arange(subcarrier_count) % 3000 - 1500).astype(np.float32)
    csi = (real + 1j * imaginary).astype(np.complex64)
    publisher.add(quantize_csi_i16(csi, scale=1.0), timestamp_us=42)
    publisher.flush()

    links = parse_batch_by_link(publisher._sock.sent[0])
    frames = links[("ee:ff", NODE)]
    assert len(frames) == 1
    got = frames[0].grid[0]
    np.testing.assert_array_equal(got.real, real)   # exact, not just correlated
    np.testing.assert_array_equal(got.imag, imaginary)


def _nexmon_payload(mac_bytes, real, imaginary):
    """Build a synthetic Nexmon CSI UDP payload: 18-byte header + int16 [real,imag] pairs."""
    header = struct.pack("<H b B 6s H H H H", 0x1111, -40, 0x08, mac_bytes, 1, 0, 0, 0)
    iq = np.empty(2 * real.size, dtype="<i2")
    iq[0::2] = real
    iq[1::2] = imaginary
    return header + iq.tobytes()


def test_parse_nexmon_csi_roundtrip():
    mac = mac_to_bytes(ACCESS_POINT_MAC)
    subcarrier_count = 128
    real = (np.arange(subcarrier_count) % 50 - 25).astype(np.int16)
    imaginary = (np.arange(subcarrier_count) % 30 - 15).astype(np.int16)
    out = parse_nexmon_csi(_nexmon_payload(mac, real, imaginary))
    assert out is not None
    source_mac, csi = out
    assert source_mac == mac
    assert csi.size == subcarrier_count
    np.testing.assert_array_equal(np.real(csi).astype(np.int16), real)
    np.testing.assert_array_equal(np.imag(csi).astype(np.int16), imaginary)


def test_parse_nexmon_csi_rejects_malformed():
    assert parse_nexmon_csi(b"\x00" * 10) is None          # shorter than header
    assert parse_nexmon_csi(b"\x00" * (18 + 3)) is None     # non-multiple-of-4 body


def test_mtu_splits_but_preserves_all_records():
    subcarrier_count = 128  # 256 CSI bytes + 12 header = 268 B/record -> ~5 per 1450 B datagram
    publisher = _publisher(max_datagram_bytes=1450)
    record_count = 17
    for j in range(record_count):
        csi = np.exp(1j * np.linspace(0, np.pi, subcarrier_count)).astype(np.complex64)
        publisher.add(quantize_csi(csi), timestamp_us=j)
    publisher.flush()

    assert len(publisher._sock.sent) > 1  # split across datagrams
    total = sum(len(parse_batch(datagram)) for datagram in publisher._sock.sent)
    assert total == record_count
    for datagram in publisher._sock.sent:
        assert len(datagram) <= 1450
