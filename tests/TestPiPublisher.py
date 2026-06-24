"""Pi node publisher: prove its bytes parse back through the real host wire-format parser.

The Pi is additive and runs on separate hardware, but its UDP datagrams must be byte-exact
for the existing host (wavetrace.Source). These tests round-trip pi/publisher.py output through
the host parser so a format drift on either side fails here."""
import os
import sys

import numpy as np
import pytest

# firmware/pi/ is a standalone runtime dir (not a package); add it to the path for import.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "firmware", "pi"))

import struct  # noqa: E402

from nexmon_reader import parse_nexmon_csi  # noqa: E402
from publisher import BatchPublisher, mac_to_bytes, quantize_csi, quantize_csi_i16  # noqa: E402
from wavetrace.Source import parse_batch, parse_batch_links  # noqa: E402

AP = "aa:bb:cc:dd:ee:ff"
NODE = 5


class _FakeSock:
    """Capture sendto payloads instead of hitting the network."""

    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append(data)

    def close(self):
        pass


def _publisher(mtu=1450):
    pub = BatchPublisher("127.0.0.1", 9876, NODE, AP, mtu=mtu)
    pub._sock = _FakeSock()  # swap in the fake after construction
    return pub


def test_mac_to_bytes_roundtrip():
    assert mac_to_bytes(AP) == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
    with pytest.raises(ValueError):
        mac_to_bytes("aa:bb:cc")


def test_quantize_preserves_amplitude_shape():
    # A frame with a clear amplitude profile across subcarriers.
    S = 64
    k = np.arange(S)
    csi = ((1.0 + k / S) * np.exp(1j * k * 0.1)).astype(np.complex64)
    raw = np.frombuffer(quantize_csi(csi), dtype=np.int8).astype(np.float32)
    assert raw.size == 2 * S
    # Host reconstruction: csi[k] = real=d[2k+1] + 1j*d[2k].
    rec = raw[1::2] + 1j * raw[0::2]
    # Quantization is a global scale: |rec| should track |csi| (correlation ~1).
    corr = np.corrcoef(np.abs(rec), np.abs(csi))[0, 1]
    assert corr > 0.99
    assert raw.max() <= 127 and raw.min() >= -128


def test_batch_parses_through_host():
    S = 128
    pub = _publisher()
    csis = []
    for j in range(4):
        csi = ((1.0 + 0.01 * j) * np.exp(1j * np.linspace(0, np.pi, S))).astype(np.complex64)
        csis.append(csi)
        pub.add(quantize_csi(csi), ts_us=1000 + j * 100)
    pub.flush()

    assert len(pub._sock.sent) == 1
    frames = parse_batch(pub._sock.sent[0])
    assert len(frames) == 4
    for fr in frames:
        assert fr.node_id == NODE
        assert fr.grid.shape == (1, S)

    # Per-link split: the link key uses the AP's last two octets and rx node.
    links = parse_batch_links(pub._sock.sent[0])
    assert ("ee:ff", NODE) in links
    assert len(links[("ee:ff", NODE)]) == 4


def test_v3_int16_preserves_exact_amplitude():
    # The weapon feature needs absolute amplitude: ver-3 int16 with a fixed scale must round-trip
    # the CSI integers EXACTLY (unlike ver-2 int8, which rescales). S=256 = HT80 width.
    S = 256
    pub = BatchPublisher("127.0.0.1", 9876, NODE, AP, ver=3)
    pub._sock = _FakeSock()
    real = (np.arange(S) % 4000 - 2000).astype(np.float32)
    imag = (np.arange(S) % 3000 - 1500).astype(np.float32)
    csi = (real + 1j * imag).astype(np.complex64)
    pub.add(quantize_csi_i16(csi, scale=1.0), ts_us=42)
    pub.flush()

    links = parse_batch_links(pub._sock.sent[0])
    frames = links[("ee:ff", NODE)]
    assert len(frames) == 1
    got = frames[0].grid[0]
    np.testing.assert_array_equal(got.real, real)   # exact, not just correlated
    np.testing.assert_array_equal(got.imag, imag)


def _nexmon_payload(mac_bytes, real, imag):
    """Build a synthetic Nexmon CSI UDP payload: 18-byte header + int16 [real,imag] pairs."""
    hdr = struct.pack("<H b B 6s H H H H", 0x1111, -40, 0x08, mac_bytes, 1, 0, 0, 0)
    iq = np.empty(2 * real.size, dtype="<i2")
    iq[0::2] = real
    iq[1::2] = imag
    return hdr + iq.tobytes()


def test_parse_nexmon_csi_roundtrip():
    mac = mac_to_bytes(AP)
    S = 128
    real = (np.arange(S) % 50 - 25).astype(np.int16)
    imag = (np.arange(S) % 30 - 15).astype(np.int16)
    out = parse_nexmon_csi(_nexmon_payload(mac, real, imag))
    assert out is not None
    src, csi = out
    assert src == mac
    assert csi.size == S
    np.testing.assert_array_equal(np.real(csi).astype(np.int16), real)
    np.testing.assert_array_equal(np.imag(csi).astype(np.int16), imag)


def test_parse_nexmon_csi_rejects_malformed():
    assert parse_nexmon_csi(b"\x00" * 10) is None          # shorter than header
    assert parse_nexmon_csi(b"\x00" * (18 + 3)) is None     # non-multiple-of-4 body


def test_mtu_splits_but_preserves_all_records():
    S = 128  # 256 CSI bytes + 12 hdr = 268 B/record -> ~5 per 1450 B datagram
    pub = _publisher(mtu=1450)
    n = 17
    for j in range(n):
        csi = np.exp(1j * np.linspace(0, np.pi, S)).astype(np.complex64)
        pub.add(quantize_csi(csi), ts_us=j)
    pub.flush()

    assert len(pub._sock.sent) > 1  # split across datagrams
    total = sum(len(parse_batch(d)) for d in pub._sock.sent)
    assert total == n
    for d in pub._sock.sent:
        assert len(d) <= 1450
