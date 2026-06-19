"""T7/P10 — UdpSource pure function tests: parse_csi_line, parse_batch."""

import json
import struct
import sys

import numpy as np
import pytest

from wavetrace.Source import SerialReader, parse_csi_line, parse_batch


def _make_csi_line(csi_ints, mac="aa:bb:cc:dd:ee:ff", local_ts=1000):
    """Construct a well-formed 25-column esp-csi CSV line."""
    # Columns match ESP_CSI_COLUMNS order; local_timestamp at index 18, data at 24.
    cols = [
        "CSI_DATA", "1", mac, "-60", "11", "0", "7", "0",
        "1", "0", "0", "0", "0", "0",
        "-95", "0", "6", "0", str(local_ts), "0", "100", "0",
        # real esp-csi compact-prints the array AND wraps it in CSV double-quotes
        str(len(csi_ints) // 2), "1", '"' + json.dumps(csi_ints, separators=(",", ":")) + '"',
    ]
    assert len(cols) == 25
    return ",".join(cols)


def _bin_rec(csi_ints, mac="aa:bb:cc:dd:ee:ff", local_ts=1000):
    """One binary v2 record: mac[6] | ts_us(u32 LE) | len(u16 LE) | int8 raw CSI."""
    mb = bytes(int(x, 16) for x in mac.split(":"))
    data = bytes(v & 0xFF for v in csi_ints)
    return mb + struct.pack("<IH", local_ts, len(csi_ints)) + data


def _bin_batch(rows, node_id=0, ntp_ms=5000):
    """rows = [(csi_ints, mac, local_ts), ...] -> one binary v2 UDP batch (header + records)."""
    hdr = struct.pack("<BBBQH", 0x57, 2, node_id, ntp_ms, len(rows))
    return hdr + b"".join(_bin_rec(*r) for r in rows)


def _make_batch_payload(frames_ints, node_id=0, ntp_ms=5000):
    """Build a binary v2 UDP batch payload (header + one record per frame)."""
    # Local timestamps step by 10 000 µs = 10 ms between frames.
    rows = [(ints, "aa:bb:cc:dd:ee:ff", 1000 + i * 10_000) for i, ints in enumerate(frames_ints)]
    return _bin_batch(rows, node_id=node_id, ntp_ms=ntp_ms)


# ---- T7d.1: parse_csi_line -----------------------------------------------------------

def test_parse_csi_line_valid():
    """parse_csi_line returns (csi, local_ts_us, mac) with correct I/Q pairing."""
    # esp-csi stores [imag0, real0, imag1, real1, ...]; csi[k] = complex(real=data[2k+1], imag=data[2k])
    csi_ints = [10, 20, 30, 40, 50, 60, 70, 80]  # 4 subcarriers
    line = _make_csi_line(csi_ints, mac="aa:bb:cc:dd:ee:ff", local_ts=12345)
    result = parse_csi_line(line)
    assert result is not None
    csi, ts, mac = result
    assert csi.shape == (4,)
    assert csi.dtype == np.complex64
    # csi[k] = complex(data[2k+1], data[2k]) i.e. Python complex(real, imag)
    assert csi[0] == pytest.approx(complex(csi_ints[1], csi_ints[0]))
    assert csi[1] == pytest.approx(complex(csi_ints[3], csi_ints[2]))
    assert ts == 12345
    assert mac.lower() == "aa:bb:cc:dd:ee:ff"


def test_parse_csi_line_quoted_real_format():
    """Regression: real esp-csi wraps the array in CSV double-quotes ("[...]"). Both quoted and
    unquoted must parse (caught on real hardware 2026-06-15 — quoted lines were dropped)."""
    quoted = ('CSI_DATA,15562,1a:00:00:00:00:00,-25,11,1,0,1,1,1,0,0,0,0,-96,0,11,2,'
              '2361919,0,47,1,4,0,"[1,2,3,4]"')
    res = parse_csi_line(quoted)
    assert res is not None and res[0].shape == (2,)
    assert res[0][0] == pytest.approx(complex(2, 1))  # [imag,real] -> complex(real=2, imag=1)
    assert res[1] == 2361919


def test_parse_csi_line_filtering():
    """tx_mac filter drops non-matching lines; malformed lines return None."""
    csi_ints = [1, 2, 3, 4]
    line = _make_csi_line(csi_ints, mac="aa:bb:cc:dd:ee:ff")
    # Matching MAC passes
    assert parse_csi_line(line, tx_mac="AA:BB:CC:DD:EE:FF") is not None  # case-insensitive
    # Non-matching MAC silently dropped
    assert parse_csi_line(line, tx_mac="11:22:33:44:55:66") is None
    # Malformed line (not 25 cols, not CSI_DATA) → None, never raises
    assert parse_csi_line("garbage,line") is None
    assert parse_csi_line("") is None


# ---- SerialReader (esp-csi over USB serial) ------------------------------------------

class _FakeSerial:
    """Minimal pyserial.Serial stand-in: readline() walks a canned list of byte lines, then ''."""
    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    def readline(self):
        return self._lines.pop(0) if self._lines else b""

    def close(self):
        self.closed = True


def _install_fake_serial(monkeypatch, lines):
    """Inject a fake `serial` module so SerialReader.frames() imports it instead of real pyserial."""
    import types
    fake = types.ModuleType("serial")
    holder = {}
    def _serial(port, baud, timeout=None):
        holder["obj"] = _FakeSerial(lines)
        holder["args"] = (port, baud, timeout)
        return holder["obj"]
    fake.Serial = _serial
    monkeypatch.setitem(sys.modules, "serial", fake)
    return holder


def test_serial_reader_yields_node_tagged_frames(monkeypatch):
    """SerialReader parses esp-csi lines into (1, S) frames tagged with node_id; tx_mac filters."""
    good = _make_csi_line([1, 2, 3, 4, 5, 6], mac="aa:bb:cc:dd:ee:ff").encode()  # S=3
    other = _make_csi_line([7, 8, 9, 10], mac="11:22:33:44:55:66").encode()      # filtered out
    holder = _install_fake_serial(monkeypatch, [good, b"garbage line", other, good])
    reader = SerialReader("/dev/ttyUSB0", node_id=4, tx_mac="aa:bb:cc:dd:ee:ff", baud=921600)

    frames = list(reader.frames())
    assert len(frames) == 2                       # two matching-MAC lines; garbage + foreign dropped
    assert all(fr.node_id == 4 for fr in frames)
    assert all(fr.num_subcarriers == 3 for fr in frames)
    assert frames[0].grid[0, 0] == pytest.approx(complex(2, 1))  # [imag, real] -> complex(real, imag)
    assert holder["obj"].closed                   # port closed on exhaustion
    assert holder["args"] == ("/dev/ttyUSB0", 921600, 5.0)


def test_serial_reader_drops_off_format_frames(monkeypatch):
    """Real RX mixes bandwidths: a frame with a different S must be dropped so the stream stays a
    fixed (1, S) shape (caught on hardware 2026-06-15 — save_recording can't stack mixed shapes)."""
    s3 = _make_csi_line([1, 2, 3, 4, 5, 6]).encode()   # S=3 (first -> sets S_ref)
    s2 = _make_csi_line([7, 8, 9, 10]).encode()         # S=2 -> off-format, dropped
    _install_fake_serial(monkeypatch, [s3, s2, s3, s2, s3])
    frames = list(SerialReader("/dev/ttyUSB0").frames())
    assert len(frames) == 3 and all(fr.num_subcarriers == 3 for fr in frames)
    # all yielded frames share one shape -> save_recording can stack them
    np.stack([np.asarray(fr.grid) for fr in frames])


def test_serial_reader_respects_max_frames(monkeypatch):
    line = _make_csi_line([1, 2, 3, 4]).encode()
    _install_fake_serial(monkeypatch, [line] * 10)
    assert len(list(SerialReader("/dev/ttyUSB0", max_frames=3).frames())) == 3


def test_serial_reader_needs_pyserial(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)  # force ImportError on `import serial`
    with pytest.raises(ImportError, match="pyserial"):
        list(SerialReader("/dev/ttyUSB0").frames())


# ---- T7d.2: parse_batch valid --------------------------------------------------------

def test_parse_batch_valid():
    """parse_batch returns CsiFrames with node_id from header and correct NTP timestamps."""
    S = 4
    csi_ints_per_frame = [[i * 2, i * 2 + 1] * S for i in range(3)]  # 3 frames, 4 subcarriers
    ntp_ms = 5000
    payload = _make_batch_payload(csi_ints_per_frame, node_id=7, ntp_ms=ntp_ms)

    result = parse_batch(payload)
    assert len(result) == 3
    for fr in result:
        assert fr.node_id == 7
        assert fr.num_subcarriers == S

    # Timestamp: t_i = ntp_ms/1000 - (last_us - local_ts_i) / 1e6
    local_ts_list = [1000 + i * 10_000 for i in range(3)]
    last_us = local_ts_list[-1]
    for fr, local_ts in zip(result, local_ts_list):
        expected_t = ntp_ms / 1000.0 - (last_us - local_ts) / 1e6
        assert fr.timestamp == pytest.approx(expected_t, abs=1e-9)


# ---- T7d.3: parse_batch error paths --------------------------------------------------

def test_parse_batch_bad_header_and_bad_lines():
    """Bad header raises ValueError; truncated/mixed-width records within a valid batch are skipped."""
    # Bad header: wrong magic (not a v2 binary header)
    with pytest.raises(ValueError, match="bad batch header"):
        parse_batch(b"not_json\nsome,csv,line")

    # Empty payload (shorter than the 13-byte header)
    with pytest.raises(ValueError, match="bad batch header"):
        parse_batch(b"")

    # Valid header but no complete records (trailing garbage) → empty list, no error
    empty = struct.pack("<BBBQH", 0x57, 2, 0, 1000, 0) + b"\x01\x02\x03"
    assert parse_batch(empty) == []

    # Valid header + mixed S records: only matching-S kept (first S sets the reference)
    mixed = _bin_batch([([1, 2, 3, 4, 5, 6, 7, 8], "aa:bb:cc:dd:ee:ff", 1000),   # S=4 (reference)
                        ([1, 2, 3, 4], "aa:bb:cc:dd:ee:ff", 2000)],              # S=2 → skipped
                       node_id=0, ntp_ms=2000)
    frames = parse_batch(mixed)
    assert len(frames) == 1  # only the S=4 record kept
    assert frames[0].num_subcarriers == 4


def test_parse_batch_honors_header_count():
    """Header n is authoritative: extras past n are ignored; n>records stops at truncation."""
    rec = _bin_rec([1, 2, 3, 4], "aa:bb:cc:dd:ee:ff", 1000)  # one S=2 record (12+4 bytes)

    def hdr(n):
        return struct.pack("<BBBQH", 0x57, 2, 0, 5000, n)

    # (a) n smaller than the encoded records -> only n parsed
    assert len(parse_batch(hdr(1) + rec + rec + rec)) == 1
    # (b) trailing garbage after exactly n records -> ignored, not parsed as CSI
    assert len(parse_batch(hdr(2) + rec + rec + b"\xde\xad\xbe\xef")) == 2
    # (c) n larger than the encoded records -> stops at truncation, returns what's there
    assert len(parse_batch(hdr(5) + rec + rec)) == 2


def test_parse_batch_handles_uint32_ts_wrap():
    """ts_us is the firmware's low-32-bit timer (wraps ~71.6 min); a within-batch rollover must not
    corrupt timestamps (the masked subtraction keeps the 512 µs gap, not a ~71 min jump)."""
    ntp_ms = 5000
    rows = [([1, 2, 3, 4], "aa:bb:cc:dd:ee:ff", 0xFFFFFF00),   # just below the 2^32 wrap
            ([5, 6, 7, 8], "aa:bb:cc:dd:ee:ff", 0x00000100)]   # wrapped: 512 µs later
    frames = parse_batch(_bin_batch(rows, node_id=0, ntp_ms=ntp_ms))
    assert len(frames) == 2
    assert frames[0].timestamp == pytest.approx(ntp_ms / 1000.0 - 512 / 1e6, abs=1e-9)
    assert frames[1].timestamp == pytest.approx(ntp_ms / 1000.0, abs=1e-9)
