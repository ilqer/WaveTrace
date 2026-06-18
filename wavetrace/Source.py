"""Phase 8 — CSI sources for the CLI.

A `CsiSource` yields CsiFrames; the rest of the pipeline (front-end → recognition → output) is
source-agnostic. Sources available today:
  * SyntheticSource — wraps an in-memory frame list (the fixtures generate it for dev/CI).
  * RecordingSource — replays frames saved by `save_recording` (the `capture` CLI mode).
  * UdpSource — batched-UDP CSI receiver (plan §3 backhaul wire format, T7/P10).
  * SerialReader — esp-csi over USB serial (one ESP = one node); the no-RF-cost bring-up backhaul
    for 1–2-node smoke tests before the UDP mesh exists. Reuses `parse_csi_line`. Needs `pyserial`.

DOCUMENTED OPTIONS (not yet implemented — deferred until hardware is confirmed):
  * NexmonSource (option): the Pi 5 GHz arm — nexmon CSI on bcm43455c0 (Pi 3B+/4B/5, fw 7_45_189),
    pcap via `csiread`; reference implementation: wifi-3d-fusion `nexmon_pcap.py`.

Recording format under out_dir (mirrors save_dataset): grid.npy (F,A,S) complex64 + t.npy (F,) +
node_id.npy (F,) + meta.json. O(F·A·S) to (de)serialize.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
import socket
import time

import numpy as np

from wavetrace import CsiFrame

# esp-csi 25-column CSV header (data = last column, JSON array of ints).
ESP_CSI_COLUMNS = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
    "ant", "sig_len", "rx_state", "len", "first_word", "data",
]  # 25 columns; data is a JSON array at index 24


def parse_csi_line(line, *, tx_mac=None):
    """One esp-csi CSV line -> (csi (S,) complex64, local_ts_us int, mac str), or None if
    malformed/filtered. Never raises on a bad line. O(S).

    I/Q pairing: esp-csi stores [imag, real] pairs — csi[k] = complex(data[2k+1], data[2k]).
    tx_mac filter: if set, lines whose MAC != tx_mac (case-insensitive) are silently dropped (removes
    beacons / foreign traffic — CSI must come from the dedicated TX only)."""
    try:
        # maxsplit=24 protects the JSON data array's internal commas
        parts = line.strip().split(",", 24)
        if len(parts) != 25 or parts[0] != "CSI_DATA":
            return None
        mac = parts[2]
        if tx_mac is not None and mac.lower() != tx_mac.lower():
            return None
        local_ts_us = int(parts[18])
        # real esp-csi wraps the array in CSV double-quotes ("[...]"); strip them (unquoted also works)
        data = json.loads(parts[24].strip().strip('"'))
        if len(data) % 2 != 0:
            return None
        S = len(data) // 2
        csi = np.empty(S, dtype=np.complex64)
        for k in range(S):
            csi[k] = complex(data[2 * k + 1], data[2 * k])  # [imag, real] pairing
        return csi, local_ts_us, mac
    except Exception:
        return None


def parse_batch(payload: bytes, *, tx_mac=None) -> list:
    """One UDP batch payload -> list[CsiFrame] (node_id + wall-clock timestamps from the header).
    O(n·S).

    Header = first line, JSON: {"v": 1, "node": int, "ntp_ms": int, "n": int}. A bad/missing header
    raises ValueError (wiring error). Bad CSI lines are skipped silently (RF noise). Lines whose
    subcarrier count S differs from the first parsed line are also skipped (format guard).

    Timestamp scheme: ntp_ms/1000.0 is the batch SEND time ≈ the LAST frame's wall time; each
    frame's absolute time is reconstructed from its local_timestamp offset relative to the last
    parsed line's local_ts:  t_i = ntp_ms/1000 - (last_us - local_ts_i) / 1e6."""
    lines = payload.decode("utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError("UdpSource: bad batch header: empty payload")
    try:
        header = json.loads(lines[0])
        node_id = int(header["node"])
        ntp_ms = int(header["ntp_ms"])
    except Exception as e:
        raise ValueError(f"UdpSource: bad batch header: {lines[0]!r}") from e

    parsed = []
    S_ref = None
    for line in lines[1:]:
        if not line:
            continue
        result = parse_csi_line(line, tx_mac=tx_mac)
        if result is None:
            continue
        csi, local_ts_us, _ = result
        S = int(csi.size)
        if S_ref is None:
            S_ref = S
        elif S != S_ref:
            continue  # mixed-S line: skip
        parsed.append((csi, local_ts_us))

    if not parsed:
        return []

    last_us = parsed[-1][1]
    frames = []
    for csi, local_ts_us in parsed:
        t = ntp_ms / 1000.0 - (last_us - local_ts_us) / 1e6
        fr = CsiFrame(1, S_ref)
        fr.grid[0, :] = csi
        fr.timestamp = t
        fr.node_id = node_id
        frames.append(fr)
    return frames


def mac_short(mac: str) -> str:
    """Last two MAC octets — the compact transmitter label for a link key (e.g. 'ee:ff')."""
    return ":".join(mac.split(":")[-2:]) if ":" in mac else mac


def parse_batch_links(payload: bytes, *, tx_mac=None) -> dict:
    """One UDP batch -> dict[(tx_short, rx_node) -> list[CsiFrame]], keeping TX identity so each
    directed (tx->rx) link is its OWN stream (the all-pairs fusion input). O(n·S).

    Same header + timestamp scheme as parse_batch (rx_node = header node; ntp_ms ≈ the last frame's
    wall time; per-frame t reconstructed from local_timestamp). Difference: frames are bucketed by the
    per-line MAC (the transmitter) instead of merged, and the subcarrier-width format guard is applied
    PER LINK (a future 5 GHz arm can carry a different width than the 2.4 GHz mesh)."""
    lines = payload.decode("utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError("UdpSource: bad batch header: empty payload")
    try:
        header = json.loads(lines[0])
        node_id = int(header["node"])
        ntp_ms = int(header["ntp_ms"])
    except Exception as e:
        raise ValueError(f"UdpSource: bad batch header: {lines[0]!r}") from e

    parsed = []  # (tx_short, csi, local_ts_us)
    for line in lines[1:]:
        if not line:
            continue
        result = parse_csi_line(line, tx_mac=tx_mac)
        if result is None:
            continue
        csi, local_ts_us, mac = result
        parsed.append((mac_short(mac), csi, local_ts_us))
    if not parsed:
        return {}

    last_us = parsed[-1][2]
    links: dict = {}
    s_ref: dict = {}  # per-link width guard
    for tx_short, csi, local_ts_us in parsed:
        key = (tx_short, node_id)
        S = int(csi.size)
        if key not in s_ref:
            s_ref[key] = S
        elif S != s_ref[key]:
            continue
        fr = CsiFrame(1, S)
        fr.grid[0, :] = csi
        fr.timestamp = ntp_ms / 1000.0 - (last_us - local_ts_us) / 1e6
        fr.node_id = node_id
        links.setdefault(key, []).append(fr)
    return links


def resample_uniform(frames, fs_hz):
    """Resample a CSI stream onto a uniform 1/fs_hz time grid (linear interp of complex CSI). O(n·A·S).

    The mesh delivers CSI at a jittery rate (round-robin + contention swings it 30-300 Hz); the
    front-end builds COUNT-based windows gated by `fs_ok` (Config.fs_tol), so off-rate windows get
    dropped — wasting capture and gapping coverage. Resampling first onto a fixed grid makes the live
    fs exactly fs_hz, so windows pass and Doppler/spectrogram features are not smeared by jitter.

    Pass a SINGLE-link/single-node stream (mixing transmitters interleaves different channels).
    Returns a new frame list on the uniform grid; input may be unsorted (sorted by timestamp here)."""
    if fs_hz <= 0:
        raise ValueError("resample_uniform: fs_hz must be positive")
    if len(frames) < 2:
        return list(frames)
    A, S = frames[0].grid.shape
    t = np.array([fr.timestamp for fr in frames], dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t = t[order]
    flat = np.stack([frames[i].grid.reshape(-1) for i in order]).astype(np.complex64)  # (n, A·S)
    n_out = max(2, int((t[-1] - t[0]) * fs_hz) + 1)
    tg = t[0] + np.arange(n_out, dtype=np.float64) / fs_hz
    re = np.empty((n_out, A * S), dtype=np.float32)
    im = np.empty_like(re)
    for c in range(A * S):  # per-cell 1-D interp; A·S is small (≈64 for HT20) so this is cheap
        re[:, c] = np.interp(tg, t, flat[:, c].real)
        im[:, c] = np.interp(tg, t, flat[:, c].imag)
    out = (re + 1j * im).astype(np.complex64).reshape(n_out, A, S)
    node_id = getattr(frames[0], "node_id", 0)
    result = []
    for i in range(n_out):
        fr = CsiFrame(A, S)
        fr.grid[:, :] = out[i]
        fr.timestamp = float(tg[i])
        fr.node_id = node_id
        result.append(fr)
    return result


class CsiSource(ABC):
    """A stream of CsiFrames feeding the front-end."""

    @abstractmethod
    def frames(self):
        """Yield CsiFrame objects in capture order."""


class SyntheticSource(CsiSource):
    """Replay an in-memory frame list (e.g. from fixtures.SyntheticCsi/SyntheticRecording)."""

    def __init__(self, frames):
        self._frames = list(frames)

    def frames(self):
        return iter(self._frames)


class RecordingSource(CsiSource):
    """Replay frames saved by `save_recording`. Reconstructs each CsiFrame on demand."""

    def __init__(self, rec_dir):
        self._dir = Path(rec_dir)

    def frames(self):
        return load_recording(self._dir)


class UdpSource(CsiSource):
    """Receive batched-UDP CSI (plan §3 backhaul). frames() binds 0.0.0.0:port and yields until
    timeout_s with no packet (or max_frames). The socket loop is a thin shell over parse_batch."""

    def __init__(self, port: int = 5566, *, tx_mac=None, timeout_s: float = 5.0,
                 max_frames=None):
        self._port = int(port)
        self._tx_mac = tx_mac
        self._timeout_s = float(timeout_s)
        self._max_frames = max_frames

    def frames(self):
        """Bind UDP socket and yield CsiFrames; stop on timeout or max_frames."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", self._port))
            sock.settimeout(self._timeout_s)
            count = 0
            while self._max_frames is None or count < self._max_frames:
                try:
                    payload, _ = sock.recvfrom(65535)
                except socket.timeout:
                    return
                for fr in parse_batch(payload, tx_mac=self._tx_mac):
                    yield fr
                    count += 1
                    if self._max_frames is not None and count >= self._max_frames:
                        return
        finally:
            sock.close()


class SerialReader(CsiSource):
    """esp-csi CSI over USB serial — one ESP32 = one node = one antenna. A thin pyserial shell over
    `parse_csi_line`; the no-RF-cost bring-up path for 1–2 nodes before the UDP mesh exists.

    Frame timestamp = PC wall-clock at read, NOT the ESP's `local_timestamp` counter: independent ESP
    clocks have no common origin, so the PC arrival time is the consistent basis for aligning against
    camera labels and other nodes (sub-ms serial latency ≪ the front-end window). `tx_mac` filters to
    the dedicated TX (drops beacons / foreign traffic). `pip install pyserial`; the import + port open
    are deferred to frames() so importing this module never requires pyserial."""

    def __init__(self, port, *, baud: int = 921600, node_id: int = 1, tx_mac=None,
                 timeout_s: float = 5.0, max_frames=None):
        self._port = port
        self._baud = int(baud)
        self._node = int(node_id)
        self._tx_mac = tx_mac
        self._timeout_s = float(timeout_s)
        self._max_frames = max_frames

    def frames(self):
        """Open the serial port and yield (1, S) CsiFrames tagged with node_id; stop on read timeout
        (no data within timeout_s) or max_frames. Malformed/filtered lines are skipped silently.

        Subcarrier-count guard (same as parse_batch): even from the dedicated TX, an RX occasionally
        receives off-format frames (legacy/HT20/HT40 differ in S). The first yielded frame sets S_ref
        and any line with a different S is dropped, so the stream is a fixed (1, S) shape downstream."""
        try:
            import serial  # pyserial
        except ImportError as e:
            raise ImportError("SerialReader needs pyserial: pip install pyserial") from e
        ser = serial.Serial(self._port, self._baud, timeout=self._timeout_s)
        try:
            count = 0
            S_ref = None
            while self._max_frames is None or count < self._max_frames:
                raw = ser.readline()
                if not raw:
                    return  # timeout with no data -> stream ended
                result = parse_csi_line(raw.decode("utf-8", errors="replace"), tx_mac=self._tx_mac)
                if result is None:
                    continue
                csi, _local_ts_us, _mac = result
                S = int(csi.size)
                if S_ref is None:
                    S_ref = S
                elif S != S_ref:
                    continue  # off-format frame (different bandwidth/mode) -> drop to keep fixed shape
                fr = CsiFrame(1, S_ref)
                fr.grid[0, :] = csi
                fr.timestamp = time.time()
                fr.node_id = self._node
                yield fr
                count += 1
        finally:
            ser.close()


class NexmonSource(CsiSource):
    """5 GHz CSI from a Raspberry Pi running nexmon_csi (bcm43455c0; Pi 3B+/4B/5, fw 7_45_189).

    Reads nexmon-encapsulated CSI from a pcap file or a live tcpdump stream.
    Decoding is delegated to `csiread.Nexmon` (pip install csiread) — only `_decode_file` changes
    if your csiread version or chip differs.

    node_id=100 by default so it never collides with ESP32 ids 1..6, and the 2.4/5 GHz split
    downstream keys on node_id >= 100 == 5 GHz (matching NodeHealthMeter's band convention).

    Two modes:
      pcap_path set  -> replay a captured file (offline dev, no hardware).
      live=True      -> spawn tcpdump on `iface` and parse packets as they arrive."""

    def __init__(self, *, pcap_path=None, iface="wlan0", live=False, node_id=100,
                 timeout_s=5.0, max_frames=None, bandwidth=80):
        self._pcap = pcap_path
        self._iface = str(iface)
        self._live = bool(live)
        self._node = int(node_id)
        self._timeout = float(timeout_s)
        self._max = max_frames
        self._bw = int(bandwidth)

    def _csiread(self):
        try:
            import csiread
            return csiread
        except ImportError as e:
            raise ImportError("NexmonSource needs csiread: pip install csiread") from e

    def _decode_file(self):
        csiread = self._csiread()
        reader = csiread.Nexmon(self._pcap, chip="43455c0", bw=self._bw)
        reader.read()
        csi = np.asarray(reader.csi)           # (F, S) complex
        ts_raw = getattr(reader, "sec", None)
        ts = np.asarray(ts_raw, dtype=float) if ts_raw is not None else np.arange(len(csi)) / 100.0
        for i in range(len(csi)):
            fr = CsiFrame(1, csi.shape[1])
            fr.grid[0, :] = csi[i].astype(np.complex64)
            fr.timestamp = float(ts[i]) if i < ts.size else i / 100.0
            fr.node_id = self._node
            yield fr

    def _decode_live(self):
        import subprocess, tempfile, os, time as _t
        tmp = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False).name
        proc = subprocess.Popen(
            ["tcpdump", "-i", self._iface, "-w", tmp, "dst port 5500"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        count = 0
        try:
            while self._max is None or count < self._max:
                _t.sleep(0.3)
                if os.path.getsize(tmp) < 64:
                    continue
                self._pcap = tmp
                for fr in self._decode_file():
                    yield fr
                    count += 1
                    if self._max is not None and count >= self._max:
                        return
                open(tmp, "wb").close()    # truncate consumed chunk
        finally:
            proc.terminate()
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def frames(self):
        if self._live:
            yield from self._decode_live()
        elif self._pcap:
            yield from self._decode_file()
        else:
            raise ValueError("NexmonSource: set pcap_path or live=True")


def save_recording(frames, out_dir) -> Path:
    """Serialize a CsiFrame list to out_dir (grid/t/node_id .npy + meta.json). O(F·A·S)."""
    frames = list(frames)
    if not frames:
        raise ValueError("save_recording: no frames")
    A, S = frames[0].num_antennas, frames[0].num_subcarriers
    grid = np.stack([np.asarray(fr.grid) for fr in frames]).astype(np.complex64)  # (F, A, S)
    t = np.asarray([float(fr.timestamp) for fr in frames], dtype=np.float64)
    node = np.asarray([int(fr.node_id) for fr in frames], dtype=np.int32)
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "grid.npy", grid)
    np.save(p / "t.npy", t)
    np.save(p / "node_id.npy", node)
    with open(p / "meta.json", "w") as f:
        json.dump({"num_frames": len(frames), "num_antennas": int(A), "num_subcarriers": int(S)}, f,
                  indent=2)
    return p


def load_recording(rec_dir):
    """Yield reconstructed CsiFrames from a saved recording. O(F·A·S)."""
    p = Path(rec_dir)
    grid = np.load(p / "grid.npy")          # (F, A, S) complex64
    t = np.load(p / "t.npy")
    node = np.load(p / "node_id.npy")
    F, A, S = grid.shape
    for i in range(F):
        fr = CsiFrame(A, S)
        fr.timestamp = float(t[i])
        fr.node_id = int(node[i])
        fr.grid[:, :] = grid[i]             # zero-copy write into the native buffer
        yield fr
