"""Phase 8 — CSI sources for the CLI.

A `CsiSource` yields CsiFrames; the rest of the pipeline (front-end → recognition → output) is
source-agnostic. Sources available today:
  * SyntheticSource — wraps an in-memory frame list (the fixtures generate it for dev/CI).
  * RecordingSource — replays frames saved by `saveRecording` (the `capture` CLI mode).
  * UdpSource — batched-UDP CSI receiver (plan §3 backhaul wire format, T7/P10).
  * SerialReader — esp-csi over USB serial (one ESP = one node); the no-RF-cost bring-up backhaul
    for 1–2-node smoke tests before the UDP mesh exists. Reuses `parseCsiLine`. Needs `pyserial`.

DOCUMENTED OPTIONS (not yet implemented — deferred until hardware is confirmed):
  * NexmonSource (option): the Pi 5 GHz arm — nexmon CSI on bcm43455c0 (Pi 3B+/4B/5, fw 7_45_189),
    pcap via `csiread`; reference implementation: wifi-3d-fusion `nexmon_pcap.py`.

Recording format under out_dir (mirrors saveDataset): grid.npy (F,A,S) complex64 + t.npy (F,) +
node_id.npy (F,) + meta.json. O(F·A·S) to (de)serialize.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import struct
import time
import warnings

import numpy as np

from wavetrace import CsiFrame

# esp-csi 25-column CSV header (data = last column, JSON array of ints).
ESP_CSI_COLUMNS = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
    "ant", "sig_len", "rx_state", "len", "first_word", "data",
]  # 25 columns; data is a JSON array at index 24


def parseCsiLine(line, *, tx_mac=None):
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
        localTsUs = int(parts[18])
        # real esp-csi wraps the array in CSV double-quotes ("[...]"); strip them (unquoted also works)
        data = json.loads(parts[24].strip().strip('"'))
        if len(data) % 2 != 0:
            return None
        S = len(data) // 2
        csi = np.empty(S, dtype=np.complex64)
        for k in range(S):
            csi[k] = complex(data[2 * k + 1], data[2 * k])  # [imag, real] pairing
        return csi, localTsUs, mac
    except Exception:
        return None


# Binary UDP wire format (little-endian). Header then packed records. ver 2 = int8 I/Q (ESP32 nodes),
# ver 3 = int16 I/Q (Pi 5 GHz node — keeps full amplitude range the weapon σ² feature needs).
_BIN_HDR = struct.Struct("<BBBQH")   # magic, ver, node, ntp_ms, n  -> 13 bytes
_BIN_MAGIC = 0x57
_BIN_VERS = (2, 3)


def _parseBinHeader(payload: bytes):
    """(node_id, ntp_ms, n, ver, body_offset) from a binary batch header; ValueError on a bad header.
    `n` is the authoritative record count — the parser trusts it over the raw byte length."""
    if len(payload) < _BIN_HDR.size:
        raise ValueError(f"UdpSource: bad batch header: {len(payload)} bytes < {_BIN_HDR.size}")
    magic, ver, nodeId, ntpMs, n = _BIN_HDR.unpack_from(payload, 0)
    if magic != _BIN_MAGIC or ver not in _BIN_VERS:
        raise ValueError(f"UdpSource: bad batch header: magic={magic:#x} ver={ver}")
    return nodeId, ntpMs, n, ver, _BIN_HDR.size


def _macToBytes(mac: str) -> bytes:
    """'aa:bb:cc:dd:ee:ff' -> 6 raw bytes (case-insensitive). Normalize a tx_mac filter once."""
    return bytes(int(x, 16) for x in mac.split(":"))


def _iterBinRecords(payload: bytes, offset: int, n: int, ver: int = 2):
    """Yield (mac_bytes, csi complex64, local_ts_us) for up to `n` packed records — the header count
    bounds the loop, so trailing bytes past `n` records are never interpreted as CSI. Stops early on a
    structurally corrupt record (truncated, or wrong byte length) since the stream can't be resynced.
    Record = mac[6] | ts_us(u32 LE) | len(u16 LE) | CSI bytes, laid out [imag0,real0,...]. CSI is
    int8 per component (ver 2, L=2*S) or int16 (ver 3, L=4*S).
    Yields the raw 6 MAC bytes (not a formatted string) so the hot path skips per-record formatting;
    callers compare bytes for the tx_mac filter and only stringify when bucketing/logging."""
    end = len(payload)
    step = 2 if ver == 2 else 4   # bytes per subcarrier: int8 I/Q (2) vs int16 I/Q (4)
    for _ in range(n):
        if offset + 12 > end:                     # 6 (mac) + 4 (ts) + 2 (len): batch shorter than n
            break
        mac = payload[offset:offset + 6]
        localTsUs, L = struct.unpack_from("<IH", payload, offset + 6)
        offset += 12
        if L % step != 0 or offset + L > end:
            break
        if ver == 2:
            d = np.frombuffer(payload, dtype=np.int8, count=L, offset=offset).astype(np.float32)
        else:
            d = np.frombuffer(payload, dtype="<i2", count=L // 2, offset=offset).astype(np.float32)
        offset += L
        csi = (d[1::2] + 1j * d[0::2]).astype(np.complex64)   # csi[k] = complex(real=d[2k+1], imag=d[2k])
        yield mac, csi, localTsUs


def parseBatch(payload: bytes, *, tx_mac=None) -> list:
    """One UDP batch payload -> list[CsiFrame] (node_id + wall-clock timestamps from the header).
    O(n·S).

    Binary v2 wire format: header csi_hdr_t {magic,ver=2,node,ntp_ms,n} then exactly n packed records
    (see _iterBinRecords). A bad/missing header raises ValueError (wiring error). The header's n
    bounds parsing, so trailing bytes are ignored; a structurally corrupt record stops parsing. Records
    whose subcarrier count S differs from the first kept one are skipped (width guard).

    Timestamp scheme: ntp_ms/1000.0 is the batch SEND time ≈ the LAST frame's wall time; each
    frame's absolute time is reconstructed from its local_timestamp offset relative to the last
    record's local_ts:  t_i = ntp_ms/1000 - (last_us - local_ts_i) / 1e6."""
    nodeId, ntpMs, n, ver, off = _parseBinHeader(payload)
    txBytes = _macToBytes(tx_mac) if tx_mac is not None else None

    parsed = []
    sRef = None
    for macB, csi, localTsUs in _iterBinRecords(payload, off, n, ver):
        if txBytes is not None and macB != txBytes:
            continue
        S = int(csi.size)
        if sRef is None:
            sRef = S
        elif S != sRef:
            continue  # mixed-S record: skip
        parsed.append((csi, localTsUs))

    if not parsed:
        return []

    lastUs = parsed[-1][1]
    frames = []
    for csi, localTsUs in parsed:
        # & 0xFFFFFFFF: masks the firmware's low-32-bit esp_timer wrap (~71.6 min), safe within a batch.
        t = ntpMs / 1000.0 - ((lastUs - localTsUs) & 0xFFFFFFFF) / 1e6
        fr = CsiFrame(1, sRef)
        fr.grid[0, :] = csi
        fr.timestamp = t
        fr.node_id = nodeId
        frames.append(fr)
    return frames


def macShort(mac: str) -> str:
    """Last two MAC octets — the compact transmitter label for a link key (e.g. 'ee:ff')."""
    return ":".join(mac.split(":")[-2:]) if ":" in mac else mac


def parseBatchLinks(payload: bytes, *, tx_mac=None) -> dict:
    """One UDP batch -> dict[(tx_short, rx_node) -> list[CsiFrame]], keeping TX identity so each
    directed (tx->rx) link is its OWN stream (the all-pairs fusion input). O(n·S).

    Same binary v2 format + timestamp scheme as parseBatch (rx_node = header node; ntp_ms ≈ the last
    frame's wall time; per-frame t reconstructed from local_timestamp; header n bounds parsing).
    Difference: frames are bucketed by the per-record MAC (the transmitter) instead of merged, and the
    subcarrier-width guard is applied PER LINK (a future 5 GHz arm can carry a different width)."""
    nodeId, ntpMs, n, ver, off = _parseBinHeader(payload)
    txBytes = _macToBytes(tx_mac) if tx_mac is not None else None

    parsed = []  # (tx_short, csi, local_ts_us)
    for macB, csi, localTsUs in _iterBinRecords(payload, off, n, ver):
        if txBytes is not None and macB != txBytes:
            continue
        txShort = f"{macB[4]:02x}:{macB[5]:02x}"   # last two octets == macShort(full mac)
        parsed.append((txShort, csi, localTsUs))
    if not parsed:
        return {}

    lastUs = parsed[-1][2]
    links: dict = {}
    sRef: dict = {}  # per-link width guard
    for txShort, csi, localTsUs in parsed:
        key = (txShort, nodeId)
        S = int(csi.size)
        if key not in sRef:
            sRef[key] = S
        elif S != sRef[key]:
            continue
        fr = CsiFrame(1, S)
        fr.grid[0, :] = csi
        # & 0xFFFFFFFF handles the firmware ts_us low-32-bit wrap (see parseBatch).
        fr.timestamp = ntpMs / 1000.0 - ((lastUs - localTsUs) & 0xFFFFFFFF) / 1e6
        fr.node_id = nodeId
        links.setdefault(key, []).append(fr)
    return links


def resampleUniform(frames, fs_hz):
    """Resample stream to uniform 1/fs_hz grid (linear interpolation). O(n·A·S).

    Mesh rate jitters (30-300 Hz). Resampling fixes fs to fs_hz, preventing dropped windows and smeared spectrograms.

    Pass a single-link stream. Output is sorted."""
    if fs_hz <= 0:
        raise ValueError("resampleUniform: fs_hz must be positive")
    if len(frames) < 2:
        return list(frames)
    A, S = frames[0].grid.shape
    t = np.array([fr.timestamp for fr in frames], dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t = t[order]
    flat = np.stack([frames[i].grid.reshape(-1) for i in order]).astype(np.complex64)  # (n, A·S)
    nOut = max(2, int((t[-1] - t[0]) * fs_hz) + 1)
    tg = t[0] + np.arange(nOut, dtype=np.float64) / fs_hz
    re = np.empty((nOut, A * S), dtype=np.float32)
    im = np.empty_like(re)
    for c in range(A * S):  # per-cell 1-D interp; A·S is small (≈64 for HT20) so this is cheap
        re[:, c] = np.interp(tg, t, flat[:, c].real)
        im[:, c] = np.interp(tg, t, flat[:, c].imag)
    out = (re + 1j * im).astype(np.complex64).reshape(nOut, A, S)
    nodeId = getattr(frames[0], "node_id", 0)
    result = []
    for i in range(nOut):
        fr = CsiFrame(A, S)
        fr.grid[:, :] = out[i]
        fr.timestamp = float(tg[i])
        fr.node_id = nodeId
        result.append(fr)
    return result


def bindUdp(port, *, timeout=None):
    """A UDP socket bound to `port` with a large (8 MB) kernel receive buffer. The big buffer absorbs
    bursts from N nodes while the main thread is busy in inference — without it the kernel drops
    datagrams before the single-threaded receiver reads them. The OS may cap the size (on macOS raise
    kern.ipc.maxsockbuf); the setsockopt failing is non-fatal."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    except OSError:
        pass
    sock.bind(("0.0.0.0", port))
    if timeout is not None:
        sock.settimeout(timeout)
    return sock


class CsiSource(ABC):
    """A stream of CsiFrames feeding the front-end."""

    @abstractmethod
    def frames(self):
        """Yield CsiFrame objects in capture order."""


class SyntheticSource(CsiSource):
    """Replay an in-memory frame list (e.g. from wavetrace.Synthetic)."""

    def __init__(self, frames):
        self._frames = list(frames)

    def frames(self):
        return iter(self._frames)


class RecordingSource(CsiSource):
    """Replay frames saved by `saveRecording`. Reconstructs each CsiFrame on demand."""

    def __init__(self, rec_dir):
        self._dir = Path(rec_dir)

    def frames(self):
        return loadRecording(self._dir)


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceOptions:
    """Settings shared by every `CsiSource` transport: how long to wait for the next packet with
    no data before ending the stream, and an optional cap on total frames yielded."""

    timeout_seconds: float = 5.0
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when given")


@dataclass(frozen=True, slots=True, kw_only=True)
class UdpSourceOptions(SourceOptions):
    """Settings for `UdpSource` (batched-UDP CSI backhaul)."""

    port: int = 5566
    tx_mac: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.port < 65536:
            raise ValueError("port must be in (0, 65536)")


class UdpSource(CsiSource):
    """Receive batched-UDP CSI (plan §3 backhaul). frames() binds 0.0.0.0:port and yields until
    timeout_seconds elapses with no packet (or max_frames). The socket loop is a thin shell over
    parseBatch."""

    def __init__(self, options: UdpSourceOptions = UdpSourceOptions()):
        self._port = options.port
        self._tx_mac = options.tx_mac
        self._timeout_seconds = options.timeout_seconds
        self._max_frames = options.max_frames

    def frames(self):
        """Bind UDP socket and yield CsiFrames; stop on timeout or max_frames."""
        sock = bindUdp(self._port, timeout=self._timeout_seconds)
        try:
            count = 0
            while self._max_frames is None or count < self._max_frames:
                try:
                    payload, _ = sock.recvfrom(65535)
                except socket.timeout:
                    return
                for fr in parseBatch(payload, tx_mac=self._tx_mac):
                    yield fr
                    count += 1
                    if self._max_frames is not None and count >= self._max_frames:
                        return
        finally:
            sock.close()


@dataclass(frozen=True, slots=True, kw_only=True)
class SerialSourceOptions(SourceOptions):
    """Settings for `SerialReader` (esp-csi over USB serial)."""

    device: str                       # e.g. "/dev/tty.usbserial-…"
    baud: int = 921600
    node_id: int = 1
    tx_mac: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.device:
            raise ValueError("device must be non-empty")
        if self.baud <= 0:
            raise ValueError("baud must be positive")
        if self.node_id < 0:
            raise ValueError("node_id must be >= 0")


class SerialReader(CsiSource):
    """esp-csi CSI over USB serial — one ESP32 = one node = one antenna. A thin pyserial shell over
    `parseCsiLine`; the no-RF-cost bring-up path for 1–2 nodes before the UDP mesh exists.

    Frame timestamp = PC wall-clock at read, NOT the ESP's `local_timestamp` counter: independent ESP
    clocks have no common origin, so the PC arrival time is the consistent basis for aligning against
    camera labels and other nodes (sub-ms serial latency ≪ the front-end window). `tx_mac` filters to
    the dedicated TX (drops beacons / foreign traffic). `pip install pyserial`; the import + port open
    are deferred to frames() so importing this module never requires pyserial."""

    def __init__(self, options: SerialSourceOptions):
        self._port = options.device
        self._baud = options.baud
        self._node = options.node_id
        self._tx_mac = options.tx_mac
        self._timeout_seconds = options.timeout_seconds
        self._max_frames = options.max_frames

    def frames(self):
        """Open the serial port and yield (1, S) CsiFrames tagged with node_id; stop on read timeout
        (no data within timeout_seconds) or max_frames. Malformed/filtered lines are skipped silently.

        Subcarrier-count guard (same as parseBatch): even from the dedicated TX, an RX occasionally
        receives off-format frames (legacy/HT20/HT40 differ in S). The first yielded frame sets S_ref
        and any line with a different S is dropped, so the stream is a fixed (1, S) shape downstream."""
        try:
            import serial  # pyserial
        except ImportError as e:
            raise ImportError("SerialReader needs pyserial: pip install pyserial") from e
        ser = serial.Serial(self._port, self._baud, timeout=self._timeout_seconds)
        try:
            count = 0
            sRef = None
            while self._max_frames is None or count < self._max_frames:
                raw = ser.readline()
                if not raw:
                    return  # timeout with no data -> stream ended
                result = parseCsiLine(raw.decode("utf-8", errors="replace"), tx_mac=self._tx_mac)
                if result is None:
                    continue
                csi, _localTsUs, _mac = result
                S = int(csi.size)
                if sRef is None:
                    sRef = S
                elif S != sRef:
                    continue  # off-format frame (different bandwidth/mode) -> drop to keep fixed shape
                fr = CsiFrame(1, sRef)
                fr.grid[0, :] = csi
                fr.timestamp = time.time()
                fr.node_id = self._node
                yield fr
                count += 1
        finally:
            ser.close()


@dataclass(frozen=True, slots=True, kw_only=True)
class NexmonSourceOptions(SourceOptions):
    """Settings for `NexmonSource` (5 GHz nexmon CSI on a Raspberry Pi).

    Exactly one capture mode must be selected: a `pcap_path` to replay, or `live=True` to spawn
    tcpdump on `iface`."""

    pcap_path: str | None = None
    iface: str = "wlan0"
    live: bool = False
    node_id: int = 100                # >=100 so it never collides with ESP32 ids 1..6
    bandwidth_mhz: int = 80

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.node_id < 0:
            raise ValueError("node_id must be >= 0")
        if self.bandwidth_mhz <= 0:
            raise ValueError("bandwidth_mhz must be positive")
        if not self.live and not self.pcap_path:
            raise ValueError("NexmonSource: set pcap_path or live=True")


class NexmonSource(CsiSource):
    """5 GHz CSI from a Raspberry Pi running nexmon_csi (bcm43455c0; Pi 3B+/4B/5, fw 7_45_189).

    Reads nexmon-encapsulated CSI from a pcap file or a live tcpdump stream.
    Decoding is delegated to `csiread.Nexmon` (pip install csiread) — only `_decodeFile` changes
    if your csiread version or chip differs.

    node_id=100 by default so it never collides with ESP32 ids 1..6, and the 2.4/5 GHz split
    downstream keys on node_id >= 100 == 5 GHz (matching NodeHealthMeter's band convention).

    Two modes:
      pcap_path set  -> replay a captured file (offline dev, no hardware).
      live=True      -> spawn tcpdump on `iface` and parse packets as they arrive."""

    def __init__(self, options: NexmonSourceOptions):
        self._pcap = options.pcap_path
        self._iface = options.iface
        self._live = options.live
        self._node = options.node_id
        self._timeout = options.timeout_seconds
        self._max = options.max_frames
        self._bw = options.bandwidth_mhz

    def _csiread(self):
        try:
            import csiread
            return csiread
        except ImportError as e:
            raise ImportError("NexmonSource needs csiread: pip install csiread") from e

    def _decodeFile(self):
        csiread = self._csiread()
        reader = csiread.Nexmon(self._pcap, chip="43455c0", bw=self._bw)
        reader.read()
        csi = np.asarray(reader.csi)           # (F, S) complex
        tsRaw = getattr(reader, "sec", None)
        ts = np.asarray(tsRaw, dtype=float) if tsRaw is not None else np.arange(len(csi)) / 100.0
        for i in range(len(csi)):
            fr = CsiFrame(1, csi.shape[1])
            fr.grid[0, :] = csi[i].astype(np.complex64)
            fr.timestamp = float(ts[i]) if i < ts.size else i / 100.0
            fr.node_id = self._node
            yield fr

    def _decodeLive(self):
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
                for fr in self._decodeFile():
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
        # options.__post_init__ already guarantees live or pcap_path is set, so no third branch.
        if self._live:
            yield from self._decodeLive()
        else:
            yield from self._decodeFile()


def saveRecording(frames, out_dir) -> Path:
    """Serialize a CsiFrame list to out_dir (grid/t/node_id .npy + meta.json). O(F·A·S)."""
    frames = list(frames)
    if not frames:
        raise ValueError("saveRecording: no frames")
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


def loadRecording(rec_dir):
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


def parseTimeSpans(spec: str) -> list[tuple[float, float]]:
    """'a:b,c:d' -> [(a,b),(c,d)]; '' -> []."""
    if not spec:
        return []
    return [tuple(float(x) for x in part.split(":")) for part in spec.split(",")]


def buildCsiSource(options) -> CsiSource:
    """Build a CsiSource from a small options bag: `options.recording` (replay a saved directory) or
    `options.synthetic` (generate frames in-process via `wavetrace.Synthetic`, from
    `.antennas`/`.subcarriers`/`.fs`/`.duration`/`.presence`/`.weapon`/`.weapon_depth`/`.seed`).
    Duck-typed so both the CLI's argparse `Namespace` and the web dashboard's request options work
    unchanged."""
    if options.recording:
        return RecordingSource(options.recording)
    if options.synthetic:
        from wavetrace.Synthetic import generatePairedRecording
        if parseTimeSpans(options.weapon) and options.weapon_depth <= 0.0:
            # depth 0 injects no signal -> weapon windows are unlearnable (single-class); warn (B3)
            warnings.warn("synthetic --weapon spans set but --weapon-depth is 0: weapon windows will "
                          "carry no signature (pass --weapon-depth > 0)", stacklevel=2)
        spans = parseTimeSpans(options.presence)
        frames, _, _ = generatePairedRecording(
            numAntennas=options.antennas, numSubcarriers=options.subcarriers, sampleRateHz=options.fs,
            durationS=options.duration, cameraFps=30.0, presenceSpans=spans or [(0.0, options.duration)],
            presenceTurbulenceStd=0.10, weaponSpans=parseTimeSpans(options.weapon),
            weaponSignatureDepth=options.weapon_depth, seed=options.seed,
        )
        return SyntheticSource(frames)
    raise SystemExit("a source is required: --recording DIR or --synthetic")
