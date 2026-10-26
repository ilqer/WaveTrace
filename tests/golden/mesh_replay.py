"""Offline mesh replay shared by the script goldens: seeded synthetic links, quantized to the int16
wire, encoded into real UDP batch datagrams, and a fake wall clock."""

import collections
import hashlib
import json
import socket
import struct
from pathlib import Path

import numpy as np

from fixtures.synthetic_csi import generate_quiet_baseline
from wavetrace.adapters.store.model_runs import (
    CURRENT_FILE_NAME, CURRENT_LINK_NAME, MANIFEST_FILE_NAME, find_current_run_dir, is_run_id,
)
from wavetrace.domain.groundtruth.labeler import ScriptedLabeler
from wavetrace.pipeline import calibrate_source, collect_source
from wavetrace.recognition import train_presence, train_weapon
from wavetrace.synthetic import generate_paired_recording

SAMPLE_RATE_HZ = 100.0
NUM_SUBCARRIERS = 32
IQ_SCALE = 1000.0
BASELINE_PACKETS = 100
RX_NODE_IDS = (1, 2)
TX_MACS = ("02:00:00:00:4f:9c", "02:00:00:00:64:b8")
CLOCK_START_S = 1000.0
FRAMES_PER_DATAGRAM = 5

WIRE_MAGIC, WIRE_VERSION_INT16 = 0x57, 3

WINDOW, HOP = 32, 16
TRAIN_DURATION_S = 6.0
LIVE_DURATION_S = 10.0
COUNT_TURBULENCE_STD_DEVIATION = {0: 0.0, 1: 0.1, 2: 0.3}
COUNT_SEGMENT_S = 2.0
COUNT_LIVE_LEVELS = (0, 1, 2, 1, 0)

IDLE_TIMEOUTS = 3
IDLE_TIMEOUT_S = 0.5
RECEIVE_STEP_S = 0.025


class FrameListSource:
    def __init__(self, frames):
        self._frames = frames

    def frames(self):
        return iter(self._frames)


class FakeClock:
    def __init__(self, start_s):
        self.now_s = start_s

    def __call__(self):
        return self.now_s


def link_seed(rx_node_id, tx_index):
    return 10 * rx_node_id + tx_index


def quantize(frames):
    for frame in frames:
        grid = np.asarray(frame.grid)
        frame.grid[:, :] = np.round(grid.real * IQ_SCALE) + 1j * np.round(grid.imag * IQ_SCALE)
    return frames


def generate_link_frames(rx_node_id, tx_index, duration_s, seed_offset=0, **paired_kwargs):
    frames, _, _ = generate_paired_recording(
        num_antennas=1, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        duration_s=duration_s, camera_frame_rate_hz=10.0, seed=link_seed(rx_node_id, tx_index) + seed_offset,
        **paired_kwargs)
    return quantize(frames)


def shift_timestamps(frames, offset_s):
    for frame in frames:
        frame.timestamp += offset_s
    return frames


def calibrate_nodes(root):
    for rx_node_id in RX_NODE_IDS:
        baseline = quantize(generate_quiet_baseline(
            num_antennas=1, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
            num_frames=BASELINE_PACKETS, seed=link_seed(rx_node_id, 0)))
        calibrate_source(FrameListSource(baseline), root / "cal" / f"node{rx_node_id}",
                         baseline_packets=BASELINE_PACKETS)


def encode_record(tx_mac, timestamp_s, grid):
    iq = np.empty(2 * grid.size, dtype="<i2")
    iq[0::2], iq[1::2] = grid.imag, grid.real  # wire layout is [imag, real] per subcarrier
    mac = bytes(int(octet, 16) for octet in tx_mac.split(":"))
    local_timestamp_us = int(round(timestamp_s * 1e6)) & 0xFFFFFFFF
    return mac + struct.pack("<IH", local_timestamp_us, iq.nbytes) + iq.tobytes()


def encode_datagrams(rx_node_id, frames_by_tx_index):
    """Batches of FRAMES_PER_DATAGRAM time steps, the tx links interleaved record by record."""
    num_frames = len(next(iter(frames_by_tx_index.values())))
    datagrams = []
    for start in range(0, num_frames, FRAMES_PER_DATAGRAM):
        records = [
            encode_record(TX_MACS[tx_index], frames[i].timestamp, np.asarray(frames[i].grid)[0])
            for i in range(start, min(start + FRAMES_PER_DATAGRAM, num_frames))
            for tx_index, frames in frames_by_tx_index.items()]
        last_timestamp_s = frames_by_tx_index[next(iter(frames_by_tx_index))][
            min(start + FRAMES_PER_DATAGRAM, num_frames) - 1].timestamp
        ntp_ms = int(round((CLOCK_START_S + last_timestamp_s) * 1000))
        header = struct.pack("<BBBQH", WIRE_MAGIC, WIRE_VERSION_INT16, rx_node_id, ntp_ms, len(records))
        datagrams.append(header + b"".join(records))
    return datagrams


def interrupt():
    raise KeyboardInterrupt


class FakeUdpSocket:
    """Replays datagrams; None is an idle recv timeout. Each recv advances the clock; a recv after the
    last datagram calls `on_drained`, which must raise."""

    def __init__(self, clock, datagrams, on_drained=interrupt):
        self._clock = clock
        self._pending = collections.deque(datagrams)
        self._on_drained = on_drained

    def setsockopt(self, *option):
        pass

    def settimeout(self, timeout_s):
        pass

    def bind(self, address):
        pass

    def recvfrom(self, buffer_size):
        if not self._pending:
            self._on_drained()
        payload = self._pending.popleft()
        if payload is None:
            self._clock.now_s += IDLE_TIMEOUT_S
            raise socket.timeout
        self._clock.now_s += RECEIVE_STEP_S
        return payload, ("127.0.0.1", 0)

    def close(self):
        pass


def collect_link(root, rx_node_id, name, frames, spans, stage, session_id, labeler=None):
    dataset_dir, _ = collect_source(
        FrameListSource(frames), root / "cal" / f"node{rx_node_id}",
        root / "ds" / f"node{rx_node_id}" / name, spans, stage=stage, window=WINDOW, hop=HOP,
        session_id=session_id, subject_id="subject", labeler=labeler)
    return dataset_dir


def train_presence_nodes(root):
    for rx_node_id in RX_NODE_IDS:
        dataset_dirs = [
            collect_link(root, rx_node_id, f"tx{tx_index}",
                         generate_link_frames(rx_node_id, tx_index, TRAIN_DURATION_S,
                                              presence_spans=[(2.0, 4.0)], presence_turbulence_std=0.15),
                         [(2.0, 4.0)], "presence", f"s{tx_index}")
            for tx_index in range(len(TX_MACS))]
        train_presence(dataset_dirs, out_dir=root / "model" / f"node{rx_node_id}")


def train_count_nodes(root):
    for rx_node_id in RX_NODE_IDS:
        dataset_dirs = []
        for tx_index in range(len(TX_MACS)):
            for count, turbulence_std in COUNT_TURBULENCE_STD_DEVIATION.items():
                frames = generate_link_frames(rx_node_id, tx_index, TRAIN_DURATION_S,
                                              presence_spans=[(0.0, TRAIN_DURATION_S)],
                                              presence_turbulence_std=turbulence_std)
                labeler = ScriptedLabeler(
                    [(0.0, TRAIN_DURATION_S, True)],
                    label_function=lambda raw, timestamp_s, class_id=count: (class_id, str(class_id)))
                dataset_dirs.append(collect_link(
                    root, rx_node_id, f"tx{tx_index}_c{count}", frames, [], "presence", f"s{tx_index}",
                    labeler=labeler))
        train_presence(dataset_dirs, out_dir=root / "model_count" / f"node{rx_node_id}")


def weapon_datasets(root, rx_node_id, tx_indices):
    return [
        collect_link(root, rx_node_id, f"tx{tx_index}",
                     generate_link_frames(rx_node_id, tx_index, TRAIN_DURATION_S, weapon_spans=[(2.0, 4.0)],
                                          weapon_signature_depth=0.4),
                     [(2.0, 4.0)], "weapon", f"s{tx_index}")
        for tx_index in tx_indices]


def train_weapon_heads(root):
    """Node 1 has a head for tx0 only, so its tx1 link is ignored; node 2 has one per tx."""
    train_weapon(weapon_datasets(root, 1, [0]), out_dir=root / "model_weapon" / "node1" / "link4f9c")
    for tx_index, link_tag in enumerate(("4f9c", "64b8")):
        train_weapon(weapon_datasets(root, 2, [tx_index]),
                     out_dir=root / "model_weapon" / "node2" / f"link{link_tag}")


def presence_live_frames(rx_node_id, tx_index):
    return generate_link_frames(rx_node_id, tx_index, LIVE_DURATION_S, presence_spans=[(3.0, 6.0)],
                                presence_turbulence_std=0.15)


def count_live_frames(rx_node_id, tx_index):
    frames = []
    for segment, count in enumerate(COUNT_LIVE_LEVELS):
        frames += shift_timestamps(
            generate_link_frames(rx_node_id, tx_index, COUNT_SEGMENT_S,
                                 presence_spans=[(0.0, COUNT_SEGMENT_S)] if count else [],
                                 presence_turbulence_std=COUNT_TURBULENCE_STD_DEVIATION[count]),
            segment * COUNT_SEGMENT_S)
    return frames


def weapon_live_frames(rx_node_id, tx_index):
    return generate_link_frames(rx_node_id, tx_index, LIVE_DURATION_S, weapon_spans=[(3.0, 6.0)],
                                weapon_signature_depth=0.4)


def live_datagrams(live_frames):
    """Both rx nodes alternate within each time step; leading idle timeouts exercise the empty tick."""
    per_node = [
        encode_datagrams(rx_node_id, {tx_index: live_frames(rx_node_id, tx_index)
                                   for tx_index in range(len(TX_MACS))})
        for rx_node_id in RX_NODE_IDS]
    interleaved = [datagram for step in zip(*per_node) for datagram in step]
    return [None] * IDLE_TIMEOUTS + interleaved


def _find_served_path(path):
    """`path` as model loaders serve it: a file of a model dir's current run at `<model_dir>/<file>`; None
    for the run pointer, an older run and the run manifest (training time, repo state, library versions)."""
    for i, part in enumerate(path.parts):
        if part == CURRENT_LINK_NAME:
            return None
        if is_run_id(part):
            run_dir = Path(*path.parts[:i + 1])
            if path.name == MANIFEST_FILE_NAME or find_current_run_dir(run_dir.parent) != run_dir:
                return None
            return Path(*path.parts[:i], *path.parts[i + 1:])
    return None if path.name == CURRENT_FILE_NAME else path


def hash_tree(root):
    """One 'sha256  relative/path' line per file under `root` as the loaders serve it (each model dir's
    current run at the model dir's own path); metrics.json is hashed without `fit_seconds`, its only
    wall-clock field."""
    lines_by_path = {}
    for path in Path(root).rglob("*"):
        served_path = None if path.is_dir() else _find_served_path(path)
        if served_path is None:
            continue
        content = path.read_bytes()
        if path.name == "metrics.json":
            metrics = json.loads(content)
            metrics.pop("fit_seconds", None)
            content = json.dumps(metrics, sort_keys=True).encode()
        relative_path = served_path.relative_to(root).as_posix()
        lines_by_path[served_path] = f"{hashlib.sha256(content).hexdigest()}  {relative_path}"
    return "\n".join(lines_by_path[path] for path in sorted(lines_by_path)) + "\n"
