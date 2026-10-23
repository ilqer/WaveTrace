"""Characterization tests: pin the payloads the web dashboard's `WaveTraceRunner.run_inference`
publishes, for the presence, count and weapon mesh roots and a single-node file model fed by the
synthetic source.

Everything is offline and deterministic. Patched seams: `socket.socket` (a fake UDP socket replaying
real batch datagrams, which stops the runner once drained), `time.time` (a fake clock), `time.sleep`,
and `asyncio.run_coroutine_threadsafe`, with queues whose `put` records synchronously.

Two snapshots per case in tests/golden/expected/: web_<case>_keys.txt, the key paths per message type,
which the UI depends on and must never change; and web_<case>_values.txt, every message in emission
order with scalars rounded and arrays reduced to their shapes. Re-generate deliberately with
`WAVETRACE_UPDATE_GOLDEN=1 pytest tests/golden`.
"""

import asyncio
import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fixtures.synthetic_csi import generate_quiet_baseline
from mesh_replay import (
    CLOCK_START_S, HOP, LIVE_DURATION_S, WINDOW, FakeClock, FakeUdpSocket, FrameListSource,
    calibrate_nodes, count_live_frames, generate_link_frames, live_datagrams,
    presence_live_frames, train_count_nodes, train_presence_nodes, train_weapon_heads, weapon_live_frames,
)
from wavetrace.pipeline import calibrate_source, collect_source
from wavetrace.recognition import train_presence
from wavetrace.synthetic import generate_paired_recording
from web.streamer import WaveTraceRunner

EXPECTED = Path(__file__).parent / "expected"

FILE_NUM_SUBCARRIERS = 32
FILE_SAMPLE_RATE_HZ = 100.0
FILE_DURATION_S = 12.0
FILE_SEED = 7


class _RecordingQueue:
    def __init__(self, messages, queue_name):
        self._messages = messages
        self._queue_name = queue_name

    def put(self, message):
        self._messages.append((self._queue_name, message))


def _make_runner(messages):
    queues = {name: _RecordingQueue(messages, name)
              for name in ("inference", "stream", "logs", "training", "telemetry")}
    return WaveTraceRunner(None, queues["inference"], queues["stream"], queues["logs"], queues["training"],
                           queues["telemetry"])


def _patch_runtime(monkeypatch, runner, datagrams):
    clock = FakeClock(CLOCK_START_S)

    def stop_runner():
        runner.is_running = False
        raise socket.timeout

    fake_socket = FakeUdpSocket(clock, datagrams, on_drained=stop_runner)
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: fake_socket)
    monkeypatch.setattr(time, "time", clock)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coroutine, loop: None)


def _classify(queue_name, payload):
    if queue_name != "inference":
        return queue_name
    if payload.get("event") == "pipeline_done":
        return "end"
    return "alert" if "event" in payload else "inference"


def run_mesh(tmp_path, monkeypatch, model_dir_name, mode, live_frames):
    messages = []
    runner = _make_runner(messages)
    _patch_runtime(monkeypatch, runner, live_datagrams(live_frames))
    runner.run_inference(None, str(tmp_path / "cal"), str(tmp_path / model_dir_name), mode, port=5599)
    return messages


def run_file(tmp_path, monkeypatch):
    messages = []
    runner = _make_runner(messages)
    _patch_runtime(monkeypatch, runner, [])
    request = SimpleNamespace(
        action="run", synthetic=True, antennas=1, subcarriers=FILE_NUM_SUBCARRIERS, fs=FILE_SAMPLE_RATE_HZ,
        duration=FILE_DURATION_S, seed=FILE_SEED, col_spans="", udp_port=5599,
        calibration=str(tmp_path / "file_cal"), model=str(tmp_path / "file_model" / "model.joblib"),
        mode="presence", vote=True, gainLock=True, frame_average=1, use_baseline=False)
    runner.run_inference_request(request)
    return messages


def published(messages):
    """[(message type, payload)] in emission order; of the logs only runner errors are kept, as
    ("error", text without the time of day)."""
    entries = []
    for queue_name, message in messages:
        if queue_name != "logs":
            payload = json.loads(message)
            entries.append((_classify(queue_name, payload), payload))
        elif message.split("] ", 1)[1].startswith("ERROR"):
            entries.append(("error", message.split("] ", 1)[1]))
    return entries


def _key_paths(value, prefix=""):
    if isinstance(value, dict):
        paths = set()
        for key, item in value.items():
            paths.add(prefix + key)
            paths |= _key_paths(item, prefix + key + ".")
        return paths
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return set().union(*(_key_paths(item, prefix.rstrip(".") + "[].") for item in value))
    return set()


def format_key_snapshot(published_messages):
    """One line per (type, key path); '?' marks a path only some messages of that type carry."""
    paths_by_type = {}
    for message_type, payload in published_messages:
        if message_type != "error":
            paths_by_type.setdefault(message_type, []).append(_key_paths(payload))
    lines = []
    for message_type in sorted(paths_by_type):
        path_sets = paths_by_type[message_type]
        always = set.intersection(*path_sets)
        for path in sorted(set.union(*path_sets)):
            lines.append(f"{message_type}: {path}{'' if path in always else '?'}")
    return "\n".join(lines) + "\n"


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return f"shape{np.shape(value)}" if not any(isinstance(item, dict) for item in value) else None
    return repr(value)


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        return [token for key, item in value.items() for token in _flatten(item, prefix + key + ".")]
    if isinstance(value, list) and value and any(isinstance(item, dict) for item in value):
        return [token for i, item in enumerate(value) for token in _flatten(item, f"{prefix.rstrip('.')}[{i}].")]
    return [f"{prefix.rstrip('.')}={_format_value(value)}"]


def format_value_snapshot(published_messages):
    return "".join(f"{message_type} {payload if message_type == 'error' else ' '.join(_flatten(payload))}\n"
                   for message_type, payload in published_messages)


def assert_matches_golden(case, published_messages):
    for kind, text in (("keys", format_key_snapshot(published_messages)),
                       ("values", format_value_snapshot(published_messages))):
        expected_path = EXPECTED / f"web_{case}_{kind}.txt"
        if os.environ.get("WAVETRACE_UPDATE_GOLDEN"):
            expected_path.write_bytes(text.encode())
        assert text == expected_path.read_bytes().decode()


def train_file_model(root):
    baseline = generate_quiet_baseline(num_antennas=1, num_subcarriers=FILE_NUM_SUBCARRIERS,
                                       sample_rate_hz=FILE_SAMPLE_RATE_HZ, num_frames=300, seed=FILE_SEED)
    calibrate_source(FrameListSource(baseline), root / "file_cal", baseline_packets=300)
    spans = [(2.0, 4.0), (6.0, 8.0)]
    frames, _, _ = generate_paired_recording(
        num_antennas=1, num_subcarriers=FILE_NUM_SUBCARRIERS, sample_rate_hz=FILE_SAMPLE_RATE_HZ,
        duration_s=10.0, camera_frame_rate_hz=30.0, presence_spans=spans, presence_turbulence_std=0.10,
        seed=FILE_SEED + 1)
    dataset_dir, _ = collect_source(FrameListSource(frames), root / "file_cal", root / "file_ds" / "s0", spans,
                                    stage="presence", window=WINDOW, hop=HOP, session_id="s0",
                                    subject_id="subject")
    train_presence([dataset_dir], out_dir=root / "file_model")


def _assert_ran_to_the_end(published_messages):
    assert [payload for message_type, payload in published_messages if message_type == "error"] == []
    assert published_messages[-1] == ("end", {"event": "pipeline_done"})


def test_web_presence_mesh_payloads(tmp_path, monkeypatch):
    calibrate_nodes(tmp_path)
    train_presence_nodes(tmp_path)
    messages = run_mesh(tmp_path, monkeypatch, "model", "presence", presence_live_frames)
    published_messages = published(messages)
    _assert_ran_to_the_end(published_messages)
    assert_matches_golden("presence_mesh", published_messages)


def test_web_count_mesh_payloads(tmp_path, monkeypatch):
    calibrate_nodes(tmp_path)
    train_count_nodes(tmp_path)
    messages = run_mesh(tmp_path, monkeypatch, "model_count", "count", count_live_frames)
    published_messages = published(messages)
    _assert_ran_to_the_end(published_messages)
    assert_matches_golden("count_mesh", published_messages)


def test_web_weapon_mesh_payloads(tmp_path, monkeypatch):
    calibrate_nodes(tmp_path)
    train_weapon_heads(tmp_path)
    messages = run_mesh(tmp_path, monkeypatch, "model_weapon", "weapon", weapon_live_frames)
    published_messages = published(messages)
    _assert_ran_to_the_end(published_messages)
    assert_matches_golden("weapon_mesh", published_messages)


def test_web_presence_file_payloads(tmp_path, monkeypatch):
    train_file_model(tmp_path)
    messages = run_file(tmp_path, monkeypatch)
    published_messages = published(messages)
    _assert_ran_to_the_end(published_messages)
    assert_matches_golden("presence_file", published_messages)



def _present_throughout(rx_node_id, tx_index):
    return generate_link_frames(rx_node_id, tx_index, LIVE_DURATION_S, presence_spans=[(0.0, LIVE_DURATION_S)],
                                presence_turbulence_std=0.15)


def _armed_throughout(rx_node_id, tx_index):
    return generate_link_frames(rx_node_id, tx_index, LIVE_DURATION_S, weapon_spans=[(0.0, LIVE_DURATION_S)],
                                weapon_signature_depth=0.4)


def test_web_presence_mesh_never_raises_a_weapon_alert(tmp_path, monkeypatch):
    calibrate_nodes(tmp_path)
    train_presence_nodes(tmp_path)
    published_messages = published(run_mesh(tmp_path, monkeypatch, "model", "presence", _present_throughout))
    verdicts = [payload["class"] for message_type, payload in published_messages if message_type == "inference"]
    assert len(verdicts) >= 4 and set(verdicts) == {1}
    assert [message_type for message_type, _ in published_messages if message_type in ("alert", "error")] == []
    assert {payload["alert_active"] for message_type, payload in published_messages
            if message_type == "telemetry"} == {False}


def test_web_weapon_mesh_still_raises_the_weapon_alert(tmp_path, monkeypatch):
    calibrate_nodes(tmp_path)
    train_weapon_heads(tmp_path)
    published_messages = published(run_mesh(tmp_path, monkeypatch, "model_weapon", "weapon", _armed_throughout))
    assert [payload["event"] for message_type, payload in published_messages
            if message_type == "alert"] == ["weapon_alert"]
    assert True in {payload["alert_active"] for message_type, payload in published_messages
                    if message_type == "telemetry"}


class _ObservedFakeSocket(FakeUdpSocket):
    """Calls `on_receive(num_received)` after each datagram and records `close`."""

    def __init__(self, clock, datagrams, on_receive):
        super().__init__(clock, datagrams)
        self._on_receive = on_receive
        self._num_received = 0
        self.num_pending = len(datagrams)
        self.is_closed = False

    def recvfrom(self, buffer_size):
        received = super().recvfrom(buffer_size)
        self._num_received += 1
        self.num_pending -= 1
        self._on_receive(self._num_received)
        return received

    def close(self):
        self.is_closed = True


def _run_mesh_observed(tmp_path, monkeypatch, on_receive):
    calibrate_nodes(tmp_path)
    train_presence_nodes(tmp_path)
    messages = []
    runner = _make_runner(messages)
    _patch_runtime(monkeypatch, runner, [])
    fake_socket = _ObservedFakeSocket(time.time, live_datagrams(presence_live_frames),
                                      lambda num_received: on_receive(runner, num_received))
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: fake_socket)
    runner.run_inference(None, str(tmp_path / "cal"), str(tmp_path / "model"), "presence", port=5599)
    return runner, fake_socket, published(messages)


def test_web_stop_ends_the_mesh_run_and_closes_the_socket(tmp_path, monkeypatch):
    def stop_after_a_hundred(runner, num_received):
        if num_received == 100:
            runner.is_running = False

    runner, fake_socket, published_messages = _run_mesh_observed(tmp_path, monkeypatch, stop_after_a_hundred)
    assert fake_socket.num_pending > 0 and fake_socket.is_closed and not runner.is_running
    assert published_messages[-1] == ("end", {"event": "pipeline_done"})


def test_web_mesh_error_is_logged_and_closes_the_socket(tmp_path, monkeypatch):
    def fail_after_a_hundred(runner, num_received):
        if num_received == 100:
            raise OSError("network is down")

    runner, fake_socket, published_messages = _run_mesh_observed(tmp_path, monkeypatch, fail_after_a_hundred)
    assert fake_socket.is_closed and not runner.is_running
    assert ("error", "ERROR Runner: OSError: network is down") in published_messages
    assert published_messages[-1] == ("end", {"event": "pipeline_done"})
