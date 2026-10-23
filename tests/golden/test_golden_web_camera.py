"""Characterization tests: pin what the web dashboard's `WaveTraceRunner.run_camera_collection_request`
logs and publishes, and the tree it writes, for the presence and weapon stages (per-link on).

Offline and deterministic like test_golden_collect_camera.py; `asyncio.run_coroutine_threadsafe` is
patched and the queues record each `put` synchronously. Log lines are kept without their time of day.
Re-generate deliberately with `WAVETRACE_UPDATE_GOLDEN=1 pytest tests/golden`.
"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from camera_replay import DURATION_S, capture_datagrams, patch_camera, patch_mesh, present_person, visible_knife
from mesh_replay import calibrate_nodes, hash_tree
from web.streamer import WaveTraceRunner

EXPECTED = Path(__file__).parent / "expected"
WEB_FRAME_RATE_HZ = 5.0
QUEUE_NAMES = ("inference", "stream", "logs", "training", "telemetry")


class _RecordingQueue:
    def __init__(self, messages, queue_name):
        self._messages = messages
        self._queue_name = queue_name

    def put(self, message):
        self._messages.append((self._queue_name, message))


def run_camera_collection(root, stage, datagrams, monkeypatch):
    patch_camera(monkeypatch, WEB_FRAME_RATE_HZ)
    patch_mesh(monkeypatch, datagrams)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coroutine, loop: None)
    messages = []
    queues = [_RecordingQueue(messages, name) for name in QUEUE_NAMES]
    runner = WaveTraceRunner(None, *queues)
    request = SimpleNamespace(calibration=str(root / "cal"), train_data=str(root), duration=DURATION_S, cam_index=0,
                              per_link=True, col_stage=stage, yolo_weights="yolov8n-seg.pt", udp_port=5599)
    runner.run_camera_collection_request(request)
    monkeypatch.undo()
    assert not runner.is_running
    assert not any("\a" in message for queue_name, message in messages), "the dashboard log rings no bell"
    return messages


def format_messages(messages, root):
    lines, key_paths = [], set()
    for queue_name, message in messages:
        if queue_name == "logs":
            lines.append(f"log {message.split('] ', 1)[1]}")
        else:
            lines.append(f"{queue_name} {message}")
            key_paths |= {f"{queue_name}: {key}" for key in json.loads(message)}
    text = "\n".join(lines) + "\n### keys\n" + "\n".join(sorted(key_paths)) + "\n"
    return text.replace(str(root.parent), "<tmp>")


def assert_matches_golden(name, messages, root):
    snapshot = f"{format_messages(messages, root)}\n### tree\n{hash_tree(root)}"
    expected_path = EXPECTED / f"web_camera_{name}.txt"
    if os.environ.get("WAVETRACE_UPDATE_GOLDEN"):
        expected_path.write_bytes(snapshot.encode())
    assert snapshot == expected_path.read_bytes().decode()


def test_web_camera_presence_pins_output(tmp_path, monkeypatch):
    root = tmp_path / "root"
    calibrate_nodes(root)
    messages = run_camera_collection(root, "presence", capture_datagrams(present_person()), monkeypatch)
    assert_matches_golden("presence", messages, root)


def test_web_camera_weapon_pins_output(tmp_path, monkeypatch):
    root = tmp_path / "root"
    calibrate_nodes(root)
    messages = run_camera_collection(root, "weapon", capture_datagrams(visible_knife()), monkeypatch)
    assert_matches_golden("weapon", messages, root)
