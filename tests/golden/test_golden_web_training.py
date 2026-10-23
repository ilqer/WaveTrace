"""Characterization tests: pin what the web dashboard's `WaveTraceRunner.run_training_request` logs and
publishes, and every file it writes, per stage and dataset choice.

One capture-profile root is built once in the collect modes' own layout (presence ds_*, count count_ds_*,
the weapon_ds pool, and a camera session from the dashboard's camera collection); every case trains on its
own copy. Offline and deterministic like test_golden_web_camera.py. Logs drop their time of day and any
traceback frames (the `ERROR ...` line stays); payloads are printed with sorted keys and without
`fit_seconds`; the tree lists only the files training added or changed. Re-generate deliberately with
`WAVETRACE_UPDATE_GOLDEN=1 pytest tests/golden`.
"""

import asyncio
import json
import os
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from camera_replay import DURATION_S, capture_datagrams, patch_camera, patch_mesh, present_person
from mesh_replay import (
    RX_NODE_IDS, TX_MACS, FrameListSource, calibrate_nodes, generate_link_frames, hash_tree, shift_timestamps,
)
from wavetrace.pipeline import collect_source, write_link_dataset
from wavetrace.recognition.collect_modes import CountCollectMode, PresenceCollectMode, WeaponCollectMode
from web.streamer import WaveTraceRunner

EXPECTED = Path(__file__).parent / "expected"
QUEUE_NAMES = ("inference", "stream", "logs", "training", "telemetry")
CAPTURE_S = 3.0
LATER_CAPTURE_OFFSET_S = 100.0
CAMERA_SESSION_ID = "cam_cam_s0"


class _RecordingQueue:
    def __init__(self, messages, queue_name):
        self._messages = messages
        self._queue_name = queue_name

    def put(self, message):
        self._messages.append((self._queue_name, message))


def _link_keys():
    return [(tx_mac[-5:], rx_node_id) for rx_node_id in RX_NODE_IDS for tx_mac in TX_MACS]


def _link_frames(link_key, capture_index, **condition_kwargs):
    tx_short, rx_node_id = link_key
    tx_index = [tx_mac[-5:] for tx_mac in TX_MACS].index(tx_short)
    return generate_link_frames(rx_node_id, tx_index, CAPTURE_S, seed_offset=100 * capture_index, **condition_kwargs)


def _write(root, request):
    write_link_dataset(request, root / "cal" / f"node{request.rx_node_id}")


def _write_presence_pool(root):
    mode = PresenceCollectMode(root=str(root), model_root=str(root / "model"))
    for session_index in range(2):
        [[conditions]] = mode.plan_session(session_index, 2)
        for link_key in _link_keys():
            empty = _link_frames(link_key, 2 * session_index)
            present = shift_timestamps(
                _link_frames(link_key, 2 * session_index + 1, presence_spans=[(0.0, CAPTURE_S)],
                             presence_turbulence_std=0.15), LATER_CAPTURE_OFFSET_S)
            _write(root, mode.describe_link_dataset(session_index, conditions, link_key, [empty, present]))


def _write_count_pool(root):
    turbulence_std_by_count = {0: 0.0, 1: 0.1, 2: 0.3}
    mode = CountCollectMode(root=str(root), model_root=str(root / "model_count"), max_count=2)
    for session_index in range(2):
        for count, [conditions] in enumerate(mode.plan_session(session_index, 2)):
            for link_key in _link_keys():
                frames = _link_frames(link_key, 10 + 3 * session_index + count,
                                      presence_spans=[(0.0, CAPTURE_S)] if count else [],
                                      presence_turbulence_std=turbulence_std_by_count[count])
                _write(root, mode.describe_link_dataset(session_index, conditions, link_key, [frames]))


def _write_weapon_pool(root):
    for subject_index, (subject_id, carry) in enumerate((("p0", "waist"), ("p1", "chest"))):
        mode = WeaponCollectMode(root=str(root), model_root=str(root / "model_weapon"), subject_id=subject_id,
                                 carry=carry, should_subtract_ic_baseline=True, is_per_link=False)
        [group] = mode.plan_session(0, 1)
        for is_armed, conditions in enumerate(group):
            condition_kwargs = {"weapon_spans": [(0.0, CAPTURE_S)], "weapon_signature_depth": 0.4} if is_armed else {}
            for link_key in _link_keys():
                frames = _link_frames(link_key, 20 + 2 * subject_index + is_armed, **condition_kwargs)
                _write(root, mode.describe_link_dataset(0, conditions, link_key, [frames]))


def _write_one_weapon_dataset(base):
    """Both classes in one dataset, as a single labelled recording makes."""
    frames = generate_link_frames(1, 0, 2 * CAPTURE_S, seed_offset=40, weapon_spans=[(2.0, 4.0)],
                                  weapon_signature_depth=0.4)
    collect_source(FrameListSource(frames), base / "root" / "cal" / "node1", base / "one_weapon_dataset",
                   [(2.0, 4.0)], stage="weapon", session_id="p0_waist_s0", subject_id="p0")


def _run(runner_method, request, monkeypatch):
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coroutine, loop: None)
    messages = []
    runner = WaveTraceRunner(None, *[_RecordingQueue(messages, name) for name in QUEUE_NAMES])
    with warnings.catch_warnings():
        # torch warns on the heatmap's own loss bookkeeping; pytest's warning capture would hide it only sometimes
        warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")
        getattr(runner, runner_method)(request)
    monkeypatch.undo()
    assert not runner.is_running
    return messages


def _collect_camera_session(root):
    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_camera(monkeypatch, 5.0)
        patch_mesh(monkeypatch, capture_datagrams(present_person()))
        request = SimpleNamespace(calibration=str(root / "cal"), train_data=str(root), duration=DURATION_S,
                                  cam_index=0, per_link=False, col_stage="presence", yolo_weights="yolov8n-seg.pt",
                                  udp_port=5599)
        _run("run_camera_collection_request", request, monkeypatch)
    assert (root / "cam_ds" / "heatmap" / CAMERA_SESSION_ID / "features.npy").exists()


@pytest.fixture(scope="module")
def built_base(tmp_path_factory):
    base = tmp_path_factory.mktemp("web_training") / "base"
    root = base / "root"
    calibrate_nodes(root)
    _write_presence_pool(root)
    _write_count_pool(root)
    _write_weapon_pool(root)
    _write_one_weapon_dataset(base)
    _collect_camera_session(root)
    return base


def _strip_fit_seconds(payload):
    if isinstance(payload, dict):
        return {key: _strip_fit_seconds(value) for key, value in payload.items() if key != "fit_seconds"}
    return payload


def format_messages(messages, case_dir):
    lines, key_paths = [], set()
    is_in_traceback = False
    for queue_name, message in messages:
        if queue_name == "logs":
            text = message.split("] ", 1)[1]
            if is_in_traceback and text.startswith("  "):
                continue
            is_in_traceback = text.startswith("ERROR")
            lines.append(f"log {text}")
        else:
            payload = _strip_fit_seconds(json.loads(message))
            lines.append(f"{queue_name} {json.dumps(payload, sort_keys=True)}")
            key_paths |= {f"{queue_name}: {key}" for key in payload}
    text = "\n".join(lines) + "\n### keys\n" + "\n".join(sorted(key_paths)) + "\n"
    return text.replace(str(case_dir), "<case>")


def _changed_files(before, after):
    old_lines = set(before.splitlines())
    return "".join(f"{line}\n" for line in after.splitlines() if line not in old_lines)


CASES = {
    "presence_one_dataset": dict(col_stage="presence", train_backend="mlp",
                                 train_data="root/ds_0/node1/link_4f9c"),
    "presence_root": dict(col_stage="presence", train_backend="svm", train_data="root"),
    "camera_presence_root": dict(col_stage="presence", train_backend="mlp", train_data="root", camera_collect=True),
    "presence_cnn_root": dict(col_stage="presence", train_backend="cnn", train_data="root"),
    "count_one_dataset": dict(col_stage="count", train_backend="mlp", train_data="root/count_ds_0/c1/node1/link_4f9c"),
    "count_root": dict(col_stage="count", train_backend="mlp", train_data="root"),
    "weapon_variance_one_dataset": dict(col_stage="weapon", train_backend="variance",
                                        train_data="one_weapon_dataset"),
    "weapon_pool_parent": dict(col_stage="weapon", train_backend="variance", train_data="root/weapon_ds/node1"),
    "weapon_root": dict(col_stage="weapon", train_backend="svm", train_data="root"),
    "per_link_root": dict(col_stage="weapon", train_backend="variance", train_data="root", per_link=True),
    "heatmap_presence": dict(col_stage="presence", train_backend="heatmap",
                             train_data=f"root/cam_ds/heatmap/{CAMERA_SESSION_ID}"),
    "heatmap_weapon": dict(col_stage="weapon", train_backend="heatmap",
                           train_data=f"root/cam_ds/heatmap/{CAMERA_SESSION_ID}"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_web_training_pins_output(name, built_base, tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    shutil.copytree(built_base, case_dir)
    tree_before = hash_tree(case_dir)
    options = {"per_link": False, "camera_collect": False, **CASES[name]}
    request = SimpleNamespace(**{**options, "train_data": str(case_dir / options["train_data"]),
                                 "train_out": str(case_dir / "out")})
    messages = _run("run_training_request", request, monkeypatch)
    assert not any("\a" in message for _, message in messages), "the dashboard log rings no bell"
    snapshot = f"{format_messages(messages, case_dir)}\n### tree\n{_changed_files(tree_before, hash_tree(case_dir))}"
    expected_path = EXPECTED / f"web_training_{name}.txt"
    if os.environ.get("WAVETRACE_UPDATE_GOLDEN"):
        expected_path.write_bytes(snapshot.encode())
    assert snapshot == expected_path.read_bytes().decode()
