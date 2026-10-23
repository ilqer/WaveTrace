"""Characterization tests: pin the console output and the written output tree of scripts/collect_camera.py
end to end, for the presence and weapon stages and a presence run with --train.

Everything is offline and deterministic (tests/golden/camera_replay.py): the camera labels every frame
before the CSI capture starts, and the capture runs on a fake monotonic clock that each received
datagram and each receive timeout advances. Re-generate deliberately with
`WAVETRACE_UPDATE_GOLDEN=1 pytest tests/golden`.
"""

import builtins
import os
import sys
import warnings
from pathlib import Path

from camera_replay import DURATION_S, capture_datagrams, patch_camera, patch_mesh, present_person, visible_knife
from mesh_replay import calibrate_nodes, hash_tree
from scripts import collect_camera

EXPECTED = Path(__file__).parent / "expected"
SCRIPT_FRAME_RATE_HZ = 15.0


def run_script(root, datagrams, monkeypatch, capsys, extra_argv=()):
    patch_camera(monkeypatch, SCRIPT_FRAME_RATE_HZ)
    patch_mesh(monkeypatch, datagrams)
    monkeypatch.setattr(builtins, "input", lambda *prompt: "")
    argv = ["collect_camera", "--port", "5599", "--root", str(root), "--duration", f"{DURATION_S:g}", *extra_argv]
    monkeypatch.setattr(sys, "argv", argv)
    with warnings.catch_warnings():
        # torch warns on the heatmap's own loss bookkeeping; pytest's warning capture would hide it only sometimes
        warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")
        collect_camera.main()
    monkeypatch.undo()
    captured = capsys.readouterr()
    assert captured.out.count("\a") == 1, "the capture's end rings the bell exactly once"
    section = f"$ {' '.join(argv)}\n{captured.out}"
    if captured.err:
        section += f"[stderr]\n{captured.err}"
    return section.replace(str(root.parent), "<tmp>")


def assert_matches_golden(name, output, root):
    snapshot = f"{output}\n### tree\n{hash_tree(root)}"
    expected_path = EXPECTED / f"collect_camera_{name}.txt"
    if os.environ.get("WAVETRACE_UPDATE_GOLDEN"):
        expected_path.write_bytes(snapshot.encode())
    assert snapshot == expected_path.read_bytes().decode()


def test_collect_camera_presence_pins_output(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    calibrate_nodes(root)
    output = run_script(root, capture_datagrams(present_person()), monkeypatch, capsys)
    assert_matches_golden("presence", output, root)


def test_collect_camera_weapon_pins_output(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    calibrate_nodes(root)
    output = run_script(root, capture_datagrams(visible_knife()), monkeypatch, capsys,
                        extra_argv=("--stage", "weapon"))
    assert_matches_golden("weapon", output, root)


def test_collect_camera_presence_train_pins_output(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    calibrate_nodes(root)
    output = run_script(root, capture_datagrams(present_person()), monkeypatch, capsys, extra_argv=("--train",))
    assert_matches_golden("presence_train", output, root)


def test_collect_camera_weapon_train_pins_output(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    calibrate_nodes(root)
    output = run_script(root, capture_datagrams(visible_knife()), monkeypatch, capsys,
                        extra_argv=("--stage", "weapon", "--train"))
    assert_matches_golden("weapon_train", output, root)
