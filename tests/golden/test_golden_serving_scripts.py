"""Characterization tests: pin the exact stdout of the three live serving scripts (run_live_mesh,
run_count, run_weapon) end to end, so a refactor of their shared loop can prove it changed nothing.

Everything is offline and deterministic: seeded synthetic links are trained into tiny heads, encoded
into the real UDP batch wire format and replayed through a fake socket under a fake clock.
Patched seams: the UDP adapter's `bind_udp_socket` name and the global `time.time`.

A failure means behaviour changed, or the snapshot in tests/golden/expected/serve_*.txt needs a
deliberate, reviewed re-generation: `WAVETRACE_UPDATE_GOLDEN=1 pytest tests/golden`.
"""

import os
import sys
import time
from pathlib import Path

from mesh_replay import (
    CLOCK_START_S, FakeClock, FakeUdpSocket, calibrate_nodes, count_live_frames, live_datagrams,
    presence_live_frames, train_count_nodes, train_presence_nodes, train_weapon_heads, weapon_live_frames,
)
from scripts import run_count, run_live_mesh, run_weapon
from wavetrace.adapters.source import udp

EXPECTED = Path(__file__).parent / "expected"


def run_script(script, root, datagrams, monkeypatch, capsys, extra_argv=()):
    clock = FakeClock(CLOCK_START_S)
    fake_socket = FakeUdpSocket(clock, datagrams)
    monkeypatch.setattr(time, "time", clock)
    monkeypatch.setattr(udp, "bind_udp_socket", lambda port, *, timeout_s=None: fake_socket)
    monkeypatch.setattr(sys, "argv", [Path(script.__file__).stem, "--port", "5599", "--root", str(root),
                                      *extra_argv])
    script.main()
    return capsys.readouterr().out


def assert_matches_golden(name, output):
    expected_path = EXPECTED / f"serve_{name}.txt"
    if os.environ.get("WAVETRACE_UPDATE_GOLDEN"):
        expected_path.write_bytes(output.encode())
    assert output == expected_path.read_bytes().decode()


def verdict_lines(output):
    return [line for line in output.replace("\r", "\n").split("\n")
            if line.startswith(("PRESENT", "absent", "PEOPLE", "WEAPON", "clear"))]


def test_run_live_mesh_pins_presence_verdicts(tmp_path, monkeypatch, capsys):
    calibrate_nodes(tmp_path)
    train_presence_nodes(tmp_path)
    output = run_script(run_live_mesh, tmp_path, live_datagrams(presence_live_frames),
                        monkeypatch, capsys)
    labels = {line.split()[0] for line in verdict_lines(output)}
    assert labels == {"PRESENT", "absent"}
    assert_matches_golden("presence", output)


def test_run_count_pins_count_verdicts(tmp_path, monkeypatch, capsys):
    calibrate_nodes(tmp_path)
    train_count_nodes(tmp_path)
    output = run_script(run_count, tmp_path, live_datagrams(count_live_frames), monkeypatch, capsys,
                        extra_argv=("--max-count", "2"))
    assert len({line.split("(")[1].split(")")[0] for line in verdict_lines(output)}) >= 2
    assert_matches_golden("count", output)


def test_run_weapon_pins_weapon_verdicts(tmp_path, monkeypatch, capsys):
    calibrate_nodes(tmp_path)
    train_weapon_heads(tmp_path)
    output = run_script(run_weapon, tmp_path, live_datagrams(weapon_live_frames), monkeypatch, capsys)
    assert {line.split()[0] for line in verdict_lines(output)} == {"WEAPON", "clear"}
    assert_matches_golden("weapon", output)
