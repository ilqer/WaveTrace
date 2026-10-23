"""Characterization tests: pin the exact console output and the written output tree of the three
collect scripts (collect_presence, collect_count, collect_weapon) end to end, so any change to their
output is caught.

Everything is offline and deterministic. Every capture binds its own UDP socket, so the patched
`socket.socket` hands each bind the next capture's queued datagrams; an exhausted queue is the
total-silence timeout that ends a capture. `input`, `time.sleep` and `time.time` are patched too, so
prompts, countdowns and the progress cadence replay identically. A tree manifest hashes every file
under the root; metrics.json is hashed without `fit_seconds`, its only wall-clock field.

A failure means behaviour changed, or a snapshot in tests/golden/expected/collect_*.txt needs a
deliberate, reviewed re-generation: `WAVETRACE_UPDATE_GOLDEN=1 pytest tests/golden`.
"""

import builtins
import collections
import itertools
import os
import socket
import sys
import time
from pathlib import Path

from mesh_replay import (
    CLOCK_START_S, RX_NODE_IDS, TX_MACS, FakeClock, calibrate_nodes, encode_datagrams, generate_link_frames,
    hash_tree, shift_timestamps,
)
from scripts import collect_count, collect_presence, collect_weapon

EXPECTED = Path(__file__).parent / "expected"

FRAMES_PER_LINK = 256  # two 128-frame windows at 100 Hz, with hop 32 five per class per link
FULL_CAPTURE_S = 3.0
SHORT_CAPTURE_S = 1.0  # 100 frames: below one window, so the link is dropped
CAPTURE_SPACING_S = 100.0
RECEIVE_STEP_S = 0.025
COUNT_TURBULENCE_STD_DEVIATION = {0: 0.0, 1: 0.1, 2: 0.3}


class FakeUdpSocket:
    """Replays one capture's datagrams, then times out as a silent mesh would. Each recv advances the
    clock."""

    def __init__(self, clock, datagrams):
        self._clock = clock
        self._pending = collections.deque(datagrams)

    def setsockopt(self, *args):
        pass

    def bind(self, address):
        pass

    def settimeout(self, timeout_s):
        pass

    def recvfrom(self, buffer_size):
        if not self._pending:
            raise socket.timeout
        self._clock.now_s += RECEIVE_STEP_S
        return self._pending.popleft(), ("127.0.0.1", 0)

    def close(self):
        pass


def capture_datagrams(capture_index, condition_kwargs, short_rx_node_ids=()):
    """One capture: every (tx -> rx) link of both rx nodes, the rx nodes interleaved datagram by
    datagram. Links of `short_rx_node_ids` carry too few frames for one window."""
    per_node = []
    for rx_node_id in RX_NODE_IDS:
        duration_s = SHORT_CAPTURE_S if rx_node_id in short_rx_node_ids else FULL_CAPTURE_S
        frames_by_tx_index = {
            tx_index: shift_timestamps(
                generate_link_frames(rx_node_id, tx_index, duration_s, seed_offset=100 * capture_index,
                                     **condition_kwargs),
                capture_index * CAPTURE_SPACING_S)
            for tx_index in range(len(TX_MACS))}
        per_node.append(encode_datagrams(rx_node_id, frames_by_tx_index))
    return [datagram for step in itertools.zip_longest(*per_node) for datagram in step
            if datagram is not None]


def empty_room():
    return {}


def present_person():
    return {"presence_spans": [(0.0, FULL_CAPTURE_S)], "presence_turbulence_std": 0.15}


def count_people(count):
    return {"presence_spans": [(0.0, FULL_CAPTURE_S)] if count else [],
            "presence_turbulence_std": COUNT_TURBULENCE_STD_DEVIATION[count]}


def concealed_weapon():
    return {"weapon_spans": [(0.0, FULL_CAPTURE_S)], "weapon_signature_depth": 0.4}


def run_script(script, root, captures, monkeypatch, capsys, extra_argv=()):
    """Runs `script.main()` once; returns '$ argv', its stdout, then its stderr if any, with the
    parent of `root` printed as <tmp>."""
    clock = FakeClock(CLOCK_START_S)
    pending_captures = collections.deque(captures)
    monkeypatch.setattr(time, "time", clock)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(builtins, "input", lambda *prompt: "")
    monkeypatch.setattr(socket, "socket",
                        lambda *args, **kwargs: FakeUdpSocket(clock, pending_captures.popleft()))
    argv = [Path(script.__file__).stem, "--port", "5599", "--root", str(root), "--frames",
            str(FRAMES_PER_LINK), *extra_argv]
    monkeypatch.setattr(sys, "argv", argv)
    script.main()
    monkeypatch.undo()
    assert not pending_captures, "the script captured fewer conditions than the test queued"
    captured = capsys.readouterr()
    section = f"$ {' '.join(argv)}\n{captured.out}"
    if captured.err:
        section += f"[stderr]\n{captured.err}"
    return section.replace(str(root.parent), "<tmp>")


def assert_matches_golden(name, output, roots):
    snapshot = output + "".join(f"\n### tree {label}\n{hash_tree(root)}" for label, root in roots)
    expected_path = EXPECTED / f"collect_{name}.txt"
    if os.environ.get("WAVETRACE_UPDATE_GOLDEN"):
        expected_path.write_bytes(snapshot.encode())
    assert snapshot == expected_path.read_bytes().decode()


def presence_captures(num_sessions, first_capture_index=0, short_session=None):
    captures = []
    for i in range(num_sessions):
        short_rx_node_ids = (2,) if i == short_session else ()
        capture_index = first_capture_index + 2 * i
        captures.append(capture_datagrams(capture_index, empty_room()))
        captures.append(capture_datagrams(capture_index + 1, present_person(), short_rx_node_ids))
    return captures


def test_collect_presence_pins_output(tmp_path, monkeypatch, capsys):
    root, node_root, missing_root = tmp_path / "all", tmp_path / "node", tmp_path / "missing"
    for calibrated_root in (root, node_root):
        calibrate_nodes(calibrated_root)
    output = run_script(collect_presence, root, presence_captures(2, short_session=1), monkeypatch, capsys,
                        extra_argv=("--sessions", "2"))
    assert "[SKIP] node 2 session 1" in output
    output += run_script(collect_presence, node_root, presence_captures(1, first_capture_index=10),
                         monkeypatch, capsys, extra_argv=("--sessions", "1", "--node", "2"))
    output += run_script(collect_presence, node_root, [], monkeypatch, capsys, extra_argv=("--node", "9"))
    output += run_script(collect_presence, missing_root, [], monkeypatch, capsys)
    assert_matches_golden("presence", output, [("all", root), ("node", node_root)])


def test_collect_count_pins_output(tmp_path, monkeypatch, capsys):
    root, missing_root = tmp_path / "all", tmp_path / "missing"
    calibrate_nodes(root)
    captures = [capture_datagrams(3 * i + count, count_people(count), (2,) if i == 1 else ())
                for i in range(2) for count in range(3)]
    output = run_script(collect_count, root, captures, monkeypatch, capsys,
                        extra_argv=("--sessions", "2", "--max-count", "2"))
    assert "per-class:" in output
    output += run_script(collect_count, root, [], monkeypatch, capsys, extra_argv=("--max-count", "0"))
    output += run_script(collect_count, root, [], monkeypatch, capsys, extra_argv=("--node", "9"))
    output += run_script(collect_count, missing_root, [], monkeypatch, capsys)
    assert_matches_golden("count", output, [("all", root)])


def test_collect_weapon_pins_output_over_the_cumulative_pool(tmp_path, monkeypatch, capsys):
    root, missing_root = tmp_path / "all", tmp_path / "missing"
    calibrate_nodes(root)
    first_run = [capture_datagrams(2 * i + armed, concealed_weapon() if armed else empty_room(),
                                   (2,) if i == 1 else ())
                 for i in range(2) for armed in (0, 1)]
    output = run_script(collect_weapon, root, first_run, monkeypatch, capsys,
                        extra_argv=("--sessions", "2", "--subject", "p0", "--carry", "waist"))
    second_run = [capture_datagrams(10 + armed, concealed_weapon() if armed else empty_room())
                  for armed in (0, 1)]
    output += run_script(collect_weapon, root, second_run, monkeypatch, capsys,
                         extra_argv=("--sessions", "1", "--subject", "p1", "--carry", "chest",
                                     "--per-link", "--no-bg-subtract"))
    assert "carry-LOGO" in output
    output += run_script(collect_weapon, root, [], monkeypatch, capsys, extra_argv=("--node", "9"))
    output += run_script(collect_weapon, missing_root, [], monkeypatch, capsys)
    assert_matches_golden("weapon", output, [("all", root)])
