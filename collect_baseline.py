"""Step 1 of live bring-up: capture a quiet baseline over UDP and calibrate.

Prereqs: both ESP32s powered + flooding, Mac on RD-WIN1, `mesh_verify.py` shows ~100 Hz.
Produces data/cal — NBVI subcarriers selected from the REAL live frame width, so it matches the
hardware (the old data/cal was built on mismatched data → run_live IndexError)."""

import collections
import os
import sys
import time

from wavetrace.Source import UdpSource, RecordingSource, save_recording
from wavetrace.Cli import calibrate_source

UDP_PORT = 9876
N_FRAMES = 3000   # ~30 s at 100 Hz


def capture(n, timeout_s=15.0):
    """Collect up to n UDP frames, then keep only the dominant subcarrier width. The RX captures
    every frame on air (incl. beacons / foreign traffic of a different width); mixing widths breaks
    the fixed-shape recording and makes NBVI indices invalid. Pin to the modal width = the TX link."""
    frames = list(UdpSource(UDP_PORT, timeout_s=timeout_s, max_frames=n).frames())
    if frames:
        S = collections.Counter(f.num_subcarriers for f in frames).most_common(1)[0][0]
        frames = [f for f in frames if f.num_subcarriers == S]
    return frames


def main():
    os.makedirs("data", exist_ok=True)
    input("Both ESP32s powered, Mac on RD-WIN1, room QUIET. Press Enter to start the baseline...")
    for d in range(5, 0, -1):
        print(f"   capturing baseline in {d}s...", end="\r")
        time.sleep(1)
    print("\n   [CAPTURING] keep the room still and empty...")

    frames = capture(N_FRAMES)
    if not frames:
        print("\n[ERROR] no frames over UDP. Are both boards powered and flooding? "
              "Run `mesh_verify.py` to confirm reception first.", file=sys.stderr)
        return
    print(f"\n   got {len(frames)} frames, {frames[0].num_subcarriers} subcarriers each")

    save_recording(frames, "data/baseline_raw")
    calibrate_source(RecordingSource("data/baseline_raw"), "data/cal",
                     baseline_packets=min(2000, len(frames)))
    print("calibration written -> data/cal")


if __name__ == "__main__":
    main()
