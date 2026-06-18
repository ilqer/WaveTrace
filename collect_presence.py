"""Step 2 of live bring-up: capture labeled empty/present sessions over UDP and train the presence
model. Run AFTER collect_baseline.py (needs data/cal). Produces data/model_presence — matched to the
live link, so run_live.py works end-to-end.

Each session = part A (zone EMPTY) then part B (you stand + MOVE in the zone). Frames in part B's
time span are labeled present(1), the rest absent(0)."""

import collections
import os
import time

from wavetrace.Source import UdpSource, RecordingSource, save_recording
from wavetrace.Cli import collect_source
from wavetrace.recognition import train_presence

UDP_PORT = 9876
CAL = "data/cal"
SUBJECT = "u0"
N_SESSIONS = 3
N = 1500   # ~15 s per state at 100 Hz


def capture_chunk(prompt, n, countdown=0):
    input(f"\n>> {prompt}\n   Press Enter to start...")
    if countdown:
        for d in range(countdown, 0, -1):
            print(f"   starting in {d}s...", end="\r")
            time.sleep(1)
        print()
    print("   [CAPTURING] hold the condition steady...")
    frames = list(UdpSource(UDP_PORT, timeout_s=15.0, max_frames=n).frames())
    if frames:  # pin to the modal width = the TX link (drop foreign/beacon frames)
        S = collections.Counter(f.num_subcarriers for f in frames).most_common(1)[0][0]
        frames = [f for f in frames if f.num_subcarriers == S]
    print(f"   got {len(frames)} frames")
    return frames


def main():
    os.makedirs("data", exist_ok=True)
    ds_dirs = []
    for i in range(N_SESSIONS):
        empty = capture_chunk(f"Session {i+1}/{N_SESSIONS} — part A: keep the zone EMPTY and still.",
                              N, countdown=5)
        present = capture_chunk(f"Session {i+1}/{N_SESSIONS} — part B: stand and MOVE in the zone.", N)
        if not empty or not present:
            print("[ERROR] missing frames — check the boards / run mesh_verify.py.")
            return
        span = (present[0].timestamp, present[-1].timestamp + 1.0)
        save_recording(empty + present, f"data/sess_{i}")
        collect_source(RecordingSource(f"data/sess_{i}"), CAL, f"data/ds_{i}", [span],
                       stage="presence", session_id=f"sess{i}", subject_id=SUBJECT)
        ds_dirs.append(f"data/ds_{i}")

    print("\nTraining presence model on", len(ds_dirs), "sessions...")
    _, m = train_presence(ds_dirs, out_dir="data/model_presence")
    print(f"\nsamples={m['n_samples']}  class_counts={m['class_counts']}")
    print(f"train_accuracy (optimistic) = {m['train_accuracy']:.3f}")
    logo = m.get("logo", {}).get("session")
    if logo:
        print(f"LEAVE-ONE-SESSION-OUT accuracy = {logo['accuracy']:.3f}  "
              f"(majority baseline {logo['majority_accuracy']:.3f}, "
              f"TPR {logo.get('tpr', 0):.3f}, FP {logo.get('fp_rate', 0):.3f})")
        print("  ^ the honest number: it must clearly beat the majority baseline to be real.")
    print("model saved -> data/model_presence")


if __name__ == "__main__":
    main()
