"""Capture labeled people-count sessions over UDP and train a count model per node.

Levels 0..N are captured in turn, with the top level N meaning "N or more people". Calibration is
shared with the presence path (<root>/cal/node{id}): gain and NBVI are per node and do not depend on
how many people are in the zone.
"""

import argparse
import sys

from wavetrace.pipeline import collect_mesh
from wavetrace.recognition.collect_modes import CountCollectMode, default_model_root


def main():
    parser = argparse.ArgumentParser(description="Capture people-count sessions and train a model per node.")
    parser.add_argument("--node", type=int, default=None, help="Train ONLY this node (default: all calibrated)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--sessions", type=int, default=3, help="Number of sessions to capture (default: 3)")
    parser.add_argument("--frames", type=int, default=1500, help="Frames per count per node (default: 1500)")
    parser.add_argument("--max-count", type=int, default=3, help="Top count level; captures 0..N, N='N+' (default: 3)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Count model root (default: <root>/model_count)")
    args = parser.parse_args()
    calibration_root = f"{args.root}/cal" if args.cal is None else args.cal
    model_root = default_model_root(args.root, "count") if args.model is None else args.model

    if args.max_count < 1:
        print("[ERROR] --max-count must be >= 1 (need at least empty vs one person).", file=sys.stderr)
        return
    collect_mesh(CountCollectMode(root=args.root, model_root=model_root, max_count=args.max_count),
                 port=args.port, calibration_root=calibration_root, node_id=args.node,
                 num_sessions=args.sessions, max_frames_per_link=args.frames)


if __name__ == "__main__":
    main()
