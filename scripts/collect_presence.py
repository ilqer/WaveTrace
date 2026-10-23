"""Capture labeled empty/present sessions over UDP and train a presence model per mesh node.

Each node is trained on its own, so live, a node that drops only lowers the vote weight instead of
taking the system down. Each session is part A (zone empty) then part B (stand and move in
the zone).
"""

import argparse

from wavetrace.pipeline import collect_mesh
from wavetrace.recognition.collect_modes import PresenceCollectMode, default_model_root


def main():
    parser = argparse.ArgumentParser(description="Capture presence sessions and train a model per node.")
    parser.add_argument("--node", type=int, default=None, help="Train ONLY this node (default: all calibrated)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--sessions", type=int, default=3, help="Number of sessions to capture (default: 3)")
    parser.add_argument("--frames", type=int, default=1500, help="Frames per condition per node (default: 1500)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Per-node model root (default: <root>/model)")
    args = parser.parse_args()
    calibration_root = f"{args.root}/cal" if args.cal is None else args.cal
    model_root = default_model_root(args.root, "presence") if args.model is None else args.model

    collect_mesh(PresenceCollectMode(root=args.root, model_root=model_root), port=args.port,
                 calibration_root=calibration_root, node_id=args.node, num_sessions=args.sessions,
                 max_frames_per_link=args.frames)


if __name__ == "__main__":
    main()
