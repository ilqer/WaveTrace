"""Live presence over every (tx->rx) link, each served through its rx node's own cal and head.

Gain is a property of the receiver (gain=LOCK on one board, gain=SKIP on another), so calibration is
per rx node, while the features stay per link to keep all N(N-1) views. LinkVoter blends every live
link weighted by its head's decision margin, so a blocked link down-weights itself and a node that
drops just removes its links from the vote.
"""

import argparse

from wavetrace.pipeline import serve_mesh
from wavetrace.recognition.serve_modes import PresenceServeMode


def main():
    parser = argparse.ArgumentParser(description="Live ALL-PAIRS presence (per-link, per-RX-node cal+head).")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Per-node model root (default: <root>/model)")
    args = parser.parse_args()
    calibration_root = f"{args.root}/cal" if args.cal is None else args.cal
    model_root = f"{args.root}/model" if args.model is None else args.model

    mode = PresenceServeMode.load(calibration_root, model_root)
    if not mode.nodes:
        print(f"[ERROR] no per-node models found under {model_root}/node*/model.joblib with a matching "
              f"{calibration_root}/node*/. Run collect_baseline.py then collect_presence.py first.")
        return
    serve_mesh(mode, args.port)


if __name__ == "__main__":
    main()
