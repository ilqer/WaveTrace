"""Capture labeled no-weapon/weapon sessions over UDP and train a weapon model per node.

The head uses the inter-carrier feature block (stage="weapon", feature_mode="ic27") rather than the
amplitude features presence uses: a concealed object shows up in inter-subcarrier structure, not in
body motion, and the subject stands still throughout. Every run appends its datasets to the
cumulative pool at <root>/weapon_ds and retrains each node on the whole pool, so subject and
carry-position diversity builds up over many runs.
"""

import argparse

from wavetrace.pipeline import collect_mesh
from wavetrace.recognition.collect_modes import WeaponCollectMode, default_model_root


def main():
    parser = argparse.ArgumentParser(description="Capture no-weapon/weapon sessions and train a weapon model per node.")
    parser.add_argument("--node", type=int, default=None, help="Train ONLY this node (default: all calibrated)")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--sessions", type=int, default=3, help="Sessions to capture THIS run (default: 3)")
    parser.add_argument("--frames", type=int, default=1500, help="Frames per condition per node (default: 1500)")
    parser.add_argument("--subject", default="p0", help="Subject id for this run (vary it across people!)")
    parser.add_argument("--carry", default="na", help="Carry position label, e.g. waist/chest/ankle (default: na)")
    parser.add_argument("--bg-subtract", action=argparse.BooleanOptionalAction, dest="bg_subtract",
                        default=True,
                        help="Subtract the quiet-room baseline from σ²[p]; serving mirrors it. "
                             "Default ON (matches the web frontend); pass --no-bg-subtract to disable.")
    parser.add_argument("--per-link", action="store_true", dest="per_link",
                        help="Train ONE head per (tx->rx) DIRECTION -> model_weapon/node<id>/link<tag>/ "
                             "instead of pooling a node's directions into one head. "
                             "The signal is per-direction; pooling sign-flips the good NLOS link. "
                             "Bad directions are NOT dropped — run_weapon's LinkVoter zeroes a "
                             "sub-chance link via its own LOGO weight (accuracy_weights).")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Weapon model root (default: <root>/model_weapon)")
    args = parser.parse_args()
    calibration_root = f"{args.root}/cal" if args.cal is None else args.cal
    model_root = default_model_root(args.root, "weapon") if args.model is None else args.model

    mode = WeaponCollectMode(root=args.root, model_root=model_root, subject_id=args.subject, carry=args.carry,
                             should_subtract_ic_baseline=args.bg_subtract, is_per_link=args.per_link)
    collect_mesh(mode, port=args.port, calibration_root=calibration_root, node_id=args.node,
                 num_sessions=args.sessions, max_frames_per_link=args.frames)


if __name__ == "__main__":
    main()
