"""Live weapon detection: every (tx->rx) link is served through its rx node's cal and weapon head
(inter-carrier features), and the links are fused into one armed/clear verdict.

LinkVoter weights each link by static reliability (LOGO accuracy) times live decision margin, so a
head validated at or below chance gets weight 0 and drops out.
"""

import argparse

from wavetrace.application.serve_mesh import RateGate
from wavetrace.domain.contracts import DEFAULT_MIN_SAMPLE_RATE_FRACTION
from wavetrace.pipeline import serve_mesh
from wavetrace.recognition.serve_modes import WeaponServeMode


def main():
    parser = argparse.ArgumentParser(description="Live ALL-PAIRS weapon detection (per-link, per-RX-node cal+head).")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Weapon model root (default: <root>/model_weapon)")
    parser.add_argument("--rate-gate", choices=[gate.value for gate in RateGate], default=RateGate.TARGET.value,
                        help="drop a link below the minimum fraction of its head's target rate (target) or of "
                             "the median delivered rate of the live links (peers) (default: target)")
    parser.add_argument("--min-rate-fraction", type=float, default=DEFAULT_MIN_SAMPLE_RATE_FRACTION,
                        help="minimum fraction of the gate's reference rate a link must deliver, in (0, 1] "
                             "(default: %(default)s)")
    args = parser.parse_args()
    calibration_root = f"{args.root}/cal" if args.cal is None else args.cal
    model_root = f"{args.root}/model_weapon" if args.model is None else args.model

    mode = WeaponServeMode.load(calibration_root, model_root)
    if not mode.nodes:
        print(f"[ERROR] no weapon heads under {model_root}/node*/[link*/]model.joblib with a matching "
              f"{calibration_root}/node*/. Run collect_baseline.py then collect_weapon.py first.")
        return
    serve_mesh(mode, args.port, rate_gate=RateGate(args.rate_gate),
               min_sample_rate_fraction=args.min_rate_fraction)


if __name__ == "__main__":
    main()
