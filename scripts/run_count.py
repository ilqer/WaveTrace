"""Live people-count fusion: every (tx->rx) link is served through its rx node's cal and count head.

Heads may have learned different class sets, so each one's probabilities are expanded into a global
class space before LinkVoter blends them, weighted by LOGO accuracy times live decision margin.
"""

import argparse

from wavetrace.application.serve_mesh import RateGate
from wavetrace.domain.contracts import DEFAULT_MIN_SAMPLE_RATE_FRACTION
from wavetrace.pipeline import serve_mesh
from wavetrace.recognition.serve_modes import CountServeMode


def main():
    parser = argparse.ArgumentParser(description="Live ALL-PAIRS people-count (per-link, per-RX-node cal+head).")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Count model root (default: <root>/model_count)")
    parser.add_argument("--max-count", type=int, default=3, help="Top count level for 'N+' formatting (default: 3)")
    parser.add_argument("--rate-gate", choices=[gate.value for gate in RateGate], default=RateGate.TARGET.value,
                        help="drop a link below the minimum fraction of its head's target rate (target) or of "
                             "the median delivered rate of the live links (peers) (default: target)")
    parser.add_argument("--min-rate-fraction", type=float, default=DEFAULT_MIN_SAMPLE_RATE_FRACTION,
                        help="minimum fraction of the gate's reference rate a link must deliver, in (0, 1] "
                             "(default: %(default)s)")
    args = parser.parse_args()
    calibration_root = f"{args.root}/cal" if args.cal is None else args.cal
    model_root = f"{args.root}/model_count" if args.model is None else args.model

    mode = CountServeMode.load(calibration_root, model_root, args.max_count)
    if not mode.nodes:
        print(f"[ERROR] no count models under {model_root}/node*/model.joblib with a matching "
              f"{calibration_root}/node*/. Run collect_baseline.py then collect_count.py first.")
        return
    serve_mesh(mode, args.port, rate_gate=RateGate(args.rate_gate),
               min_sample_rate_fraction=args.min_rate_fraction)


if __name__ == "__main__":
    main()
