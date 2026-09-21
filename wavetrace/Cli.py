"""The command-line entry point: argparse modes wired to the `wavetrace.application` use-cases.

Six modes: capture / calibrate / collect-data / train / localize / run. `run` is the real-time path
(front-end -> head -> publish) and reuses `Frontend.iterWindows` so the served features match
training exactly.

CSI source today = synthetic (wavetrace.Synthetic) or a saved recording. All non-`run` modes are
offline.
"""

import argparse
import sys

import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.Source import buildCsiSource, parseTimeSpans, saveRecording
from wavetrace.application.calibrate import calibrate_source
from wavetrace.application.collect import collect_source
from wavetrace.application.localize import localize_source
from wavetrace.application.serve import run_inference
from wavetrace.output import JsonlPublisher
from wavetrace.recognition import trainPresence, trainWeapon

# tests/golden/TestGoldenPipeline.py imports these three names directly.
calibrateSource = calibrate_source
collectSource = collect_source
runInference = run_inference


# ----- argparse layer -----------------------------------------------------------------------------

def _sourceFromArgs(args):
    """Alias of `wavetrace.Source.buildCsiSource`. `tests/TestRegression.py` imports it by this
    name."""
    return buildCsiSource(args)


def _add_source_args(parser):
    parser.add_argument("--recording", help="replay a saved recording directory")
    parser.add_argument("--synthetic", action="store_true", help="generate frames in-process (no hardware)")
    parser.add_argument("--antennas", type=int, default=2)
    parser.add_argument("--subcarriers", type=int, default=32)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--presence", default="", help="presence spans 'a:b,c:d' (synthetic)")
    parser.add_argument("--weapon", default="", help="weapon spans 'a:b,c:d' (synthetic)")
    parser.add_argument("--weapon-depth", type=float, default=0.0, dest="weapon_depth")
    parser.add_argument("--seed", type=int, default=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wavetrace", description="WiFi-CSI sensing pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    capture_parser = subparsers.add_parser("capture", help="record CSI frames to disk")
    _add_source_args(capture_parser)
    capture_parser.add_argument("--out", required=True)

    calibrate_parser = subparsers.add_parser("calibrate", help="quiet-baseline calibration -> calibration dir")
    _add_source_args(calibrate_parser)
    calibrate_parser.add_argument("--out", required=True)
    calibrate_parser.add_argument("--baseline-packets", type=int, default=300, dest="baseline_packets")
    calibrate_parser.add_argument("--no-gain-lock", action="store_true", dest="no_gain_lock")

    collect_parser = subparsers.add_parser("collect-data", help="frames + scripted labels -> dataset dir")
    _add_source_args(collect_parser)
    collect_parser.add_argument("--calibration", required=True)
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--stage", choices=["presence", "weapon"], default="presence")
    collect_parser.add_argument("--label-spans", default="", dest="label_spans",
                                 help="present/weapon spans 'a:b,c:d'")
    collect_parser.add_argument("--window", type=int, default=128)
    collect_parser.add_argument("--hop", type=int, default=32)
    collect_parser.add_argument("--session-id", default="", dest="session_id")
    collect_parser.add_argument("--subject-id", default="", dest="subject_id")
    collect_parser.add_argument("--tier", choices=["open", "wrapped", "concealed"], default="",
                                 help="weapon ground-truth tier -> meta['tier'] (concealed = held-out test split)")
    collect_parser.add_argument("--frame-average", type=int, default=1, dest="frame_average")
    collect_parser.add_argument("--subtract-baseline", action="store_true", dest="subtract_baseline")

    train_parser = subparsers.add_parser("train", help="dataset(s) -> model")
    train_parser.add_argument("datasets", nargs="+")
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--stage", choices=["presence", "weapon"], default="presence")
    train_parser.add_argument("--backend", default=None, help="mlp|svm|variance|cnn (default per stage)")
    train_parser.add_argument("--feature-mode", default="ic27", dest="feature_mode",
                               choices=["ic27", "fusion", "cnn"], help="weapon stage only")

    localize_parser = subparsers.add_parser("localize", help="AoA spatial heatmap (where) -> track + heatmap dir")
    _add_source_args(localize_parser)
    localize_parser.add_argument("--out", required=True)
    localize_parser.add_argument("--spacing", type=float, default=0.5,
                                  help="ULA element spacing in wavelengths (default 0.5 = lambda/2)")
    localize_parser.add_argument("--method", choices=["music", "bartlett"], default="music")
    localize_parser.add_argument("--num-sources", type=int, default=1, dest="num_sources")
    localize_parser.add_argument("--num-angles", type=int, default=181, dest="num_angles")
    localize_parser.add_argument("--subcarrier-hz", type=float, default=312.5e3, dest="subcarrier_hz",
                                  help="subcarrier spacing for the range axis (HT20/64 = 312.5 kHz)")
    localize_parser.add_argument("--max-range-m", type=float, default=12.0, dest="max_range_m")
    localize_parser.add_argument("--num-ranges", type=int, default=64, dest="num_ranges",
                                  help="range grid resolution of the joint 2-D room map")
    localize_parser.add_argument("--no-range", action="store_true", dest="no_range",
                                  help="azimuth only (skip the joint 2-D range axis)")
    localize_parser.add_argument("--track", default=None,
                                  help="JSONL file for the per-frame localization track (default <out>/track.jsonl)")
    localize_parser.add_argument("--no-filter", action="store_true", dest="no_filter",
                                  help="publish raw per-frame fixes (skip the constant-velocity Kalman tracker)")

    run_parser = subparsers.add_parser("run", help="stream inference -> publish verdicts")
    _add_source_args(run_parser)
    run_parser.add_argument("--calibration", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--head-mode", choices=["presence", "weapon"], default="presence",
                             dest="head_mode", help="which operating mode to serve")
    run_parser.add_argument("--out", default=None, help="JSONL output file (default stdout)")
    run_parser.add_argument("--vote", action="store_true", help="also emit a final soft-vote verdict")
    run_parser.add_argument("--guard", action="store_true", help="enable AlertGuard debounce + DriftMonitor advisory")

    args = parser.parse_args(argv)

    if args.mode == "capture":
        saveRecording(list(_sourceFromArgs(args).frames()), args.out)
        print(f"captured -> {args.out}", file=sys.stderr)
    elif args.mode == "calibrate":
        path, _ = calibrate_source(_sourceFromArgs(args), args.out,
                                    baseline_packets=args.baseline_packets,
                                    use_gain_lock=not args.no_gain_lock)
        print(f"calibration -> {path}", file=sys.stderr)
    elif args.mode == "collect-data":
        path, dataset = collect_source(_sourceFromArgs(args), args.calibration, args.out,
                                        parseTimeSpans(args.label_spans), stage=args.stage,
                                        window=args.window, hop=args.hop,
                                        session_id=args.session_id, subject_id=args.subject_id,
                                        frame_average=args.frame_average,
                                        subtract_baseline=args.subtract_baseline, tier=args.tier)
        print(f"dataset ({dataset.y.size} samples) -> {path}", file=sys.stderr)
    elif args.mode == "train":
        if args.stage == "presence":
            _, metrics = trainPresence(args.datasets, out_dir=args.out)  # k taken from dataset meta
        else:
            config = None
            if args.backend:
                # config only needed to override the backend; k still comes from dataset meta
                from wavetrace.groundtruth import loadDataset
                nbvi_subcarrier_count = int(loadDataset(args.datasets[0]).meta["K"])
                config = ModelConfig(stage="weapon", k=nbvi_subcarrier_count, backend=args.backend)
            _, metrics = trainWeapon(args.datasets, out_dir=args.out, config=config,
                                      feature_mode=args.feature_mode)
        print(f"model -> {args.out} ({metrics})", file=sys.stderr)
    elif args.mode == "localize":
        from pathlib import Path
        track = args.track or str(Path(args.out) / "track.jsonl")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with JsonlPublisher(track, mode="localize") as publisher:
            path, agg = localize_source(
                _sourceFromArgs(args), args.out, num_antennas=args.antennas,
                antenna_spacing_wavelengths=args.spacing,
                method=args.method, num_sources=args.num_sources, num_angles=args.num_angles,
                subcarrier_spacing_hz=args.subcarrier_hz, max_range_m=args.max_range_m,
                num_ranges=args.num_ranges, range_enabled=not args.no_range,
                filter_track=not args.no_filter, publisher=publisher,
            )
        rng = "n/a" if np.isnan(agg.peak_range_m) else f"{agg.peak_range_m:.2f} m"
        print(f"localization -> {path} (track {track}) | peak az={agg.peak_angle_deg:.1f} deg "
              f"range={rng} x={agg.x_m:.2f} y={agg.y_m:.2f} conf={agg.confidence:.2f}",
              file=sys.stderr)
    elif args.mode == "run":
        publisher = JsonlPublisher(args.out, mode=args.head_mode)
        with publisher:
            results = run_inference(_sourceFromArgs(args), args.calibration, args.model,
                                     args.head_mode, publisher, vote=args.vote, guard=args.guard)
        print(f"published {len(results)} verdict(s)", file=sys.stderr)
    print('\a', end='', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
