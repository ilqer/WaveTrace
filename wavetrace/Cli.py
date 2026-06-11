"""Phase 8 — the command-line entry point wiring hardware → signal → recognition → output.

Five modes (plan §5 Phase 8): capture / calibrate / collect-data / train / run. Each mode is a thin
argparse handler over a testable helper function; `run` is the real-time path (front-end → head →
publish) and reuses `Frontend.iter_windows` so the served features match training exactly.

CSI source today = synthetic (fixtures) or a saved recording; live serial capture is a Phase-0 seam
(see Source.py). All non-`run` modes are offline.
"""

import argparse
import sys
import warnings

import numpy as np

from wavetrace.Calibration import Calibration, load_calibration, save_calibration
from wavetrace.Config import ModelConfig
from wavetrace.Frontend import iter_windows
from wavetrace.Localize import Localizer, Tracker, save_localization
from wavetrace.Source import RecordingSource, SyntheticSource, load_recording, save_recording
from wavetrace.groundtruth import (
    build_dataset,
    presence_label_fn,
    save_dataset,
    weapon_label_fn,
)
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler
from wavetrace.output import JsonlPublisher
from wavetrace.recognition import SegmentVoter, mode_session, train_presence, train_weapon
from wavetrace import RecognitionResult


# ----- front-end serving config: how (mode, head) maps to the inference input ----------------------

def _serving_plan(mode: str, head):
    """Return (apply_lock, intercarrier, pick) for the run loop. `pick(features, image, ic) -> x` is
    the row fed to predict_window. Encodes the plan's (mode, backend) wiring table."""
    if mode == "presence":
        return True, False, (lambda f, i, ic: f)
    # weapon: self-describing via head.feature_mode (fallback by backend for pre-P8 models)
    fm = getattr(head, "feature_mode", None) or ("cnn" if head.config.backend == "cnn" else "ic27")
    if fm == "cnn":
        return False, False, (lambda f, i, ic: i.reshape(-1))
    if fm == "fusion":
        return True, True, (lambda f, i, ic: np.hstack([ic, f]))
    return False, True, (lambda f, i, ic: ic)  # ic27 / variance


# ----- mode helpers (testable; argparse handlers below just parse + call these) --------------------

def calibrate_source(source, out_dir, *, baseline_packets=300, use_gain_lock=True, nbvi_max=12):
    """Run the calibration flow over a quiet-baseline source and persist the result. Offline."""
    cal = Calibration(baseline_packets=baseline_packets, nbvi_max=nbvi_max, use_gain_lock=use_gain_lock)
    for fr in source.frames():
        cal.observe(fr)
    result = cal.finalize()
    return save_calibration(result, out_dir), result


def collect_source(source, calib_dir, out_dir, spans, *, stage="presence", window=128, hop=32,
                   session_id="", subject_id=""):
    """Build + serialize a labeled dataset from a source + scripted spans. weapon stage emits the
    dual-block (intercarrier) dataset under the locked calibration; presence emits the feature path."""
    result, gain_lock = load_calibration(calib_dir)
    label_fn = weapon_label_fn if stage == "weapon" else presence_label_fn
    labeler = ScriptedLabeler([(s, e, True) for s, e in spans], label_fn=label_fn)
    intercarrier = stage == "weapon"
    ds = build_dataset(list(source.frames()), result, gain_lock, labeler, window=window, hop=hop,
                       session_id=session_id, subject_id=subject_id, intercarrier=intercarrier)
    return save_dataset(ds, out_dir), ds


def _spatial_result(t, x_m, y_m, angle_deg, range_m, confidence, located):
    """A spatial fix -> RecognitionResult on the wire schema: location rides in bbox [x, y, 0, 0]
    (Publisher.result_to_dict emits it), azimuth + range in keypoints. class_id = 1 when this frame
    carries a real (measured/confident) fix, 0 when it is a coasted/low-confidence estimate. nan
    range -> -1 (JSON-safe)."""
    r = RecognitionResult()
    r.class_id = 1 if located else 0
    r.confidence = float(confidence)
    r.timestamp = float(t)
    r.bbox = [float(x_m), float(y_m), 0.0, 0.0]
    r.keypoints = [float(angle_deg), (-1.0 if np.isnan(range_m) else float(range_m))]
    return r


def localize_source(source, out_dir, *, num_antennas, spacing=0.5, method="music", num_sources=1,
                    num_angles=181, subcarrier_spacing_hz=312.5e3, max_range_m=12.0, num_ranges=64,
                    range_enabled=True, filter_track=True, publisher=None):
    """Stream a source through the AoA Localizer: publish the per-frame track as RecognitionResults
    (Publisher wire schema) and persist the aggregate joint-2-D room map. Returns (path, aggregate
    Localization). Needs >= 2 RX antennas (2-antenna ESP32 / Pi NIC).

    filter_track (default on): smooth the raw per-frame measurements with a constant-velocity Kalman
    `Tracker` — predict from motion (no teleporting), fuse each measurement weighted by its confidence,
    and gate impossible jumps. The PUBLISHED track is the filtered one; the saved room map is the
    raw aggregate. O(F·(A²S + A·G) + (A·S)³)."""
    loc = Localizer(num_antennas, spacing=spacing, method=method, num_sources=num_sources,
                    num_angles=num_angles, subcarrier_spacing_hz=subcarrier_spacing_hz,
                    max_range_m=max_range_m, num_ranges=num_ranges, range_enabled=range_enabled)
    tracker = Tracker(range_enabled=range_enabled) if filter_track else None
    frames = list(source.frames())
    for l in loc.locate_stream(frames):
        if publisher is None:
            continue
        if tracker is not None:
            st = tracker.update(l)
            publisher.publish(_spatial_result(st.timestamp, st.x_m, st.y_m, st.angle_deg, st.range_m,
                                               st.confidence, st.measured))
        else:
            publisher.publish(_spatial_result(l.timestamp, l.x_m, l.y_m, l.peak_angle_deg,
                                              l.peak_range_m, l.confidence, l.confidence >= 0.5))
    agg = loc.aggregate(frames)
    return save_localization(agg, out_dir), agg


def run_inference(source, calib_dir, model_path, mode, publisher, *, vote=False):
    """Stream a source through the front-end and publish one verdict per window (+ a final soft-vote
    verdict when vote=True). Returns the published RecognitionResults. O(windows)."""
    result, gain_lock = load_calibration(calib_dir)
    session = mode_session(mode, model_path)
    apply_lock, intercarrier, pick = _serving_plan(mode, session.head)
    cfg = session.head.config
    voter = SegmentVoter() if vote else None
    out = []
    for t, features, image, ic in iter_windows(
        source.frames(), result.subcarriers, gain_lock if apply_lock else None,
        window=cfg.window, hop=cfg.hop, intercarrier=intercarrier,
    ):
        cls, conf = session.predict_window(pick(features, image, ic))
        r = RecognitionResult(); r.class_id = cls; r.confidence = conf; r.timestamp = t
        publisher.publish(r)
        out.append(r)
        if voter is not None:
            voter.add(session.head.predict_proba(np.asarray(pick(features, image, ic),
                                                            dtype=np.float32).reshape(1, -1))[0])
    if voter is not None and len(voter):
        vcls, vmean = voter.finalize()
        r = RecognitionResult(); r.class_id = int(vcls); r.confidence = float(vmean[vcls])
        r.timestamp = out[-1].timestamp if out else 0.0
        publisher.publish(r)
        out.append(r)
    return out


# ----- argparse layer -----------------------------------------------------------------------------

def _source_from_args(args):
    """Build a CsiSource from CLI args: --recording DIR (replay) or --synthetic (fixtures)."""
    if args.recording:
        return RecordingSource(args.recording)
    if args.synthetic:
        from fixtures.SyntheticRecording import generatePairedRecording
        if _parse_spans(args.weapon) and args.weapon_depth <= 0.0:
            # weapon spans with depth 0 inject NO signal -> weapon windows are physically identical
            # to no-weapon ones; the resulting dataset is unlearnable (and WeaponHead.fit will reject
            # it as single-class). Warn loudly instead of silently producing a dead model (B3).
            warnings.warn("synthetic --weapon spans set but --weapon-depth is 0: weapon windows will "
                          "carry no signature (pass --weapon-depth > 0)", stacklevel=2)
        spans = _parse_spans(args.presence)
        frames, _, _ = generatePairedRecording(
            numAntennas=args.antennas, numSubcarriers=args.subcarriers, sampleRateHz=args.fs,
            durationS=args.duration, cameraFps=30.0, presenceSpans=spans or [(0.0, args.duration)],
            presenceTurbulenceStd=0.10, weaponSpans=_parse_spans(args.weapon),
            weaponSignatureDepth=args.weapon_depth, seed=args.seed,
        )
        return SyntheticSource(frames)
    raise SystemExit("a source is required: --recording DIR or --synthetic")


def _parse_spans(s):
    """'a:b,c:d' -> [(a,b),(c,d)]; '' -> []."""
    if not s:
        return []
    return [tuple(float(x) for x in part.split(":")) for part in s.split(",")]


def _add_source_args(p):
    p.add_argument("--recording", help="replay a saved recording directory")
    p.add_argument("--synthetic", action="store_true", help="generate frames via the fixtures")
    p.add_argument("--antennas", type=int, default=2)
    p.add_argument("--subcarriers", type=int, default=32)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--presence", default="", help="presence spans 'a:b,c:d' (synthetic)")
    p.add_argument("--weapon", default="", help="weapon spans 'a:b,c:d' (synthetic)")
    p.add_argument("--weapon-depth", type=float, default=0.0, dest="weapon_depth")
    p.add_argument("--seed", type=int, default=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wavetrace", description="WiFi-CSI sensing pipeline")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_cap = sub.add_parser("capture", help="record CSI frames to disk")
    _add_source_args(p_cap)
    p_cap.add_argument("--out", required=True)

    p_cal = sub.add_parser("calibrate", help="quiet-baseline calibration -> calibration dir")
    _add_source_args(p_cal)
    p_cal.add_argument("--out", required=True)
    p_cal.add_argument("--baseline-packets", type=int, default=300, dest="baseline_packets")
    p_cal.add_argument("--no-gain-lock", action="store_true", dest="no_gain_lock")

    p_col = sub.add_parser("collect-data", help="frames + scripted labels -> dataset dir")
    _add_source_args(p_col)
    p_col.add_argument("--calibration", required=True)
    p_col.add_argument("--out", required=True)
    p_col.add_argument("--stage", choices=["presence", "weapon"], default="presence")
    p_col.add_argument("--label-spans", default="", dest="label_spans",
                       help="present/weapon spans 'a:b,c:d'")
    p_col.add_argument("--window", type=int, default=128)
    p_col.add_argument("--hop", type=int, default=32)
    p_col.add_argument("--session-id", default="", dest="session_id")
    p_col.add_argument("--subject-id", default="", dest="subject_id")

    p_tr = sub.add_parser("train", help="dataset(s) -> model")
    p_tr.add_argument("datasets", nargs="+")
    p_tr.add_argument("--out", required=True)
    p_tr.add_argument("--stage", choices=["presence", "weapon"], default="presence")
    p_tr.add_argument("--backend", default=None, help="mlp|svm|variance|cnn (default per stage)")
    p_tr.add_argument("--feature-mode", default="ic27", dest="feature_mode",
                      choices=["ic27", "fusion", "cnn"], help="weapon stage only")

    p_loc = sub.add_parser("localize", help="AoA spatial heatmap (where) -> track + heatmap dir")
    _add_source_args(p_loc)
    p_loc.add_argument("--out", required=True)
    p_loc.add_argument("--spacing", type=float, default=0.5,
                       help="ULA element spacing in wavelengths (default 0.5 = lambda/2)")
    p_loc.add_argument("--method", choices=["music", "bartlett"], default="music")
    p_loc.add_argument("--num-sources", type=int, default=1, dest="num_sources")
    p_loc.add_argument("--num-angles", type=int, default=181, dest="num_angles")
    p_loc.add_argument("--subcarrier-hz", type=float, default=312.5e3, dest="subcarrier_hz",
                       help="subcarrier spacing for the range axis (HT20/64 = 312.5 kHz)")
    p_loc.add_argument("--max-range-m", type=float, default=12.0, dest="max_range_m")
    p_loc.add_argument("--num-ranges", type=int, default=64, dest="num_ranges",
                       help="range grid resolution of the joint 2-D room map")
    p_loc.add_argument("--no-range", action="store_true", dest="no_range",
                       help="azimuth only (skip the joint 2-D range axis)")
    p_loc.add_argument("--track", default=None,
                       help="JSONL file for the per-frame localization track (default <out>/track.jsonl)")
    p_loc.add_argument("--no-filter", action="store_true", dest="no_filter",
                       help="publish raw per-frame fixes (skip the constant-velocity Kalman tracker)")

    p_run = sub.add_parser("run", help="stream inference -> publish verdicts")
    _add_source_args(p_run)
    p_run.add_argument("--calibration", required=True)
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--head-mode", choices=["presence", "weapon"], default="presence",
                       dest="head_mode", help="which operating mode to serve")
    p_run.add_argument("--out", default=None, help="JSONL output file (default stdout)")
    p_run.add_argument("--vote", action="store_true", help="also emit a final soft-vote verdict")

    args = ap.parse_args(argv)

    if args.mode == "capture":
        save_recording(list(_source_from_args(args).frames()), args.out)
        print(f"captured -> {args.out}", file=sys.stderr)
    elif args.mode == "calibrate":
        path, _ = calibrate_source(_source_from_args(args), args.out,
                                   baseline_packets=args.baseline_packets,
                                   use_gain_lock=not args.no_gain_lock)
        print(f"calibration -> {path}", file=sys.stderr)
    elif args.mode == "collect-data":
        path, ds = collect_source(_source_from_args(args), args.calibration, args.out,
                                  _parse_spans(args.label_spans), stage=args.stage,
                                  window=args.window, hop=args.hop,
                                  session_id=args.session_id, subject_id=args.subject_id)
        print(f"dataset ({ds.y.size} samples) -> {path}", file=sys.stderr)
    elif args.mode == "train":
        if args.stage == "presence":
            _, m = train_presence(args.datasets, out_dir=args.out)  # k taken from dataset meta
        else:
            cfg = None
            if args.backend:
                # k is filled from the dataset meta inside train_weapon when config is None; pass a
                # config only to override the backend
                from wavetrace.groundtruth import load_dataset
                k = int(load_dataset(args.datasets[0]).meta["K"])
                cfg = ModelConfig(stage="weapon", k=k, backend=args.backend)
            _, m = train_weapon(args.datasets, out_dir=args.out, config=cfg,
                                feature_mode=args.feature_mode)
        print(f"model -> {args.out} ({m})", file=sys.stderr)
    elif args.mode == "localize":
        from pathlib import Path
        track = args.track or str(Path(args.out) / "track.jsonl")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with JsonlPublisher(track, mode="localize") as pub:
            path, agg = localize_source(
                _source_from_args(args), args.out, num_antennas=args.antennas, spacing=args.spacing,
                method=args.method, num_sources=args.num_sources, num_angles=args.num_angles,
                subcarrier_spacing_hz=args.subcarrier_hz, max_range_m=args.max_range_m,
                num_ranges=args.num_ranges, range_enabled=not args.no_range,
                filter_track=not args.no_filter, publisher=pub,
            )
        rng = "n/a" if np.isnan(agg.peak_range_m) else f"{agg.peak_range_m:.2f} m"
        print(f"localization -> {path} (track {track}) | peak az={agg.peak_angle_deg:.1f} deg "
              f"range={rng} x={agg.x_m:.2f} y={agg.y_m:.2f} conf={agg.confidence:.2f}",
              file=sys.stderr)
    elif args.mode == "run":
        pub = JsonlPublisher(args.out, mode=args.head_mode)
        with pub:
            results = run_inference(_source_from_args(args), args.calibration, args.model,
                                    args.head_mode, pub, vote=args.vote)
        print(f"published {len(results)} verdict(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
