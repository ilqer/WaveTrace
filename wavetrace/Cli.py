"""Phase 8 — the command-line entry point wiring hardware → signal → recognition → output.

Five modes (plan §5 Phase 8): capture / calibrate / collect-data / train / run. Each mode is a thin
argparse handler over a testable helper function; `run` is the real-time path (front-end → head →
publish) and reuses `Frontend.iterWindows` so the served features match training exactly.

CSI source today = synthetic (wavetrace.Synthetic) or a saved recording; live serial capture is a Phase-0 seam
(see Source.py). All non-`run` modes are offline.
"""

import argparse
import sys

import numpy as np

from wavetrace.Calibration import Calibration, imageBaseline, loadCalibration, saveCalibration
from wavetrace.Config import ModelConfig
from wavetrace.Frontend import iterWindows
from wavetrace.Localize import Localizer, Tracker, saveLocalization
from wavetrace.Source import buildCsiSource, parseTimeSpans, saveRecording
from wavetrace.groundtruth import (
    buildDataset,
    presenceLabelFn,
    saveDataset,
    weaponLabelFn,
)
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler
from wavetrace.output import JsonlPublisher
from wavetrace.recognition import SegmentVoter, modeSession, planInferenceInput, trainPresence, trainWeapon
from wavetrace import RecognitionResult


# ----- mode helpers (testable; argparse handlers below just parse + call these) --------------------

def calibrateSource(source, out_dir, *, baseline_packets=300, use_gain_lock=True, nbvi_max=12):
    """Run the calibration flow over a quiet-baseline source and persist the result. Offline."""
    cal = Calibration(baseline_packets=baseline_packets, nbvi_max=nbvi_max, use_gain_lock=use_gain_lock)
    for fr in source.frames():
        cal.observe(fr)
    result = cal.finalize()
    return saveCalibration(result, out_dir), result


def collectSource(source, calib_dir, out_dir, spans, *, stage="presence", window=128, hop=32,
                   session_id="", subject_id="", frame_average=1, subtract_baseline=False,
                   subtract_ic_baseline=False, labeler=None, tier=""):
    """Build + serialize a labeled dataset from a source + a label source. weapon stage emits the
    dual-block (intercarrier) dataset; presence emits the feature path.

    labeler: explicit label source (list[Label] / callable / Labeler) to use INSTEAD of the default
    scripted spans — pass a SegmentationLabeler/YoloSegLabeler (or a list of camera-produced Labels)
    to collect a mask-bearing, camera-supervised dataset that feeds the heatmap head. Defaults to a
    ScriptedLabeler over `spans` (the no-camera path).
    tier: 'open' | 'wrapped' | 'concealed', stamped into meta['tier'] so a concealed collection can be
    held out by evaluateConcealmentGap (the open→concealed transfer measurement)."""
    result, gainLock = loadCalibration(calib_dir)
    if labeler is None:
        labelFn = weaponLabelFn if stage == "weapon" else presenceLabelFn
        labeler = ScriptedLabeler([(s, e, True) for s, e in spans], label_fn=labelFn)
    intercarrier = stage == "weapon"
    # weapon IC/CNN paths need raw magnitudes: gain-locking cancels sigma2[p] (the metal discriminator)
    effectiveLock = None if intercarrier else gainLock
    ds = buildDataset(list(source.frames()), result, effectiveLock, labeler, window=window, hop=hop,
                       session_id=session_id, subject_id=subject_id, intercarrier=intercarrier,
                       frame_average=frame_average, subtract_baseline=subtract_baseline,
                       subtract_ic_baseline=subtract_ic_baseline)
    if tier:
        ds.meta["tier"] = tier
    return saveDataset(ds, out_dir), ds


def _spatialResult(t, x_m, y_m, angle_deg, range_m, confidence, located):
    """A spatial fix -> RecognitionResult on the wire schema: location rides in bbox [x, y, 0, 0]
    (Publisher.resultToDict emits it), azimuth + range in keypoints. class_id = 1 when this frame
    carries a real (measured/confident) fix, 0 when it is a coasted/low-confidence estimate. nan
    range -> -1 (JSON-safe)."""
    r = RecognitionResult()
    r.class_id = 1 if located else 0
    r.confidence = float(confidence)
    r.timestamp = float(t)
    r.bbox = [float(x_m), float(y_m), 0.0, 0.0]
    r.keypoints = [float(angle_deg), (-1.0 if np.isnan(range_m) else float(range_m))]
    return r


def localizeSource(source, out_dir, *, num_antennas, spacing=0.5, method="music", num_sources=1,
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
    for l in loc.locateStream(frames):
        if publisher is None:
            continue
        if tracker is not None:
            st = tracker.update(l)
            publisher.publish(_spatialResult(st.timestamp, st.x_m, st.y_m, st.angle_deg, st.range_m,
                                               st.confidence, st.measured))
        else:
            publisher.publish(_spatialResult(l.timestamp, l.x_m, l.y_m, l.peak_angle_deg,
                                              l.peak_range_m, l.confidence, l.confidence >= 0.5))
    agg = loc.aggregate(frames)
    return saveLocalization(agg, out_dir), agg


def runInference(source, calib_dir, model_path, mode, publisher, *, vote=False, guard=False):
    """Stream a source through the front-end and publish one verdict per window (+ a final soft-vote
    verdict when vote=True). When guard=True, wires AlertGuard+DriftMonitor for debounce and drift
    advisory (O(S)/frame extra — acceptable on the Pi serving side). O(windows)."""
    result, gainLock = loadCalibration(calib_dir)
    session = modeSession(mode, model_path)
    applyLock, intercarrier, pick = planInferenceInput(mode, session.head)
    cfg = session.head.config

    imgSubc = getattr(result, "image_subcarriers", None)
    imgBase = None
    if cfg.subtract_baseline:
        imgBase = imageBaseline(result, locked=(applyLock and gainLock is not None))
    # weapon IC background subtraction (Item 10/CAUSE 2B): raw baseline, IC path only, mirrors training
    icBase = result.baseline_mag if getattr(cfg, "subtract_ic_baseline", False) else None

    framesIter = source.frames()
    if guard:
        from wavetrace.output.Guard import AlertGuard, DriftMonitor
        driftMon = DriftMonitor(result.baseline_mag)
        alertGuard = AlertGuard()
        # tee raw (pre-lock) per-frame mags to DriftMonitor without disrupting the frame stream
        def _teeDrift(frames, monitor, pub):
            import numpy as _np
            for fr in frames:
                ev = monitor.update(float(fr.timestamp),
                                    _np.abs(_np.asarray(fr.grid)).mean(axis=0).astype(_np.float32))
                if ev:
                    pub.publishEvent(ev)
                yield fr
        framesIter = _teeDrift(framesIter, driftMon, publisher)

    voter = SegmentVoter() if vote else None
    out = []
    for t, features, image, ic in iterWindows(
        framesIter, result.subcarriers, gainLock if applyLock else None,
        window=cfg.window, hop=cfg.hop, intercarrier=intercarrier,
        image_subcarriers=imgSubc,
        frame_average=cfg.frame_average,
        imageBaseline=imgBase,
        ic_baseline=icBase,
    ):
        cls, conf = session.predictWindow(pick(features, image, ic))
        r = RecognitionResult(); r.class_id = cls; r.confidence = conf; r.timestamp = t
        publisher.publish(r)
        out.append(r)
        if guard:
            ev = alertGuard.update(t, cls)
            if ev:
                publisher.publishEvent(ev)
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

def _sourceFromArgs(args):
    """Thin alias over `wavetrace.Source.buildCsiSource` kept under this private name so existing
    direct importers of it are unaffected; new code should import `buildCsiSource`."""
    return buildCsiSource(args)


def _addSourceArgs(p):
    p.add_argument("--recording", help="replay a saved recording directory")
    p.add_argument("--synthetic", action="store_true", help="generate frames in-process (no hardware)")
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

    pCap = sub.add_parser("capture", help="record CSI frames to disk")
    _addSourceArgs(pCap)
    pCap.add_argument("--out", required=True)

    pCal = sub.add_parser("calibrate", help="quiet-baseline calibration -> calibration dir")
    _addSourceArgs(pCal)
    pCal.add_argument("--out", required=True)
    pCal.add_argument("--baseline-packets", type=int, default=300, dest="baseline_packets")
    pCal.add_argument("--no-gain-lock", action="store_true", dest="no_gain_lock")

    pCol = sub.add_parser("collect-data", help="frames + scripted labels -> dataset dir")
    _addSourceArgs(pCol)
    pCol.add_argument("--calibration", required=True)
    pCol.add_argument("--out", required=True)
    pCol.add_argument("--stage", choices=["presence", "weapon"], default="presence")
    pCol.add_argument("--label-spans", default="", dest="label_spans",
                       help="present/weapon spans 'a:b,c:d'")
    pCol.add_argument("--window", type=int, default=128)
    pCol.add_argument("--hop", type=int, default=32)
    pCol.add_argument("--session-id", default="", dest="session_id")
    pCol.add_argument("--subject-id", default="", dest="subject_id")
    pCol.add_argument("--tier", choices=["open", "wrapped", "concealed"], default="",
                       help="weapon ground-truth tier -> meta['tier'] (concealed = held-out test split)")
    pCol.add_argument("--frame-average", type=int, default=1, dest="frame_average")
    pCol.add_argument("--subtract-baseline", action="store_true", dest="subtract_baseline")

    pTr = sub.add_parser("train", help="dataset(s) -> model")
    pTr.add_argument("datasets", nargs="+")
    pTr.add_argument("--out", required=True)
    pTr.add_argument("--stage", choices=["presence", "weapon"], default="presence")
    pTr.add_argument("--backend", default=None, help="mlp|svm|variance|cnn (default per stage)")
    pTr.add_argument("--feature-mode", default="ic27", dest="feature_mode",
                      choices=["ic27", "fusion", "cnn"], help="weapon stage only")

    pLoc = sub.add_parser("localize", help="AoA spatial heatmap (where) -> track + heatmap dir")
    _addSourceArgs(pLoc)
    pLoc.add_argument("--out", required=True)
    pLoc.add_argument("--spacing", type=float, default=0.5,
                       help="ULA element spacing in wavelengths (default 0.5 = lambda/2)")
    pLoc.add_argument("--method", choices=["music", "bartlett"], default="music")
    pLoc.add_argument("--num-sources", type=int, default=1, dest="num_sources")
    pLoc.add_argument("--num-angles", type=int, default=181, dest="num_angles")
    pLoc.add_argument("--subcarrier-hz", type=float, default=312.5e3, dest="subcarrier_hz",
                       help="subcarrier spacing for the range axis (HT20/64 = 312.5 kHz)")
    pLoc.add_argument("--max-range-m", type=float, default=12.0, dest="max_range_m")
    pLoc.add_argument("--num-ranges", type=int, default=64, dest="num_ranges",
                       help="range grid resolution of the joint 2-D room map")
    pLoc.add_argument("--no-range", action="store_true", dest="no_range",
                       help="azimuth only (skip the joint 2-D range axis)")
    pLoc.add_argument("--track", default=None,
                       help="JSONL file for the per-frame localization track (default <out>/track.jsonl)")
    pLoc.add_argument("--no-filter", action="store_true", dest="no_filter",
                       help="publish raw per-frame fixes (skip the constant-velocity Kalman tracker)")

    pRun = sub.add_parser("run", help="stream inference -> publish verdicts")
    _addSourceArgs(pRun)
    pRun.add_argument("--calibration", required=True)
    pRun.add_argument("--model", required=True)
    pRun.add_argument("--head-mode", choices=["presence", "weapon"], default="presence",
                       dest="head_mode", help="which operating mode to serve")
    pRun.add_argument("--out", default=None, help="JSONL output file (default stdout)")
    pRun.add_argument("--vote", action="store_true", help="also emit a final soft-vote verdict")
    pRun.add_argument("--guard", action="store_true", help="enable AlertGuard debounce + DriftMonitor advisory")

    args = ap.parse_args(argv)

    if args.mode == "capture":
        saveRecording(list(_sourceFromArgs(args).frames()), args.out)
        print(f"captured -> {args.out}", file=sys.stderr)
    elif args.mode == "calibrate":
        path, _ = calibrateSource(_sourceFromArgs(args), args.out,
                                   baseline_packets=args.baseline_packets,
                                   use_gain_lock=not args.no_gain_lock)
        print(f"calibration -> {path}", file=sys.stderr)
    elif args.mode == "collect-data":
        path, ds = collectSource(_sourceFromArgs(args), args.calibration, args.out,
                                  parseTimeSpans(args.label_spans), stage=args.stage,
                                  window=args.window, hop=args.hop,
                                  session_id=args.session_id, subject_id=args.subject_id,
                                  frame_average=args.frame_average,
                                  subtract_baseline=args.subtract_baseline, tier=args.tier)
        print(f"dataset ({ds.y.size} samples) -> {path}", file=sys.stderr)
    elif args.mode == "train":
        if args.stage == "presence":
            _, m = trainPresence(args.datasets, out_dir=args.out)  # k taken from dataset meta
        else:
            cfg = None
            if args.backend:
                # config only needed to override the backend; k still comes from dataset meta
                from wavetrace.groundtruth import loadDataset
                k = int(loadDataset(args.datasets[0]).meta["K"])
                cfg = ModelConfig(stage="weapon", k=k, backend=args.backend)
            _, m = trainWeapon(args.datasets, out_dir=args.out, config=cfg,
                                feature_mode=args.feature_mode)
        print(f"model -> {args.out} ({m})", file=sys.stderr)
    elif args.mode == "localize":
        from pathlib import Path
        track = args.track or str(Path(args.out) / "track.jsonl")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with JsonlPublisher(track, mode="localize") as pub:
            path, agg = localizeSource(
                _sourceFromArgs(args), args.out, num_antennas=args.antennas, spacing=args.spacing,
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
            results = runInference(_sourceFromArgs(args), args.calibration, args.model,
                                    args.head_mode, pub, vote=args.vote, guard=args.guard)
        print(f"published {len(results)} verdict(s)", file=sys.stderr)
    print('\a', end='', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
