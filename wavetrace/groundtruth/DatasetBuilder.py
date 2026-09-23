"""Run a CSI recording through the front-end, attach aligned labels, and save a labeled dataset.

Every sample stores both recognition inputs, so a later model can pick either without re-running
the recording:
  * X_features (n, 9·K) - nine features per NBVI subcarrier
  * X_image    (n, K_img, window) - subcarrier × time CSI "image"; K_img is the full set that
    passed the noise gate when calibration selected one, otherwise the NBVI K

The label itself is binary, but its box, keypoints and any weapon position ride along into the
manifest, so heatmap and location work needs no re-run either.

Serialization = JSONL manifest + .npy arrays under data/<name>/ (gitignored):
    meta.json       dataset-level: fs, K, K_img, subcarriers, image_subcarriers, window/hop,
                    tolerance, class_names, sync_error, frame_average, subtract_baseline
    manifest.jsonl  one line per sample: i, t, class_id, name, bbox, keypoints, mask, mask_grid, dt
    features.npy    (n, 9K) float32        images.npy  (n, K_img, window) float32
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from wavetrace import InterCarrierExtractor, Label
from wavetrace.Calibration import build_image_baseline
from wavetrace.Frontend import demuxByNode, iterWindows, iterWindowsStacked
from wavetrace.groundtruth.Align import align


@dataclass
class Dataset:
    X_features: np.ndarray            # (n, 9K) float32
    X_image: np.ndarray              # (n, K_img, window) float32
    y: np.ndarray                    # (n,) int64 stage class
    t: np.ndarray                    # (n,) float64 window-END timestamps
    labels: list[Label]             # full Labels (box/keypoints/position preserved)
    meta: dict = field(default_factory=dict)
    session_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    subject_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    # (n, 27) inter-carrier block (µ|σ²|CV × 9) from RAW magnitudes; None unless built with
    # intercarrier=True, which requires gainLock=None.
    X_intercarrier: np.ndarray | None = None


def _attachLabels(wts, label_source, tolerance):
    """Attach aligned or callable labels to a list of window timestamps. Returns (sel, sel_labels, stats)."""
    if callable(label_source):
        selLabels = [label_source(t) for t in wts]
        sel = list(range(len(wts)))
        stats = {"mean_dt": 0.0, "max_abs_dt": 0.0, "p95_abs_dt": 0.0,
                 "matched": len(wts), "dropped": 0}
    else:
        res = align(wts, label_source, tolerance)
        sel = [wi for wi, _ in res.matched]
        selLabels = [lab for _, lab in res.matched]
        stats = res.stats
    return sel, selLabels, stats


def buildDataset(
    frames,
    calibration_result,
    gainLock,
    label_source,
    *,
    window: int = 128,
    hop: int = 32,
    tolerance: float = 0.05,
    class_names=None,
    session_id: str = "",
    subject_id: str = "",
    intercarrier: bool = False,
    frame_average: int = 1,
    subtract_baseline: bool = False,
    subtract_ic_baseline: bool = False,
) -> "Dataset":
    """Build a labeled dataset from a CSI recording + a label source.

    `label_source` is either a list[Label] - camera or replay, nearest-matched to each window with
    the sync error measured - or a callable t->Label on the same CSI clock, evaluated at every
    window timestamp and never dropped.

    `gainLock` = the locked GainLock from `calibration`, OR None to skip the per-frame amplitude
    rescale. Pass None for material and weapon datasets: the gain lock normalizes every frame to a
    common mean, which erases the bulk attenuation σ²[p] and reflectionSignature measure. It is
    for the amplitude and presence path only.

    session_id / subject_id: group ids of this recording, stamped on every sample; the
      leave-one-group-out evaluation folds on them.
    intercarrier: also emit the (n, 27) inter-carrier block from RAW magnitudes.
    frame_average: M >= 1 non-overlapping decimating mean; 1 leaves the rate alone.
    subtract_baseline: subtract the quiet-room baseline from the image path only.
    subtract_ic_baseline: subtract the raw quiet-room baseline from the inter-carrier path, so the
      weapon's σ²[p] sees the perturbation rather than the whole static room. Independent of
      subtract_baseline."""
    subc = np.asarray(calibration_result.subcarriers, dtype=np.intp)
    K = int(subc.size)
    imgSubcList = getattr(calibration_result, "image_subcarriers", None) or list(calibration_result.subcarriers)
    KImg = len(imgSubcList)

    imgBaselineArr = None
    if subtract_baseline:
        imgBaselineArr = build_image_baseline(calibration_result, locked=(gainLock is not None))
    # IC background subtraction uses the RAW baseline (the IC path is always raw, gainLock=None for
    # weapon), so no locked-basis rescale — just the quiet-room mean |H| per subcarrier.
    icBaselineArr = (np.asarray(calibration_result.baseline_mag, dtype=np.float32)
                       if (subtract_ic_baseline and intercarrier) else None)

    # materialize once: iterWindows consumes the stream, and the fs estimate below re-indexes
    # frames[-1], which a bare generator would have exhausted by then.
    frames = list(frames)

    feats: list[np.ndarray] = []
    imgs: list[np.ndarray] = []
    ics: list[np.ndarray] = []
    wts: list[float] = []
    for t, features, image, ic in iterWindows(
        frames, subc, gainLock, window=window, hop=hop, intercarrier=intercarrier,
        image_subcarriers=(imgSubcList if imgSubcList != list(calibration_result.subcarriers) else None),
        frame_average=frame_average,
        imageBaseline=imgBaselineArr,
        ic_baseline=icBaselineArr,
    ):
        feats.append(features.copy())
        imgs.append(image.copy())
        if ic is not None:
            ics.append(ic.copy())
        wts.append(t)

    sel, selLabels, stats = _attachLabels(wts, label_source, tolerance)

    if sel:
        XFeatures = np.stack([feats[i] for i in sel]).astype(np.float32)
        XImage = np.stack([imgs[i] for i in sel]).astype(np.float32)
    else:
        XFeatures = np.empty((0, 9 * K), np.float32)
        XImage = np.empty((0, KImg, window), np.float32)
    y = np.asarray([l.class_id for l in selLabels], dtype=np.int64)
    tArr = np.asarray([wts[i] for i in sel], dtype=np.float64)

    # measure fs from the frame timestamps; the configured packet rate is not what arrives
    fs = ((len(frames) - 1) / (frames[-1].timestamp - frames[0].timestamp)
          if len(frames) > 1 and frames[-1].timestamp > frames[0].timestamp else 0.0)
    meta = {
        "fs": float(fs),
        "K": K,
        "K_img": KImg,
        # raw capture width (64/128/192 by band), not K, which is the NBVI-selected subset of it
        "num_subcarriers": int(frames[0].num_subcarriers),
        "subcarriers": [int(s) for s in calibration_result.subcarriers],
        "image_subcarriers": imgSubcList,
        "subtract_ic_baseline": bool(icBaselineArr is not None),
        "window": window,
        "hop": hop,
        "tolerance": tolerance,
        "gain_locked": gainLock is not None,
        "class_names": dict(class_names) if class_names else {},
        "sync_error": {"mean_dt": stats["mean_dt"], "max_abs_dt": stats["max_abs_dt"],
                       "p95_abs_dt": stats["p95_abs_dt"]},
        "n_samples": int(y.size),
        "n_dropped": int(stats["dropped"]),
        "intercarrier": bool(intercarrier),
        "frame_average": int(frame_average),
        "subtract_baseline": bool(subtract_baseline),
    }
    XIc = None
    if intercarrier:
        icWidth = InterCarrierExtractor(window=window, hop=hop).output_size
        XIc = (np.stack([ics[i] for i in sel]).astype(np.float32) if sel
                else np.empty((0, icWidth), np.float32))
    return Dataset(
        X_features=XFeatures, X_image=XImage, y=y, t=tArr, labels=selLabels, meta=meta,
        session_ids=np.full(y.size, str(session_id), dtype=object),
        subject_ids=np.full(y.size, str(subject_id), dtype=object),
        X_intercarrier=XIc,
    )


def buildDatasetStacked(
    frames,
    calibrations,
    label_source,
    *,
    window: int = 128,
    hop: int = 32,
    tolerance: float = 0.05,
    node_tolerance: float = 0.05,
    class_names=None,
    session_id: str = "",
    subject_id: str = "",
    intercarrier: bool = False,
    frame_average: int = 1,
    subtract_baseline: bool = False,
) -> "Dataset":
    """Build a labeled dataset from a multi-node CSI recording (nodes stacked as channels).

    calibrations: dict[node_id -> (CalibrationResult, GainLock|None)].
    Frames from all nodes are demuxed by node_id and fed through iterWindowsStacked.
    Shapes: X_features (n, N·9·K), X_image (n, N, K_img, window), X_intercarrier (n, N·27).
    """
    frames = list(frames)
    byNode = demuxByNode(frames)

    nodeIds = sorted(calibrations.keys())
    perNodeCalib = {}
    for nid in nodeIds:
        calResult, gainLock = calibrations[nid]
        imgSubc = getattr(calResult, "image_subcarriers", None) or list(calResult.subcarriers)
        base = build_image_baseline(calResult, locked=(gainLock is not None)) if subtract_baseline else None
        perNodeCalib[nid] = (list(calResult.subcarriers), imgSubc, gainLock, base)

    # all nodes share these widths; iterWindowsStacked has already checked that
    firstCal = calibrations[nodeIds[0]][0]
    K = len(firstCal.subcarriers)
    imgSubcList = getattr(firstCal, "image_subcarriers", None) or list(firstCal.subcarriers)
    KImg = len(imgSubcList)
    N = len(nodeIds)

    feats: list[np.ndarray] = []
    imgs: list[np.ndarray] = []
    ics: list[np.ndarray] = []
    wts: list[float] = []
    for t, feat, image, ic in iterWindowsStacked(
        byNode, perNodeCalib, window=window, hop=hop, intercarrier=intercarrier,
        frame_average=frame_average, node_tolerance=node_tolerance,
    ):
        feats.append(feat.copy())
        imgs.append(image.copy())
        if ic is not None:
            ics.append(ic.copy())
        wts.append(t)

    sel, selLabels, stats = _attachLabels(wts, label_source, tolerance)

    if sel:
        XFeatures = np.stack([feats[i] for i in sel]).astype(np.float32)
        XImage = np.stack([imgs[i] for i in sel]).astype(np.float32)
    else:
        XFeatures = np.empty((0, N * 9 * K), np.float32)
        XImage = np.empty((0, N, KImg, window), np.float32)
    y = np.asarray([l.class_id for l in selLabels], dtype=np.int64)
    tArr = np.asarray([wts[i] for i in sel], dtype=np.float64)

    # fs from the lowest node id's frames
    node0Frames = byNode.get(nodeIds[0], [])
    fs = ((len(node0Frames) - 1) / (node0Frames[-1].timestamp - node0Frames[0].timestamp)
          if len(node0Frames) > 1 and node0Frames[-1].timestamp > node0Frames[0].timestamp
          else 0.0)

    meta = {
        "fs": float(fs),
        "K": K,
        "K_img": KImg,
        # raw per-frame capture width of node0's link, same meaning as buildDataset's; empty
        # node0Frames (degenerate input) falls back to K rather than indexing an empty list.
        "num_subcarriers": int(node0Frames[0].num_subcarriers) if node0Frames else K,
        "subcarriers": [int(s) for s in firstCal.subcarriers],
        "image_subcarriers": imgSubcList,
        "window": window,
        "hop": hop,
        "tolerance": tolerance,
        "node_ids": nodeIds,
        "num_nodes": N,
        "node_tolerance": node_tolerance,
        "gain_locked": any(calibrations[nid][1] is not None for nid in nodeIds),
        "class_names": dict(class_names) if class_names else {},
        "sync_error": {"mean_dt": stats["mean_dt"], "max_abs_dt": stats["max_abs_dt"],
                       "p95_abs_dt": stats["p95_abs_dt"]},
        "n_samples": int(y.size),
        "n_dropped": int(stats["dropped"]),
        "intercarrier": bool(intercarrier),
        "frame_average": int(frame_average),
        "subtract_baseline": bool(subtract_baseline),
    }
    XIc = None
    if intercarrier:
        icWidth = N * 27
        XIc = (np.stack([ics[i] for i in sel]).astype(np.float32) if sel
                else np.empty((0, icWidth), np.float32))
    return Dataset(
        X_features=XFeatures, X_image=XImage, y=y, t=tArr, labels=selLabels, meta=meta,
        session_ids=np.full(y.size, str(session_id), dtype=object),
        subject_ids=np.full(y.size, str(subject_id), dtype=object),
        X_intercarrier=XIc,
    )


def saveDataset(dataset: "Dataset", out_dir) -> Path:
    """Serialize to JSONL manifest + .npy arrays under out_dir (created if missing). O(n)."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "features.npy", dataset.X_features)
    np.save(p / "images.npy", dataset.X_image)
    if dataset.X_intercarrier is not None:
        np.save(p / "features_ic.npy", dataset.X_intercarrier)
    with open(p / "manifest.jsonl", "w") as f:
        for i, lab in enumerate(dataset.labels):
            rec = {
                "i": i,
                "t": float(dataset.t[i]),
                "class_id": int(dataset.y[i]),
                "name": lab.name,
                "bbox": list(lab.bbox) if lab.bbox is not None else None,
                "keypoints": list(lab.keypoints),
                # heatmap target: the camera mask pooled to a G×G grid, set only by a SegmentationLabeler
                "mask": [float(v) for v in lab.mask] if lab.mask else None,
                "mask_grid": int(lab.mask_grid) if lab.mask_grid else None,
                "session_id": str(dataset.session_ids[i]) if dataset.session_ids.size else "",
                "subject_id": str(dataset.subject_ids[i]) if dataset.subject_ids.size else "",
            }
            f.write(json.dumps(rec) + "\n")
    with open(p / "meta.json", "w") as f:
        json.dump(dataset.meta, f, indent=2)
    return p


def loadDataset(out_dir) -> "Dataset":
    """Round-trip load of a saved dataset. O(n)."""
    p = Path(out_dir)
    XFeatures = np.load(p / "features.npy")
    XImage = np.load(p / "images.npy")
    icPath = p / "features_ic.npy"
    XIc = np.load(icPath) if icPath.exists() else None
    with open(p / "meta.json") as f:
        meta = json.load(f)
    labels: list[Label] = []
    ys: list[int] = []
    ts: list[float] = []
    sess: list[str] = []
    subj: list[str] = []
    with open(p / "manifest.jsonl") as f:
        for line in f:
            r = json.loads(line)
            sess.append(r.get("session_id", ""))
            subj.append(r.get("subject_id", ""))
            lab = Label()
            lab.class_id = r["class_id"]
            lab.name = r["name"]
            lab.timestamp = r["t"]
            if r["bbox"] is not None:
                lab.bbox = r["bbox"]
            if r["keypoints"]:
                lab.keypoints = r["keypoints"]
            if r.get("mask"):
                lab.mask = r["mask"]
            if r.get("mask_grid"):
                lab.mask_grid = r["mask_grid"]
            labels.append(lab)
            ys.append(r["class_id"])
            ts.append(r["t"])
    return Dataset(
        X_features=XFeatures,
        X_image=XImage,
        y=np.asarray(ys, dtype=np.int64),
        t=np.asarray(ts, dtype=np.float64),
        labels=labels,
        meta=meta,
        session_ids=np.asarray(sess, dtype=object),
        subject_ids=np.asarray(subj, dtype=object),
        X_intercarrier=XIc,
    )
