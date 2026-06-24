"""Phase 5c — drive the shared P4 front-end over a CSI recording, attach aligned labels, emit and
serialize a labeled dataset {(x_t, label_t)} (OFFLINE — correctness + sync, not Big-O).

Both recognition inputs are stored (MINDMAP "Recognition input contract") so Phase 6/7 picks
MLP/SVM or heatmap-CNN with no re-run:
  * X_features (n, 9·K) — §2.9 nine features per NBVI subcarrier
  * X_image    (n, K_img, window) — selected-subcarrier × time CSI "image" (T1/P10: K_img is the
    full noise-gate-passing set when calibration has image_subcarriers; fallback = NBVI K)

Labels stay binary (A presence / E weapon) via the labeler's `label_fn`, but the raw box/keypoints
and any weapon position ride along on the Label and into the manifest, so the location/heatmap work
needs no re-run.

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
from wavetrace.Calibration import image_baseline as _image_baseline
from wavetrace.Frontend import demux_by_node, iter_windows, iter_windows_stacked
from wavetrace.groundtruth.Align import align


@dataclass
class Dataset:
    X_features: np.ndarray            # (n, 9K) float32
    X_image: np.ndarray              # (n, K_img, window) float32
    y: np.ndarray                    # (n,) int64 stage class
    t: np.ndarray                    # (n,) float64 window-END timestamps
    labels: list[Label]             # full Labels (box/keypoints/position preserved)
    meta: dict = field(default_factory=dict)
    # P6a group ids (one per sample) — the leave-one-session/subject-out eval gate folds on these.
    session_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    subject_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    # P7p-a: (n, 27) inter-carrier block (µ|σ²|CV × 9) from RAW magnitudes — the σ²[p] weapon-head
    # input. None unless built with intercarrier=True (which requires gain_lock=None).
    X_intercarrier: np.ndarray | None = None


def _attach_labels(wts, label_source, tolerance):
    """Attach aligned or callable labels to a list of window timestamps. Returns (sel, sel_labels, stats)."""
    if callable(label_source):
        sel_labels = [label_source(t) for t in wts]
        sel = list(range(len(wts)))
        stats = {"mean_dt": 0.0, "max_abs_dt": 0.0, "p95_abs_dt": 0.0,
                 "matched": len(wts), "dropped": 0}
    else:
        res = align(wts, label_source, tolerance)
        sel = [wi for wi, _ in res.matched]
        sel_labels = [lab for _, lab in res.matched]
        stats = res.stats
    return sel, sel_labels, stats


def build_dataset(
    frames,
    calibration_result,
    gain_lock,
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

    `label_source` is either a list[Label] (camera/replay — nearest-match Aligned to windows + sync
    error measured) OR a callable t->Label (scripted / location-chip — same CSI clock, evaluated at
    each window timestamp, no drop).

    `gain_lock` = the locked GainLock from `calibration`, OR None to skip the per-frame amplitude
    rescale. Pass None for material / weapon datasets (σ²[p], reflection_signature): gain lock
    normalizes every frame to a common mean, erasing the bulk attenuation those features measure.
    Use it only for the amplitude / presence feature path.

    session_id / subject_id (P6a): group ids of THIS recording, stamped on every sample.
    intercarrier (P7p-a): also emit the (n, 27) IC block from RAW magnitudes.
    frame_average (T2/P10): non-overlapping decimating mean (M=1 = no change).
    subtract_baseline (T3/P10): subtract the quiet-room baseline from the image path only.
    subtract_ic_baseline (Item 10/CAUSE 2B): subtract the raw quiet-room baseline from the IC path
      (weapon σ²[p] background subtraction). Independent of subtract_baseline (image path)."""
    subc = np.asarray(calibration_result.subcarriers, dtype=np.intp)
    K = int(subc.size)
    img_subc_list = getattr(calibration_result, "image_subcarriers", None) or list(calibration_result.subcarriers)
    K_img = len(img_subc_list)

    img_baseline_arr = None
    if subtract_baseline:
        img_baseline_arr = _image_baseline(calibration_result, locked=(gain_lock is not None))
    # IC background subtraction uses the RAW baseline (the IC path is always raw, gain_lock=None for
    # weapon), so no locked-basis rescale — just the quiet-room mean |H| per subcarrier.
    ic_baseline_arr = (np.asarray(calibration_result.baseline_mag, dtype=np.float32)
                       if (subtract_ic_baseline and intercarrier) else None)

    # materialize once: iter_windows consumes the stream, and the fs estimate below re-indexes
    # frames[-1] — a bare generator would be exhausted by then (B5).
    frames = list(frames)

    feats: list[np.ndarray] = []
    imgs: list[np.ndarray] = []
    ics: list[np.ndarray] = []
    wts: list[float] = []
    for t, features, image, ic in iter_windows(
        frames, subc, gain_lock, window=window, hop=hop, intercarrier=intercarrier,
        image_subcarriers=(img_subc_list if img_subc_list != list(calibration_result.subcarriers) else None),
        frame_average=frame_average,
        image_baseline=img_baseline_arr,
        ic_baseline=ic_baseline_arr,
    ):
        feats.append(features.copy())
        imgs.append(image.copy())
        if ic is not None:
            ics.append(ic.copy())
        wts.append(t)

    # ----- attach labels --------------------------------------------------------------------------
    sel, sel_labels, stats = _attach_labels(wts, label_source, tolerance)

    if sel:
        X_features = np.stack([feats[i] for i in sel]).astype(np.float32)
        X_image = np.stack([imgs[i] for i in sel]).astype(np.float32)
    else:
        X_features = np.empty((0, 9 * K), np.float32)
        X_image = np.empty((0, K_img, window), np.float32)
    y = np.asarray([l.class_id for l in sel_labels], dtype=np.int64)
    t_arr = np.asarray([wts[i] for i in sel], dtype=np.float64)

    # fs estimated live from CsiFrame timestamps (never assume packet rate, REFERENCE §4)
    fs = ((len(frames) - 1) / (frames[-1].timestamp - frames[0].timestamp)
          if len(frames) > 1 and frames[-1].timestamp > frames[0].timestamp else 0.0)
    meta = {
        "fs": float(fs),
        "K": K,
        "K_img": K_img,
        "subcarriers": [int(s) for s in calibration_result.subcarriers],
        "image_subcarriers": img_subc_list,
        "subtract_ic_baseline": bool(ic_baseline_arr is not None),
        "window": window,
        "hop": hop,
        "tolerance": tolerance,
        "gain_locked": gain_lock is not None,
        "class_names": dict(class_names) if class_names else {},
        "sync_error": {"mean_dt": stats["mean_dt"], "max_abs_dt": stats["max_abs_dt"],
                       "p95_abs_dt": stats["p95_abs_dt"]},
        "n_samples": int(y.size),
        "n_dropped": int(stats["dropped"]),
        "intercarrier": bool(intercarrier),
        "frame_average": int(frame_average),
        "subtract_baseline": bool(subtract_baseline),
    }
    X_ic = None
    if intercarrier:
        ic_width = InterCarrierExtractor(window=window, hop=hop).output_size
        X_ic = (np.stack([ics[i] for i in sel]).astype(np.float32) if sel
                else np.empty((0, ic_width), np.float32))
    return Dataset(
        X_features=X_features, X_image=X_image, y=y, t=t_arr, labels=sel_labels, meta=meta,
        session_ids=np.full(y.size, str(session_id), dtype=object),
        subject_ids=np.full(y.size, str(subject_id), dtype=object),
        X_intercarrier=X_ic,
    )


def build_dataset_stacked(
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
    Frames from all nodes are demuxed by node_id and fed through iter_windows_stacked.
    Shapes: X_features (n, N·9·K), X_image (n, N, K_img, window), X_intercarrier (n, N·27).
    """
    frames = list(frames)
    by_node = demux_by_node(frames)

    node_ids = sorted(calibrations.keys())
    per_node_calib = {}
    for nid in node_ids:
        cal_result, gain_lock = calibrations[nid]
        img_subc = getattr(cal_result, "image_subcarriers", None) or list(cal_result.subcarriers)
        base = _image_baseline(cal_result, locked=(gain_lock is not None)) if subtract_baseline else None
        per_node_calib[nid] = (list(cal_result.subcarriers), img_subc, gain_lock, base)

    # K and K_img from lowest node id (all nodes must match — validated by iter_windows_stacked)
    first_cal = calibrations[node_ids[0]][0]
    K = len(first_cal.subcarriers)
    img_subc_list = getattr(first_cal, "image_subcarriers", None) or list(first_cal.subcarriers)
    K_img = len(img_subc_list)
    N = len(node_ids)

    feats: list[np.ndarray] = []
    imgs: list[np.ndarray] = []
    ics: list[np.ndarray] = []
    wts: list[float] = []
    for t, feat, image, ic in iter_windows_stacked(
        by_node, per_node_calib, window=window, hop=hop, intercarrier=intercarrier,
        frame_average=frame_average, node_tolerance=node_tolerance,
    ):
        feats.append(feat.copy())
        imgs.append(image.copy())
        if ic is not None:
            ics.append(ic.copy())
        wts.append(t)

    sel, sel_labels, stats = _attach_labels(wts, label_source, tolerance)

    if sel:
        X_features = np.stack([feats[i] for i in sel]).astype(np.float32)
        X_image = np.stack([imgs[i] for i in sel]).astype(np.float32)
    else:
        X_features = np.empty((0, N * 9 * K), np.float32)
        X_image = np.empty((0, N, K_img, window), np.float32)
    y = np.asarray([l.class_id for l in sel_labels], dtype=np.int64)
    t_arr = np.asarray([wts[i] for i in sel], dtype=np.float64)

    # fs from the lowest node id's frames
    node0_frames = by_node.get(node_ids[0], [])
    fs = ((len(node0_frames) - 1) / (node0_frames[-1].timestamp - node0_frames[0].timestamp)
          if len(node0_frames) > 1 and node0_frames[-1].timestamp > node0_frames[0].timestamp
          else 0.0)

    meta = {
        "fs": float(fs),
        "K": K,
        "K_img": K_img,
        "subcarriers": [int(s) for s in first_cal.subcarriers],
        "image_subcarriers": img_subc_list,
        "window": window,
        "hop": hop,
        "tolerance": tolerance,
        "node_ids": node_ids,
        "num_nodes": N,
        "node_tolerance": node_tolerance,
        "gain_locked": any(calibrations[nid][1] is not None for nid in node_ids),
        "class_names": dict(class_names) if class_names else {},
        "sync_error": {"mean_dt": stats["mean_dt"], "max_abs_dt": stats["max_abs_dt"],
                       "p95_abs_dt": stats["p95_abs_dt"]},
        "n_samples": int(y.size),
        "n_dropped": int(stats["dropped"]),
        "intercarrier": bool(intercarrier),
        "frame_average": int(frame_average),
        "subtract_baseline": bool(subtract_baseline),
    }
    X_ic = None
    if intercarrier:
        ic_width = N * 27
        X_ic = (np.stack([ics[i] for i in sel]).astype(np.float32) if sel
                else np.empty((0, ic_width), np.float32))
    return Dataset(
        X_features=X_features, X_image=X_image, y=y, t=t_arr, labels=sel_labels, meta=meta,
        session_ids=np.full(y.size, str(session_id), dtype=object),
        subject_ids=np.full(y.size, str(subject_id), dtype=object),
        X_intercarrier=X_ic,
    )


def save_dataset(dataset: "Dataset", out_dir) -> Path:
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
                # heatmap target: the camera mask pooled to a G×G grid (None unless a SegmentationLabeler set it)
                "mask": [float(v) for v in lab.mask] if lab.mask else None,
                "mask_grid": int(lab.mask_grid) if lab.mask_grid else None,
                "session_id": str(dataset.session_ids[i]) if dataset.session_ids.size else "",
                "subject_id": str(dataset.subject_ids[i]) if dataset.subject_ids.size else "",
            }
            f.write(json.dumps(rec) + "\n")
    with open(p / "meta.json", "w") as f:
        json.dump(dataset.meta, f, indent=2)
    return p


def load_dataset(out_dir) -> "Dataset":
    """Round-trip load of a saved dataset. O(n)."""
    p = Path(out_dir)
    X_features = np.load(p / "features.npy")
    X_image = np.load(p / "images.npy")
    ic_path = p / "features_ic.npy"
    X_ic = np.load(ic_path) if ic_path.exists() else None
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
        X_features=X_features,
        X_image=X_image,
        y=np.asarray(ys, dtype=np.int64),
        t=np.asarray(ts, dtype=np.float64),
        labels=labels,
        meta=meta,
        session_ids=np.asarray(sess, dtype=object),
        subject_ids=np.asarray(subj, dtype=object),
        X_intercarrier=X_ic,
    )
