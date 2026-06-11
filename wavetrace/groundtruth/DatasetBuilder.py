"""Phase 5c — drive the shared P4 front-end over a CSI recording, attach aligned labels, emit and
serialize a labeled dataset {(x_t, label_t)} (OFFLINE — correctness + sync, not Big-O).

Both recognition inputs are stored (MINDMAP "Recognition input contract") so Phase 6/7 picks
MLP/SVM or heatmap-CNN with no re-run:
  * X_features (n, 9·K) — §2.9 nine features per NBVI subcarrier
  * X_image    (n, K, window) — selected-subcarrier × time CSI "image" (the internal heatmap the
    weapon head collapses to a yes/no label)

Labels stay binary (A presence / E weapon) via the labeler's `label_fn`, but the raw box/keypoints
and any weapon position ride along on the Label and into the manifest, so the location/heatmap work
needs no re-run.

Serialization = JSONL manifest + .npy arrays under data/<name>/ (gitignored):
    meta.json       dataset-level: fs, K, subcarriers, window/hop, tolerance, class_names, sync_error
    manifest.jsonl  one line per sample: i, t, class_id, name, bbox, keypoints, dt
    features.npy    (n, 9K) float32        images.npy  (n, K, window) float32
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from wavetrace import InterCarrierExtractor, Label
from wavetrace.Frontend import iter_windows
from wavetrace.groundtruth.Align import align


@dataclass
class Dataset:
    X_features: np.ndarray            # (n, 9K) float32
    X_image: np.ndarray              # (n, K, window) float32
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
) -> Dataset:
    """Build a labeled dataset from a CSI recording + a label source.

    `label_source` is either a list[Label] (camera/replay — nearest-match Aligned to windows + sync
    error measured) OR a callable t->Label (scripted / location-chip — same CSI clock, evaluated at
    each window timestamp, no drop).

    `gain_lock` = the locked GainLock from `calibration`, OR None to skip the per-frame amplitude
    rescale. Pass None for material / weapon datasets (σ²[p], reflection_signature): gain lock
    normalizes every frame to a common mean, erasing the bulk attenuation those features measure.
    Use it only for the amplitude / presence feature path.

    session_id / subject_id (P6a): group ids of THIS recording, stamped on every sample — the
    leave-one-session-out / leave-one-subject-out eval gate folds on them (rev-7 #1: never a random
    within-session split). One recording = one (session, subject); concat datasets to mix groups.

    intercarrier (P7p-a): also emit the (n, 27) `InterCarrierExtractor` block over ALL subcarriers —
    the σ²[p] weapon-head input. The IC block ALWAYS sees raw (pre-lock) magnitudes (gain lock
    cancels the cross-subcarrier flatness the metal signature lives in). When gain_lock is also set,
    a DUAL-BLOCK dataset is produced: IC from raw mags + X_features from locked mags — fusion-ready."""
    subc = np.asarray(calibration_result.subcarriers, dtype=np.intp)
    K = int(subc.size)

    feats: list[np.ndarray] = []
    imgs: list[np.ndarray] = []
    ics: list[np.ndarray] = []
    wts: list[float] = []
    # shared front-end (wavetrace.Frontend) — the SAME emit loop Cli.run serves on, so the trained
    # model sees identical features. iter_windows yields reused buffers -> copy each before advancing.
    for t, features, image, ic in iter_windows(
        frames, subc, gain_lock, window=window, hop=hop, intercarrier=intercarrier
    ):
        feats.append(features.copy())
        imgs.append(image.copy())
        if ic is not None:
            ics.append(ic.copy())
        wts.append(t)                          # window timestamp = END frame (Q5)

    # ----- attach labels --------------------------------------------------------------------------
    if callable(label_source):  # time-style: scripted / location-chip share the CSI clock
        sel_labels = [label_source(t) for t in wts]
        sel = list(range(len(wts)))
        stats = {"mean_dt": 0.0, "max_abs_dt": 0.0, "p95_abs_dt": 0.0,
                 "matched": len(wts), "dropped": 0}
    else:                       # camera/replay: nearest-match within tolerance + measure sync error
        res = align(wts, label_source, tolerance)
        sel = [wi for wi, _ in res.matched]
        sel_labels = [lab for _, lab in res.matched]
        stats = res.stats

    if sel:
        X_features = np.stack([feats[i] for i in sel]).astype(np.float32)
        X_image = np.stack([imgs[i] for i in sel]).astype(np.float32)
    else:
        X_features = np.empty((0, 9 * K), np.float32)
        X_image = np.empty((0, K, window), np.float32)
    y = np.asarray([l.class_id for l in sel_labels], dtype=np.int64)
    t = np.asarray([wts[i] for i in sel], dtype=np.float64)

    # fs estimated live from CsiFrame timestamps (never assume packet rate, REFERENCE §4)
    fs = ((len(frames) - 1) / (frames[-1].timestamp - frames[0].timestamp)
          if len(frames) > 1 and frames[-1].timestamp > frames[0].timestamp else 0.0)
    meta = {
        "fs": float(fs),
        "K": K,
        "subcarriers": [int(s) for s in calibration_result.subcarriers],
        "window": window,
        "hop": hop,
        "tolerance": tolerance,
        "gain_locked": gain_lock is not None,  # amplitude basis: rescaled to the locked ref, or raw
        "class_names": dict(class_names) if class_names else {},
        "sync_error": {"mean_dt": stats["mean_dt"], "max_abs_dt": stats["max_abs_dt"],
                       "p95_abs_dt": stats["p95_abs_dt"]},
        "n_samples": int(y.size),
        "n_dropped": int(stats["dropped"]),
        "intercarrier": bool(intercarrier),
    }
    X_ic = None
    if intercarrier:
        ic_width = InterCarrierExtractor(window=window, hop=hop).output_size  # fixed (µ|σ²|CV × 9)
        X_ic = (np.stack([ics[i] for i in sel]).astype(np.float32) if sel
                else np.empty((0, ic_width), np.float32))
    return Dataset(
        X_features=X_features, X_image=X_image, y=y, t=t, labels=sel_labels, meta=meta,
        session_ids=np.full(y.size, str(session_id), dtype=object),
        subject_ids=np.full(y.size, str(subject_id), dtype=object),
        X_intercarrier=X_ic,
    )


def save_dataset(dataset: Dataset, out_dir) -> Path:
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
                "session_id": str(dataset.session_ids[i]) if dataset.session_ids.size else "",
                "subject_id": str(dataset.subject_ids[i]) if dataset.subject_ids.size else "",
            }
            f.write(json.dumps(rec) + "\n")
    with open(p / "meta.json", "w") as f:
        json.dump(dataset.meta, f, indent=2)
    return p


def load_dataset(out_dir) -> Dataset:
    """Round-trip load of a saved dataset. O(n)."""
    p = Path(out_dir)
    X_features = np.load(p / "features.npy")
    X_image = np.load(p / "images.npy")
    ic_path = p / "features_ic.npy"
    X_ic = np.load(ic_path) if ic_path.exists() else None  # absent in pre-P7 datasets
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
            sess.append(r.get("session_id", ""))   # absent in pre-P6 manifests -> ""
            subj.append(r.get("subject_id", ""))
            lab = Label()
            lab.class_id = r["class_id"]
            lab.name = r["name"]
            lab.timestamp = r["t"]
            if r["bbox"] is not None:
                lab.bbox = r["bbox"]
            if r["keypoints"]:
                lab.keypoints = r["keypoints"]
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
