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
from wavetrace.calibration import build_image_baseline
from wavetrace.frontend import group_frames_by_node, iter_windows, iter_stacked_windows
from wavetrace.groundtruth.alignment import align


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
    # intercarrier=True, which requires gain_lock=None.
    X_intercarrier: np.ndarray | None = None


def _attach_labels(window_timestamps, label_source, tolerance):
    """Returns (selected_indices, selected_labels, stats)."""
    if callable(label_source):
        selected_labels = [label_source(window_timestamp) for window_timestamp in window_timestamps]
        selected_indices = list(range(len(window_timestamps)))
        stats = {"mean_dt": 0.0, "max_abs_dt": 0.0, "p95_abs_dt": 0.0,
                 "matched": len(window_timestamps), "dropped": 0}
    else:
        alignment = align(window_timestamps, label_source, tolerance)
        selected_indices = [window_index for window_index, _ in alignment.matched]
        selected_labels = [label for _, label in alignment.matched]
        stats = alignment.stats
    return selected_indices, selected_labels, stats


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
    """`label_source` is either a list[Label] - camera or replay, nearest-matched to each window with
    the sync error measured - or a callable t->Label on the same CSI clock, evaluated at every
    window timestamp and never dropped.

    `gain_lock` = the locked GainLock from `calibration`, OR None to skip the per-frame amplitude
    rescale. Pass None for material and weapon datasets: the gain lock normalizes every frame to a
    common mean, which erases the bulk attenuation σ²[p] and reflection_signature measure. It is
    for the amplitude and presence path only.

    session_id / subject_id: group ids of this recording, stamped on every sample; the
      leave-one-group-out evaluation folds on them.
    intercarrier: also emit the (n, 27) inter-carrier block from RAW magnitudes.
    frame_average: M >= 1 non-overlapping decimating mean; 1 leaves the rate alone.
    subtract_baseline: subtract the quiet-room baseline from the image path only.
    subtract_ic_baseline: subtract the raw quiet-room baseline from the inter-carrier path, so the
      weapon's σ²[p] sees the perturbation rather than the whole static room. Independent of
      subtract_baseline."""
    subcarrier_indices = np.asarray(calibration_result.subcarriers, dtype=np.intp)
    num_feature_subcarriers = int(subcarrier_indices.size)
    image_subcarrier_indices = (getattr(calibration_result, "image_subcarriers", None)
                                or list(calibration_result.subcarriers))
    num_image_subcarriers = len(image_subcarrier_indices)

    image_baseline = None
    if subtract_baseline:
        image_baseline = build_image_baseline(calibration_result, locked=(gain_lock is not None))
    # IC background subtraction uses the RAW baseline (the IC path is always raw, gain_lock=None for
    # weapon), so no locked-basis rescale — just the quiet-room mean |H| per subcarrier.
    intercarrier_baseline = (np.asarray(calibration_result.baseline_mag, dtype=np.float32)
                             if (subtract_ic_baseline and intercarrier) else None)

    # materialize once: iter_windows consumes the stream, and the fs estimate below re-indexes
    # frames[-1], which a bare generator would have exhausted by then.
    frames = list(frames)

    features_per_window: list[np.ndarray] = []
    images_per_window: list[np.ndarray] = []
    intercarrier_features_per_window: list[np.ndarray] = []
    window_timestamps: list[float] = []
    for window_timestamp, features, image, intercarrier_features in iter_windows(
        frames, subcarrier_indices, gain_lock, window=window, hop=hop, intercarrier=intercarrier,
        image_subcarriers=(image_subcarrier_indices
                           if image_subcarrier_indices != list(calibration_result.subcarriers) else None),
        frame_average=frame_average,
        image_baseline=image_baseline,
        intercarrier_baseline=intercarrier_baseline,
    ):
        features_per_window.append(features.copy())
        images_per_window.append(image.copy())
        if intercarrier_features is not None:
            intercarrier_features_per_window.append(intercarrier_features.copy())
        window_timestamps.append(window_timestamp)

    selected_indices, selected_labels, stats = _attach_labels(window_timestamps, label_source, tolerance)

    if selected_indices:
        X_features = np.stack([features_per_window[i] for i in selected_indices]).astype(np.float32)
        X_image = np.stack([images_per_window[i] for i in selected_indices]).astype(np.float32)
    else:
        X_features = np.empty((0, 9 * num_feature_subcarriers), np.float32)
        X_image = np.empty((0, num_image_subcarriers, window), np.float32)
    y = np.asarray([label.class_id for label in selected_labels], dtype=np.int64)
    selected_window_timestamps = np.asarray([window_timestamps[i] for i in selected_indices],
                                            dtype=np.float64)

    # measure fs from the frame timestamps; the configured packet rate is not what arrives
    sample_rate_hz = ((len(frames) - 1) / (frames[-1].timestamp - frames[0].timestamp)
                      if len(frames) > 1 and frames[-1].timestamp > frames[0].timestamp else 0.0)
    meta = {
        "fs": float(sample_rate_hz),
        "K": num_feature_subcarriers,
        "K_img": num_image_subcarriers,
        # raw capture width (64/128/192 by band), not K, which is the NBVI-selected subset of it
        "num_subcarriers": int(frames[0].num_subcarriers),
        "subcarriers": [int(subcarrier_index) for subcarrier_index in calibration_result.subcarriers],
        "image_subcarriers": image_subcarrier_indices,
        "subtract_ic_baseline": bool(intercarrier_baseline is not None),
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
    X_intercarrier = None
    if intercarrier:
        intercarrier_width = InterCarrierExtractor(window=window, hop=hop).output_size
        X_intercarrier = (
            np.stack([intercarrier_features_per_window[i] for i in selected_indices]).astype(np.float32)
            if selected_indices else np.empty((0, intercarrier_width), np.float32)
        )
    return Dataset(
        X_features=X_features, X_image=X_image, y=y, t=selected_window_timestamps, labels=selected_labels,
        meta=meta,
        session_ids=np.full(y.size, str(session_id), dtype=object),
        subject_ids=np.full(y.size, str(subject_id), dtype=object),
        X_intercarrier=X_intercarrier,
    )


def build_stacked_dataset(
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
    """Multi-node variant of build_dataset: nodes are stacked as channels.

    calibrations: dict[node_id -> (CalibrationResult, GainLock|None)].
    Frames from all nodes are demuxed by node_id and fed through iter_stacked_windows.
    Shapes: X_features (n, N·9·K), X_image (n, N, K_img, window), X_intercarrier (n, N·27).
    """
    frames = list(frames)
    per_node_frames = group_frames_by_node(frames)

    node_ids = sorted(calibrations.keys())
    per_node_calibration = {}
    for node_id in node_ids:
        node_calibration, gain_lock = calibrations[node_id]
        node_image_subcarrier_indices = (getattr(node_calibration, "image_subcarriers", None)
                                         or list(node_calibration.subcarriers))
        node_image_baseline = (build_image_baseline(node_calibration, locked=(gain_lock is not None))
                               if subtract_baseline else None)
        per_node_calibration[node_id] = (list(node_calibration.subcarriers), node_image_subcarrier_indices,
                                         gain_lock, node_image_baseline)

    # all nodes share these widths; iter_stacked_windows has already checked that
    first_calibration = calibrations[node_ids[0]][0]
    num_feature_subcarriers = len(first_calibration.subcarriers)
    image_subcarrier_indices = (getattr(first_calibration, "image_subcarriers", None)
                                or list(first_calibration.subcarriers))
    num_image_subcarriers = len(image_subcarrier_indices)
    num_nodes = len(node_ids)

    features_per_window: list[np.ndarray] = []
    images_per_window: list[np.ndarray] = []
    intercarrier_features_per_window: list[np.ndarray] = []
    window_timestamps: list[float] = []
    for window_timestamp, features, image, intercarrier_features in iter_stacked_windows(
        per_node_frames, per_node_calibration, window=window, hop=hop, intercarrier=intercarrier,
        frame_average=frame_average, node_tolerance=node_tolerance,
    ):
        features_per_window.append(features.copy())
        images_per_window.append(image.copy())
        if intercarrier_features is not None:
            intercarrier_features_per_window.append(intercarrier_features.copy())
        window_timestamps.append(window_timestamp)

    selected_indices, selected_labels, stats = _attach_labels(window_timestamps, label_source, tolerance)

    if selected_indices:
        X_features = np.stack([features_per_window[i] for i in selected_indices]).astype(np.float32)
        X_image = np.stack([images_per_window[i] for i in selected_indices]).astype(np.float32)
    else:
        X_features = np.empty((0, num_nodes * 9 * num_feature_subcarriers), np.float32)
        X_image = np.empty((0, num_nodes, num_image_subcarriers, window), np.float32)
    y = np.asarray([label.class_id for label in selected_labels], dtype=np.int64)
    selected_window_timestamps = np.asarray([window_timestamps[i] for i in selected_indices],
                                            dtype=np.float64)

    # fs from the lowest node id's frames
    first_node_frames = per_node_frames.get(node_ids[0], [])
    sample_rate_hz = ((len(first_node_frames) - 1)
                      / (first_node_frames[-1].timestamp - first_node_frames[0].timestamp)
                      if len(first_node_frames) > 1
                      and first_node_frames[-1].timestamp > first_node_frames[0].timestamp
                      else 0.0)

    meta = {
        "fs": float(sample_rate_hz),
        "K": num_feature_subcarriers,
        "K_img": num_image_subcarriers,
        # raw per-frame capture width of node0's link, same meaning as build_dataset's; empty
        # first_node_frames (degenerate input) falls back to K rather than indexing an empty list.
        "num_subcarriers": (int(first_node_frames[0].num_subcarriers) if first_node_frames
                            else num_feature_subcarriers),
        "subcarriers": [int(subcarrier_index) for subcarrier_index in first_calibration.subcarriers],
        "image_subcarriers": image_subcarrier_indices,
        "window": window,
        "hop": hop,
        "tolerance": tolerance,
        "node_ids": node_ids,
        "num_nodes": num_nodes,
        "node_tolerance": node_tolerance,
        "gain_locked": any(calibrations[node_id][1] is not None for node_id in node_ids),
        "class_names": dict(class_names) if class_names else {},
        "sync_error": {"mean_dt": stats["mean_dt"], "max_abs_dt": stats["max_abs_dt"],
                       "p95_abs_dt": stats["p95_abs_dt"]},
        "n_samples": int(y.size),
        "n_dropped": int(stats["dropped"]),
        "intercarrier": bool(intercarrier),
        "frame_average": int(frame_average),
        "subtract_baseline": bool(subtract_baseline),
    }
    X_intercarrier = None
    if intercarrier:
        intercarrier_width = num_nodes * 27
        X_intercarrier = (
            np.stack([intercarrier_features_per_window[i] for i in selected_indices]).astype(np.float32)
            if selected_indices else np.empty((0, intercarrier_width), np.float32)
        )
    return Dataset(
        X_features=X_features, X_image=X_image, y=y, t=selected_window_timestamps, labels=selected_labels,
        meta=meta,
        session_ids=np.full(y.size, str(session_id), dtype=object),
        subject_ids=np.full(y.size, str(subject_id), dtype=object),
        X_intercarrier=X_intercarrier,
    )


def save_dataset(dataset: "Dataset", out_dir) -> Path:
    """Serialize to JSONL manifest + .npy arrays under out_dir (created if missing). O(n)."""
    dataset_dir = Path(out_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "features.npy", dataset.X_features)
    np.save(dataset_dir / "images.npy", dataset.X_image)
    if dataset.X_intercarrier is not None:
        np.save(dataset_dir / "features_ic.npy", dataset.X_intercarrier)
    with open(dataset_dir / "manifest.jsonl", "w") as json_file:
        for i, label in enumerate(dataset.labels):
            record = {
                "i": i,
                "t": float(dataset.t[i]),
                "class_id": int(dataset.y[i]),
                "name": label.name,
                "bbox": list(label.bbox) if label.bbox is not None else None,
                "keypoints": list(label.keypoints),
                # heatmap target: the camera mask pooled to a G×G grid, set only by a SegmentationLabeler
                "mask": [float(mask_value) for mask_value in label.mask] if label.mask else None,
                "mask_grid": int(label.mask_grid) if label.mask_grid else None,
                "session_id": str(dataset.session_ids[i]) if dataset.session_ids.size else "",
                "subject_id": str(dataset.subject_ids[i]) if dataset.subject_ids.size else "",
            }
            json_file.write(json.dumps(record) + "\n")
    with open(dataset_dir / "meta.json", "w") as json_file:
        json.dump(dataset.meta, json_file, indent=2)
    return dataset_dir


def load_dataset(dataset_dir) -> "Dataset":
    dataset_dir = Path(dataset_dir)
    X_features = np.load(dataset_dir / "features.npy")
    X_image = np.load(dataset_dir / "images.npy")
    intercarrier_path = dataset_dir / "features_ic.npy"
    X_intercarrier = np.load(intercarrier_path) if intercarrier_path.exists() else None
    with open(dataset_dir / "meta.json") as json_file:
        meta = json.load(json_file)
    labels: list[Label] = []
    class_ids: list[int] = []
    timestamps: list[float] = []
    session_ids: list[str] = []
    subject_ids: list[str] = []
    with open(dataset_dir / "manifest.jsonl") as json_file:
        for line in json_file:
            record = json.loads(line)
            session_ids.append(record.get("session_id", ""))
            subject_ids.append(record.get("subject_id", ""))
            label = Label()
            label.class_id = record["class_id"]
            label.name = record["name"]
            label.timestamp = record["t"]
            if record["bbox"] is not None:
                label.bbox = record["bbox"]
            if record["keypoints"]:
                label.keypoints = record["keypoints"]
            if record.get("mask"):
                label.mask = record["mask"]
            if record.get("mask_grid"):
                label.mask_grid = record["mask_grid"]
            labels.append(label)
            class_ids.append(record["class_id"])
            timestamps.append(record["t"])
    return Dataset(
        X_features=X_features,
        X_image=X_image,
        y=np.asarray(class_ids, dtype=np.int64),
        t=np.asarray(timestamps, dtype=np.float64),
        labels=labels,
        meta=meta,
        session_ids=np.asarray(session_ids, dtype=object),
        subject_ids=np.asarray(subject_ids, dtype=object),
        X_intercarrier=X_intercarrier,
    )
