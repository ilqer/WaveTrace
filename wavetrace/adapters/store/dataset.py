"""Dataset persistence: JSONL manifest + .npy arrays under data/<name>/ (gitignored).

    meta.json       dataset-level: fs, K, K_img, subcarriers, image_subcarriers, window/hop,
                    tolerance, class_names, sync_error, frame_average, subtract_baseline
    manifest.jsonl  one line per sample: i, t, class_id, name, bbox, keypoints, mask, mask_grid, dt
    features.npy    (n, 9K) float32        images.npy  (n, K_img, window) float32
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wavetrace import Label
from wavetrace.domain.groundtruth.dataset import Dataset

FEATURES_FILE_NAME = "features.npy"


@dataclass(frozen=True, slots=True)
class DatasetLabels:
    """A dataset's window classes and LOGO groups, without its feature and image arrays."""

    y: np.ndarray            # (n,) int64
    session_ids: np.ndarray  # (n,) object
    subject_ids: np.ndarray  # (n,) object
    meta: dict


def save_dataset(dataset: "Dataset", out_dir) -> Path:
    """Creates out_dir if missing. O(n)."""
    dataset_dir = Path(out_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / FEATURES_FILE_NAME, dataset.X_features)
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


def _read_manifest(dataset_dir: Path) -> list[dict]:
    with open(dataset_dir / "manifest.jsonl") as json_file:
        return [json.loads(line) for line in json_file]


def _read_meta(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta.json") as json_file:
        return json.load(json_file)


def _labels_from_manifest(records: list[dict], meta: dict) -> DatasetLabels:
    # manifests written before LOGO grouping carry no session_id / subject_id
    return DatasetLabels(y=np.asarray([record["class_id"] for record in records], dtype=np.int64),
                         session_ids=np.asarray([record.get("session_id", "") for record in records], dtype=object),
                         subject_ids=np.asarray([record.get("subject_id", "") for record in records], dtype=object),
                         meta=meta)


def _label_from_record(record: dict) -> Label:
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
    return label


def load_dataset_labels(dataset_dir) -> DatasetLabels:
    """Reads only the manifest and meta.json. O(n)."""
    dataset_dir = Path(dataset_dir)
    return _labels_from_manifest(_read_manifest(dataset_dir), _read_meta(dataset_dir))


def load_dataset(dataset_dir) -> "Dataset":
    dataset_dir = Path(dataset_dir)
    X_features = np.load(dataset_dir / FEATURES_FILE_NAME)
    X_image = np.load(dataset_dir / "images.npy")
    intercarrier_path = dataset_dir / "features_ic.npy"
    X_intercarrier = np.load(intercarrier_path) if intercarrier_path.exists() else None
    records = _read_manifest(dataset_dir)
    window_labels = _labels_from_manifest(records, _read_meta(dataset_dir))
    return Dataset(
        X_features=X_features,
        X_image=X_image,
        y=window_labels.y,
        t=np.asarray([record["t"] for record in records], dtype=np.float64),
        labels=[_label_from_record(record) for record in records],
        meta=window_labels.meta,
        session_ids=window_labels.session_ids,
        subject_ids=window_labels.subject_ids,
        X_intercarrier=X_intercarrier,
    )
