"""Dataset persistence: JSONL manifest + .npy arrays under data/<name>/ (gitignored).

    meta.json       dataset-level: fs, K, K_img, subcarriers, image_subcarriers, window/hop,
                    tolerance, class_names, sync_error, frame_average, subtract_baseline
    manifest.jsonl  one line per sample: i, t, class_id, name, bbox, keypoints, mask, mask_grid, dt
    features.npy    (n, 9K) float32        images.npy  (n, K_img, window) float32
"""

import json
from pathlib import Path

import numpy as np

from wavetrace import Label
from wavetrace.domain.groundtruth.dataset import Dataset


def save_dataset(dataset: "Dataset", out_dir) -> Path:
    """Creates out_dir if missing. O(n)."""
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
