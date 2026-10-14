"""Train-set accuracy is a sanity check only. The number that counts is leave_one_group_out, held out
over sessions and subjects.
"""

from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np

from wavetrace.adapters.recognition import DEFAULT_BACKEND_BY_FEATURE_MODE
from wavetrace.adapters.recognition.heads import build_presence_head, build_weapon_head
from wavetrace.adapters.store.dataset import load_dataset
from wavetrace.domain.groundtruth.dataset import Dataset
from wavetrace.domain.model_config import ModelConfig
from wavetrace.domain.recognition import PresenceHead, WeaponHead
from wavetrace.recognition.evaluate import leave_one_group_out


def _carry_groups(session_ids):
    """Carry position per window, parsed from session ids shaped `<...>_<carry>_s<digits>`; None when
    any id does not match. Folding on carry pose exposes a head that learned the pose, not the weapon."""
    carry_positions = []
    for session_id in session_ids:
        parts = str(session_id).split("_")
        if len(parts) < 3 or parts[-1][:1] != "s" or not parts[-1][1:].isdigit():
            return None
        carry_positions.append(parts[-2])
    return np.asarray(carry_positions)


def _logo_metrics(X, y, session_ids, subject_ids, make_head) -> dict:
    """LOGO accuracy over sessions and subjects (and carry position). O(folds*fit)."""
    metrics: dict = {}
    axes = [("session", session_ids), ("subject", subject_ids)]
    carry_positions = _carry_groups(session_ids)
    if carry_positions is not None:
        axes.append(("carry", carry_positions))  # weapon only: catches a head that learned the carry pose
    for axis, groups in axes:
        if np.unique(groups).size >= 2:
            report = leave_one_group_out(X, y, groups, make_head)
            metrics[axis] = {key: report[key] for key in ("accuracy", "majority_accuracy") if key in report}
            metrics[axis].update({key: report[key] for key in ("tpr", "fp_rate") if key in report})
            if "confusion" in report:
                metrics[axis]["confusion"] = np.asarray(report["confusion"]).tolist()
    return metrics


def concat_datasets(datasets) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    recordings = list(datasets)
    if not recordings:
        raise ValueError("concat_datasets: no datasets")
    feature_widths = {recording.X_features.shape[1] for recording in recordings}
    if len(feature_widths) != 1:
        raise ValueError(f"concat_datasets: feature dims differ across datasets: {sorted(feature_widths)}")
    X = np.concatenate([recording.X_features for recording in recordings]).astype(np.float32)
    y = np.concatenate([recording.y for recording in recordings])
    session_ids = np.concatenate([recording.session_ids for recording in recordings])
    subject_ids = np.concatenate([recording.subject_ids for recording in recordings])
    return X, y, session_ids, subject_ids


def concat_arrays(datasets, field_name: str) -> np.ndarray:
    field_values = []
    for i, dataset in enumerate(datasets):
        field_value = getattr(dataset, field_name)
        if field_value is None:
            raise ValueError(f"concat_arrays: dataset {i} has no {field_name} (rebuild with it enabled)")
        field_values.append(field_value)
    if not field_values:
        raise ValueError("concat_arrays: no datasets")
    return np.concatenate(field_values)


def train_presence(
    dataset_dirs,
    out_dir="models/presence",
    config: ModelConfig | None = None,
) -> tuple[PresenceHead, dict]:
    """Writes `model.joblib` and `metrics.json` into out_dir."""
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [dataset_dirs]
    loaded: list[Dataset] = [load_dataset(dataset_dir) for dataset_dir in dataset_dirs]
    X, y, session_ids, subject_ids = concat_datasets(loaded)
    metadata = loaded[0].meta

    if config is None:
        # window/hop come from the dataset, so serving frames windows exactly as training did
        config = ModelConfig(stage="presence", k=int(metadata["K"]),
                             window=int(metadata["window"]), hop=int(metadata["hop"]),
                             frame_average=int(metadata.get("frame_average", 1)),
                             subtract_baseline=bool(metadata.get("subtract_baseline", False)))

    fit_start_s = time.perf_counter()
    head = build_presence_head(config).fit(X, y)
    fit_duration_s = time.perf_counter() - fit_start_s
    # the raw capture width, never the NBVI-selected k; None for datasets saved before
    # `num_subcarriers` was recorded, so the two can never be confused
    if "num_subcarriers" in metadata:
        head.contract = replace(head.contract, subcarrier_width=int(metadata["num_subcarriers"]))

    classes, counts = np.unique(y, return_counts=True)
    metrics = {
        "stage": config.stage,
        "backend": config.backend,
        "k": config.k,
        "n_samples": int(y.size),
        "n_features": int(X.shape[1]),
        "class_counts": {  # str keys: JSON-stable
            str(int(class_label)): int(sample_count) for class_label, sample_count in zip(classes, counts)
        },
        "sessions": sorted({str(group_id) for group_id in session_ids}),
        "subjects": sorted({str(group_id) for group_id in subject_ids}),
        "train_accuracy": float((head.predict(X) == y).mean()),
        "logo": _logo_metrics(X, y, session_ids, subject_ids, lambda: build_presence_head(config)),
        "fit_seconds": round(fit_duration_s, 3),
    }
    model_dir = Path(out_dir)
    head.save(model_dir / "model.joblib")
    with open(model_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    return head, metrics


def train_weapon(
    dataset_dirs,
    out_dir="models/weapon",
    config: ModelConfig | None = None,
    feature_mode: str = "ic27",
    report=None,
) -> tuple[WeaponHead, dict]:
    """Writes `model.joblib` and `metrics.json` into out_dir.

    feature_mode:
    - 'ic27': inter-carrier block (variance/mlp/svm).
    - 'fusion': inter-carrier + features (mlp/svm). Overfitting risk.
    - 'cnn': CSI image (cnn)."""
    if feature_mode not in DEFAULT_BACKEND_BY_FEATURE_MODE:
        raise ValueError(
            f"feature_mode must be one of {sorted(DEFAULT_BACKEND_BY_FEATURE_MODE)}, got {feature_mode!r}"
        )
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [dataset_dirs]
    loaded: list[Dataset] = [load_dataset(dataset_dir) for dataset_dir in dataset_dirs]
    X_features, y, session_ids, subject_ids = concat_datasets(loaded)

    if feature_mode == "ic27":
        X = concat_arrays(loaded, "X_intercarrier")
    elif feature_mode == "fusion":
        X_intercarrier = concat_arrays(loaded, "X_intercarrier")
        X = np.hstack([X_intercarrier, X_features]).astype(np.float32)
    else:  # cnn
        X = concat_arrays(loaded, "X_image")

    metadata = loaded[0].meta
    num_feature_subcarriers = int(metadata["K"])
    if config is None:
        config = ModelConfig(stage="weapon", k=num_feature_subcarriers,
                             backend=DEFAULT_BACKEND_BY_FEATURE_MODE[feature_mode],
                             window=int(metadata["window"]), hop=int(metadata["hop"]),
                             frame_average=int(metadata.get("frame_average", 1)),
                             subtract_baseline=bool(metadata.get("subtract_baseline", False)),
                             subtract_ic_baseline=bool(metadata.get("subtract_ic_baseline", False)))
    else:
        # the dataset's cadence dictates serving, so enforce it here
        config = replace(config, window=int(metadata["window"]), hop=int(metadata["hop"]),
                         frame_average=int(metadata.get("frame_average", 1)),
                         subtract_baseline=bool(metadata.get("subtract_baseline", False)),
                         subtract_ic_baseline=bool(metadata.get("subtract_ic_baseline", False)))

    head = build_weapon_head(config)
    head.feature_mode = feature_mode  # serving reads this to assemble the input row
    fit_start_s = time.perf_counter()
    head.fit(X, y, report=report)  # report fires per epoch on the cnn backend, ignored elsewhere
    fit_duration_s = time.perf_counter() - fit_start_s
    if "num_subcarriers" in metadata:
        head.contract = replace(head.contract, subcarrier_width=int(metadata["num_subcarriers"]))

    classes, counts = np.unique(y, return_counts=True)
    metrics = {
        "stage": config.stage,
        "backend": config.backend,
        "feature_mode": feature_mode,
        "k": config.k,
        "n_samples": int(y.size),
        "n_features": int(X.shape[1]) if X.ndim == 2 else None,
        "image_shape": list(X.shape[1:]) if X.ndim == 3 else None,
        "class_counts": {
            str(int(class_label)): int(sample_count) for class_label, sample_count in zip(classes, counts)
        },
        "sessions": sorted({str(group_id) for group_id in session_ids}),
        "subjects": sorted({str(group_id) for group_id in subject_ids}),
        "train_accuracy": float((head.predict(X) == y).mean()),
        "logo": _logo_metrics(X, y, session_ids, subject_ids, lambda: build_weapon_head(config)),
        "subtract_ic_baseline": bool(config.subtract_ic_baseline),
        "fit_seconds": round(fit_duration_s, 3),
    }
    model_dir = Path(out_dir)
    head.save(model_dir / "model.joblib")
    with open(model_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    return head, metrics
