"""Offline training: load datasets, concatenate them, fit a head, write the model and its metrics.

Train-set accuracy is a sanity check only. The number that counts is leaveOneGroupOut, held out
over sessions and subjects.
"""

from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np

from wavetrace.adapters.recognition import DEFAULT_BACKEND_BY_FEATURE_MODE
from wavetrace.adapters.recognition.heads import build_presence_head, build_weapon_head
from wavetrace.Config import ModelConfig
from wavetrace.domain.recognition import PresenceHead, WeaponHead
from wavetrace.groundtruth.DatasetBuilder import Dataset, loadDataset
from wavetrace.recognition.Evaluate import leaveOneGroupOut


def _carryGroups(sess):
    """Carry-position group per window from session ids. Folds on carry pose to detect nuisance learning."""
    carries = []
    for s in sess:
        parts = str(s).split("_")
        if len(parts) < 3 or parts[-1][:1] != "s" or not parts[-1][1:].isdigit():
            return None
        carries.append(parts[-2])
    return np.asarray(carries)


def _logoMetrics(X, y, sess, subj, make_head) -> dict:
    """LOGO accuracy over sessions and subjects (and carry position). O(folds*fit)."""
    out: dict = {}
    axes = [("session", sess), ("subject", subj)]
    carry = _carryGroups(sess)
    if carry is not None:
        axes.append(("carry", carry))  # weapon only: catches a head that learned the carry pose
    for axis, groups in axes:
        if np.unique(groups).size >= 2:
            rep = leaveOneGroupOut(X, y, groups, make_head)
            out[axis] = {k: rep[k] for k in ("accuracy", "majority_accuracy") if k in rep}
            out[axis].update({k: rep[k] for k in ("tpr", "fp_rate") if k in rep})
            if "confusion" in rep:
                out[axis]["confusion"] = np.asarray(rep["confusion"]).tolist()
    return out


def concatDatasets(datasets) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack multiple recordings. O(n total)."""
    ds = list(datasets)
    if not ds:
        raise ValueError("concatDatasets: no datasets")
    ks = {d.X_features.shape[1] for d in ds}
    if len(ks) != 1:
        raise ValueError(f"concatDatasets: feature dims differ across datasets: {sorted(ks)}")
    X = np.concatenate([d.X_features for d in ds]).astype(np.float32)
    y = np.concatenate([d.y for d in ds])
    sess = np.concatenate([d.session_ids for d in ds])
    subj = np.concatenate([d.subject_ids for d in ds])
    return X, y, sess, subj


def concatArrays(datasets, attr: str) -> np.ndarray:
    """Stack optional array field across recordings. O(n total)."""
    arrs = []
    for i, d in enumerate(datasets):
        a = getattr(d, attr)
        if a is None:
            raise ValueError(f"concatArrays: dataset {i} has no {attr} (rebuild with it enabled)")
        arrs.append(a)
    if not arrs:
        raise ValueError("concatArrays: no datasets")
    return np.concatenate(arrs)


def trainPresence(
    dataset_dirs,
    out_dir="models/presence",
    config: ModelConfig | None = None,
) -> tuple[PresenceHead, dict]:
    """Train and persist Stage-A presence head. Returns (head, metrics)."""
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [dataset_dirs]
    loaded: list[Dataset] = [loadDataset(d) for d in dataset_dirs]
    X, y, sess, subj = concatDatasets(loaded)
    meta = loaded[0].meta

    if config is None:
        # window/hop come from the dataset, so serving frames windows exactly as training did
        config = ModelConfig(stage="presence", k=int(meta["K"]),
                             window=int(meta["window"]), hop=int(meta["hop"]),
                             frame_average=int(meta.get("frame_average", 1)),
                             subtract_baseline=bool(meta.get("subtract_baseline", False)))

    t0 = time.perf_counter()
    head = build_presence_head(config).fit(X, y)
    fitS = time.perf_counter() - t0
    # the raw capture width, never the NBVI-selected k; None for datasets saved before
    # `num_subcarriers` was recorded, so the two can never be confused
    if "num_subcarriers" in meta:
        head.contract = replace(head.contract, subcarrier_width=int(meta["num_subcarriers"]))

    classes, counts = np.unique(y, return_counts=True)
    metrics = {
        "stage": config.stage,
        "backend": config.backend,
        "k": config.k,
        "n_samples": int(y.size),
        "n_features": int(X.shape[1]),
        "class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},  # str: JSON-stable
        "sessions": sorted({str(s) for s in sess}),
        "subjects": sorted({str(s) for s in subj}),
        "train_accuracy": float((head.predict(X) == y).mean()),  # sanity only — see module note
        "logo": _logoMetrics(X, y, sess, subj, lambda: build_presence_head(config)),  # the HEADLINE number
        "fit_seconds": round(fitS, 3),
    }
    out = Path(out_dir)
    head.save(out / "model.joblib")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return head, metrics


def trainWeapon(
    dataset_dirs,
    out_dir="models/weapon",
    config: ModelConfig | None = None,
    feature_mode: str = "ic27",
    report=None,
) -> tuple[WeaponHead, dict]:
    """Train and persist Stage-E weapon head.

    feature_mode:
    - 'ic27': inter-carrier block (variance/mlp/svm).
    - 'fusion': inter-carrier + features (mlp/svm). Overfitting risk.
    - 'cnn': CSI image (cnn).
    Returns (head, metrics)."""
    if feature_mode not in DEFAULT_BACKEND_BY_FEATURE_MODE:
        raise ValueError(
            f"feature_mode must be one of {sorted(DEFAULT_BACKEND_BY_FEATURE_MODE)}, got {feature_mode!r}"
        )
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [dataset_dirs]
    loaded: list[Dataset] = [loadDataset(d) for d in dataset_dirs]
    XFeat, y, sess, subj = concatDatasets(loaded)

    if feature_mode == "ic27":
        X = concatArrays(loaded, "X_intercarrier")
    elif feature_mode == "fusion":
        XIc = concatArrays(loaded, "X_intercarrier")
        X = np.hstack([XIc, XFeat]).astype(np.float32)
    else:  # cnn
        X = concatArrays(loaded, "X_image")

    meta = loaded[0].meta
    K = int(meta["K"])
    if config is None:
        config = ModelConfig(stage="weapon", k=K, backend=DEFAULT_BACKEND_BY_FEATURE_MODE[feature_mode],
                             window=int(meta["window"]), hop=int(meta["hop"]),
                             frame_average=int(meta.get("frame_average", 1)),
                             subtract_baseline=bool(meta.get("subtract_baseline", False)),
                             subtract_ic_baseline=bool(meta.get("subtract_ic_baseline", False)))
    else:
        # the dataset's cadence dictates serving, so enforce it here
        config = replace(config, window=int(meta["window"]), hop=int(meta["hop"]),
                         frame_average=int(meta.get("frame_average", 1)),
                         subtract_baseline=bool(meta.get("subtract_baseline", False)),
                         subtract_ic_baseline=bool(meta.get("subtract_ic_baseline", False)))

    head = build_weapon_head(config)
    head.feature_mode = feature_mode  # serving reads this to assemble the input row
    t0 = time.perf_counter()
    head.fit(X, y, report=report)  # report fires per epoch on the cnn backend, ignored elsewhere
    fitS = time.perf_counter() - t0
    if "num_subcarriers" in meta:
        head.contract = replace(head.contract, subcarrier_width=int(meta["num_subcarriers"]))

    classes, counts = np.unique(y, return_counts=True)
    metrics = {
        "stage": config.stage,
        "backend": config.backend,
        "feature_mode": feature_mode,
        "k": config.k,
        "n_samples": int(y.size),
        "n_features": int(X.shape[1]) if X.ndim == 2 else None,
        "image_shape": list(X.shape[1:]) if X.ndim == 3 else None,
        "class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
        "sessions": sorted({str(s) for s in sess}),
        "subjects": sorted({str(s) for s in subj}),
        "train_accuracy": float((head.predict(X) == y).mean()),
        "logo": _logoMetrics(X, y, sess, subj, lambda: build_weapon_head(config)),  # the HEADLINE number
        "subtract_ic_baseline": bool(config.subtract_ic_baseline),
        "fit_seconds": round(fitS, 3),
    }
    out = Path(out_dir)
    head.save(out / "model.joblib")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return head, metrics
