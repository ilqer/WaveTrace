"""Phase 6b — offline training driver: Phase-5 dataset(s) -> fitted PresenceHead under models/.

Loads one or more serialized datasets (`groundtruth.load_dataset`), concatenates them (one recording
= one (session, subject) group), fits the head on X_features, and persists model.joblib +
metrics.json. models/ is gitignored — artifacts never enter the repo. OFFLINE (correctness over
Big-O); the saved model is what Infer.py serves on the real-time path.

NOTE: train-set accuracy in metrics.json is a sanity number only. The HEADLINE number must come from
Evaluate.leave_one_group_out (session AND subject) — never a random within-session split (rev-7 #1).
"""

from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.groundtruth.DatasetBuilder import Dataset, load_dataset
from wavetrace.recognition.Evaluate import leave_one_group_out
from wavetrace.recognition.Model import PresenceHead
from wavetrace.recognition.Weapon import WeaponHead


def _carry_groups(sess):
    """Carry-position group per window, parsed from weapon session ids `<subject>_<carry>_s<n>`
    (collect_weapon's sess_id). Returns None unless EVERY id matches that shape, so presence/other
    session ids are left untouched and never get a spurious carry axis. The point (diagnosis CAUSE
    5E): a below-chance head is often keying on a NUISANCE like carry pose, not the weapon — folding
    on carry exposes that."""
    carries = []
    for s in sess:
        parts = str(s).split("_")
        if len(parts) < 3 or parts[-1][:1] != "s" or not parts[-1][1:].isdigit():
            return None
        carries.append(parts[-2])
    return np.asarray(carries)


def _logo_metrics(X, y, sess, subj, make_head) -> dict:
    """Headline leave-one-group-out accuracy over sessions AND subjects (AND carry position when the
    session ids encode it), computed only when a group has >= 2 distinct values (a single synthetic
    session can't be folded — rev-7 #1). Stores the pooled accuracy + majority baseline per axis;
    absent axes are skipped. Confusion drops out (not JSON-native). O(folds·fit)."""
    out: dict = {}
    axes = [("session", sess), ("subject", subj)]
    carry = _carry_groups(sess)
    if carry is not None:
        axes.append(("carry", carry))  # weapon-only confound axis (diagnosis Item 13)
    for axis, groups in axes:
        if np.unique(groups).size >= 2:
            rep = leave_one_group_out(X, y, groups, make_head)
            out[axis] = {k: rep[k] for k in ("accuracy", "majority_accuracy") if k in rep}
            out[axis].update({k: rep[k] for k in ("tpr", "fp_rate") if k in rep})
            if "confusion" in rep:
                out[axis]["confusion"] = np.asarray(rep["confusion"]).tolist()
    return out


def concat_datasets(datasets) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack multiple recordings into (X_features, y, session_ids, subject_ids). O(n total).
    Group ids ride along so the Evaluate fold gate can split on them."""
    ds = list(datasets)
    if not ds:
        raise ValueError("concat_datasets: no datasets")
    ks = {d.X_features.shape[1] for d in ds}
    if len(ks) != 1:
        raise ValueError(f"concat_datasets: feature dims differ across datasets: {sorted(ks)}")
    X = np.concatenate([d.X_features for d in ds]).astype(np.float32)
    y = np.concatenate([d.y for d in ds])
    sess = np.concatenate([d.session_ids for d in ds])
    subj = np.concatenate([d.subject_ids for d in ds])
    return X, y, sess, subj


def concat_arrays(datasets, attr: str) -> np.ndarray:
    """Stack one optional array field (e.g. "X_intercarrier", "X_image") across recordings.
    Raises if any dataset lacks it (built without intercarrier=True). O(n total)."""
    arrs = []
    for i, d in enumerate(datasets):
        a = getattr(d, attr)
        if a is None:
            raise ValueError(f"concat_arrays: dataset {i} has no {attr} (rebuild with it enabled)")
        arrs.append(a)
    if not arrs:
        raise ValueError("concat_arrays: no datasets")
    return np.concatenate(arrs)


def train_presence(
    dataset_dirs,
    out_dir="models/presence",
    config: ModelConfig | None = None,
) -> tuple[PresenceHead, dict]:
    """Train the Stage-A presence head and persist it.

    dataset_dirs: one dataset directory or a sequence of them (each from `save_dataset`).
    config: ModelConfig; default = presence/MLP with k taken from the dataset meta.
    Writes <out_dir>/model.joblib + <out_dir>/metrics.json; returns (head, metrics)."""
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [dataset_dirs]
    loaded: list[Dataset] = [load_dataset(d) for d in dataset_dirs]
    X, y, sess, subj = concat_datasets(loaded)

    if config is None:
        meta = loaded[0].meta
        # window/hop come from the dataset's front-end cadence so serving (Cli.run) matches training
        config = ModelConfig(stage="presence", k=int(meta["K"]),
                             window=int(meta["window"]), hop=int(meta["hop"]),
                             frame_average=int(meta.get("frame_average", 1)),
                             subtract_baseline=bool(meta.get("subtract_baseline", False)))

    t0 = time.perf_counter()
    head = PresenceHead(config).fit(X, y)
    fit_s = time.perf_counter() - t0

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
        "logo": _logo_metrics(X, y, sess, subj, lambda: PresenceHead(config)),  # the HEADLINE number
        "fit_seconds": round(fit_s, 3),
    }
    out = Path(out_dir)
    head.save(out / "model.joblib")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return head, metrics


def train_weapon(
    dataset_dirs,
    out_dir="models/weapon",
    config: ModelConfig | None = None,
    feature_mode: str = "ic27",
) -> tuple[WeaponHead, dict]:
    """Train the Stage-E weapon head and persist it.

    feature_mode selects which X the head trains on:
      "ic27"   — X_intercarrier (n, 27) inter-carrier block; requires intercarrier=True datasets.
                 Works with "variance"/"mlp"/"svm" backends. The baseline to beat under LOGO first.
      "fusion" — np.hstack([X_intercarrier, X_features]) → (n, 27+9·K); requires dual-block datasets
                 (intercarrier=True + gain_lock). Only "mlp"/"svm". Test against ic27 baseline; on
                 small real datasets (≥2 sessions/subjects) overfitting is a real risk.
      "cnn"    — X_image (n, K, window); only the "cnn" backend.
    Writes <out_dir>/model.joblib + <out_dir>/metrics.json; returns (head, metrics)."""
    if feature_mode not in ("ic27", "fusion", "cnn"):
        raise ValueError(f"feature_mode must be 'ic27', 'fusion', or 'cnn', got {feature_mode!r}")
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [dataset_dirs]
    loaded: list[Dataset] = [load_dataset(d) for d in dataset_dirs]
    X_feat, y, sess, subj = concat_datasets(loaded)

    if feature_mode == "ic27":
        X = concat_arrays(loaded, "X_intercarrier")
    elif feature_mode == "fusion":
        X_ic = concat_arrays(loaded, "X_intercarrier")
        X = np.hstack([X_ic, X_feat]).astype(np.float32)
    else:  # cnn
        X = concat_arrays(loaded, "X_image")

    meta = loaded[0].meta
    K = int(meta["K"])
    if config is None:
        backend = "variance" if feature_mode == "ic27" else "cnn" if feature_mode == "cnn" else "mlp"
        config = ModelConfig(stage="weapon", k=K, backend=backend,
                             window=int(meta["window"]), hop=int(meta["hop"]),
                             frame_average=int(meta.get("frame_average", 1)),
                             subtract_baseline=bool(meta.get("subtract_baseline", False)),
                             subtract_ic_baseline=bool(meta.get("subtract_ic_baseline", False)))
    else:
        # the dataset's front-end cadence dictates serving; enforce it so Cli.run matches training
        config = replace(config, window=int(meta["window"]), hop=int(meta["hop"]),
                         frame_average=int(meta.get("frame_average", 1)),
                         subtract_baseline=bool(meta.get("subtract_baseline", False)),
                         subtract_ic_baseline=bool(meta.get("subtract_ic_baseline", False)))

    head = WeaponHead(config)
    head.feature_mode = feature_mode  # self-describing: Cli.run reads it to assemble x at serve time
    t0 = time.perf_counter()
    head.fit(X, y)
    fit_s = time.perf_counter() - t0

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
        "logo": _logo_metrics(X, y, sess, subj, lambda: WeaponHead(config)),  # the HEADLINE number
        "subtract_ic_baseline": bool(config.subtract_ic_baseline),
        "fit_seconds": round(fit_s, 3),
    }
    out = Path(out_dir)
    head.save(out / "model.joblib")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return head, metrics
