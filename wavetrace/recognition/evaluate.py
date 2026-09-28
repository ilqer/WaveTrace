"""The evaluation gate: leave-one-group-out, against two baselines.

Only LOGO counts. A random split leaks, because neighbouring windows overlap and are correlated.

The baselines are the majority class of the training fold, and PresenceSegmenter - an untrained DSP
gate over the windowed CV of mean channel energy, mapped from segments onto windows.
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from wavetrace import PresenceSegmenter


def leave_one_group_out(X, y, groups, make_head) -> dict:
    """Hold out one group per fold and fit a fresh `make_head()` on the rest. O(folds * fit).
    Raises ValueError with fewer than 2 distinct groups."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    if np.unique(groups).size < 2:
        raise ValueError("leave_one_group_out: need >= 2 distinct groups")

    folds = []
    fold_true_labels: list[np.ndarray] = []
    fold_predictions: list[np.ndarray] = []
    fold_majority_predictions: list[np.ndarray] = []
    for train_indices, test_indices in LeaveOneGroupOut().split(X, y, groups):
        head = make_head().fit(X[train_indices], y[train_indices])
        test_predictions = head.predict(X[test_indices])
        majority_class = int(np.bincount(y[train_indices]).argmax())
        majority_predictions = np.full(test_indices.size, majority_class, dtype=np.int64)
        folds.append({
            "group": str(groups[test_indices[0]]),
            "n": int(test_indices.size),
            "accuracy": float((test_predictions == y[test_indices]).mean()),
            "majority_accuracy": float((majority_predictions == y[test_indices]).mean()),
        })
        fold_true_labels.append(y[test_indices])
        fold_predictions.append(test_predictions)
        fold_majority_predictions.append(majority_predictions)

    y_true = np.concatenate(fold_true_labels)
    y_pred = np.concatenate(fold_predictions)
    y_majority = np.concatenate(fold_majority_predictions)
    pooled_confusion = confusion_matrix(y_true, y_pred, labels=np.unique(y))
    report = {
        "folds": folds,
        "accuracy": float((y_pred == y_true).mean()),
        "majority_accuracy": float((y_majority == y_true).mean()),
        "confusion": pooled_confusion,
    }
    if pooled_confusion.shape == (2, 2):  # binary stage: the tier-gate rates ride along
        report.update(binary_rates(pooled_confusion))
    return report


def binary_rates(confusion) -> dict:
    """{tpr, fp_rate} from a 2x2 confusion matrix (rows = true class). O(1)."""
    confusion_counts = np.asarray(confusion, dtype=np.float64)
    if confusion_counts.shape != (2, 2):
        raise ValueError(f"binary_rates expects a 2x2 confusion matrix, got {confusion_counts.shape}")
    positive_count = confusion_counts[1].sum()
    negative_count = confusion_counts[0].sum()
    return {
        # correct weapon detections
        "tpr": float(confusion_counts[1, 1] / positive_count) if positive_count else 0.0,
        # false alarms on the negative class
        "fp_rate": float(confusion_counts[0, 1] / negative_count) if negative_count else 0.0,
    }


def tier_verdict(
    reports, *, max_false_positive_rate: float = 0.10, min_true_positive_rate: float = 0.90,
) -> dict:
    """PASS only when every report meets FP <= max_false_positive_rate and TPR >= min_true_positive_rate."""
    worst_true_positive_rate = min(report["tpr"] for report in reports.values())
    worst_false_positive_rate = max(report["fp_rate"] for report in reports.values())
    reasons = []
    if worst_false_positive_rate > max_false_positive_rate:
        reasons.append(f"fp_rate {worst_false_positive_rate:.3f} > {max_false_positive_rate}")
    if worst_true_positive_rate < min_true_positive_rate:
        reasons.append(f"tpr {worst_true_positive_rate:.3f} < {min_true_positive_rate}")
    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "tpr": worst_true_positive_rate,
        "fp_rate": worst_false_positive_rate,
        "fp_max": max_false_positive_rate,
        "tpr_min": min_true_positive_rate,
        "reasons": reasons,
    }


def evaluate_weapon(
    X, y, *, session_ids, subject_ids, make_head,
    max_false_positive_rate: float = 0.10, min_true_positive_rate: float = 0.90,
) -> dict:
    """LOGO over session and subject, plus a verdict. make_head returns an unfitted WeaponHead."""
    reports = {
        "session": leave_one_group_out(X, y, session_ids, make_head),
        "subject": leave_one_group_out(X, y, subject_ids, make_head),
    }
    reports["verdict"] = tier_verdict(
        {axis: reports[axis] for axis in ("session", "subject")},
        max_false_positive_rate=max_false_positive_rate, min_true_positive_rate=min_true_positive_rate,
    )
    return reports


def evaluate_concealment_gap(
    X, y, is_concealed, groups, make_head, *,
    max_false_positive_rate: float = 0.10, min_true_positive_rate: float = 0.90,
) -> dict:
    """Evaluate open->concealed transfer. Train on visible, test on concealed.

    is_concealed: mask for concealed samples.
    groups: visible reference folds.
    The verdict is judged on the concealed split only."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    concealed_mask = np.asarray(is_concealed, dtype=bool)
    if not concealed_mask.any() or concealed_mask.all():
        raise ValueError("evaluate_concealment_gap: need both visible and concealed samples")

    X_visible, y_visible, groups_visible = (
        X[~concealed_mask], y[~concealed_mask], np.asarray(groups)[~concealed_mask]
    )
    X_concealed, y_concealed = X[concealed_mask], y[concealed_mask]

    # fit on all visible, predict the held-out concealed set: the honest transfer number
    head = make_head().fit(X_visible, y_visible)
    concealed_predictions = head.predict(X_concealed)
    concealed_confusion = confusion_matrix(y_concealed, concealed_predictions, labels=[0, 1])
    concealed = {"n": int(concealed_mask.sum()),
                 "accuracy": float((concealed_predictions == y_concealed).mean()),
                 **binary_rates(concealed_confusion)}

    # visible reference: within-condition LOGO (same head recipe) — the "seen condition" ceiling
    visible = leave_one_group_out(X_visible, y_visible, groups_visible, make_head)

    reasons = []
    if concealed["fp_rate"] > max_false_positive_rate:
        reasons.append(f"concealed fp_rate {concealed['fp_rate']:.3f} > {max_false_positive_rate}")
    if concealed["tpr"] < min_true_positive_rate:
        reasons.append(f"concealed tpr {concealed['tpr']:.3f} < {min_true_positive_rate}")
    return {
        "concealed": concealed,
        "visible": visible,
        "tpr_gap": float(visible.get("tpr", 0.0) - concealed["tpr"]),  # how much transfer costs
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
    }


def segmenter_baseline(
    X_image, *, cv_window: int = 32, enter_cv: float = 0.08, exit_cv: float = 0.04
) -> np.ndarray:
    """No-train DSP baseline: per-window present/absent from PresenceSegmenter. O(n * window * cv_window)."""
    X_image = np.asarray(X_image, dtype=np.float32)
    if X_image.ndim != 3:
        raise ValueError(f"segmenter_baseline expects (n, K, window), got {X_image.shape}")
    window_count, _, frames_per_window = X_image.shape
    if cv_window > frames_per_window:
        raise ValueError("cv_window must be <= the front-end window length")
    predictions = np.zeros(window_count, dtype=np.int64)
    for i in range(window_count):
        segmenter = PresenceSegmenter(cv_window, enter_cv, exit_cv)
        frame_magnitudes = np.ascontiguousarray(X_image[i].T)  # (window, K): one frame's K magnitudes per row
        for frame_index in range(frames_per_window):
            if segmenter.push(frame_magnitudes[frame_index]):
                predictions[i] = 1
                break
    return predictions


def evaluate_presence(
    X_features, y, *, session_ids, subject_ids, make_head, X_image=None,
    segmenter_kwargs: dict | None = None,
) -> dict:
    """LOGO over sessions and subjects, plus the baselines. `make_head` returns an unfitted
    `PresenceHead`, the same contract as `evaluate_weapon`'s."""
    report = {
        "session": leave_one_group_out(X_features, y, session_ids, make_head),
        "subject": leave_one_group_out(X_features, y, subject_ids, make_head),
    }
    if X_image is not None:
        segmenter_predictions = segmenter_baseline(X_image, **(segmenter_kwargs or {}))
        report["segmenter_accuracy"] = float((segmenter_predictions == np.asarray(y)).mean())
    return report
