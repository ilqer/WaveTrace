"""Phase 6c: Eval gate (LOGO + baselines).

Always use LeaveOneGroupOut (LOGO) for generalization metrics. Random splits leak autocorrelated windows.

Baselines:
* majority-class: predict training fold's frequent class.
* PresenceSegmenter: no-train DSP gate. Windowed CV of mean channel energy. It's a segment detector, mapped here to per-window.
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from wavetrace import PresenceSegmenter
from wavetrace.Config import ModelConfig
from wavetrace.recognition.Model import PresenceHead


def leave_one_group_out(X, y, groups, make_head) -> dict:
    """Hold out one group per fold; fit a fresh head on the rest. Returns metrics dict."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    if np.unique(groups).size < 2:
        raise ValueError("leave_one_group_out: need >= 2 distinct groups")

    folds = []
    true_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    maj_all: list[np.ndarray] = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        head = make_head().fit(X[tr], y[tr])
        pred = head.predict(X[te])
        majority = int(np.bincount(y[tr]).argmax())
        maj_pred = np.full(te.size, majority, dtype=np.int64)
        folds.append({
            "group": str(groups[te[0]]),
            "n": int(te.size),
            "accuracy": float((pred == y[te]).mean()),
            "majority_accuracy": float((maj_pred == y[te]).mean()),
        })
        true_all.append(y[te])
        pred_all.append(pred)
        maj_all.append(maj_pred)

    y_true = np.concatenate(true_all)
    y_pred = np.concatenate(pred_all)
    y_maj = np.concatenate(maj_all)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
    report = {
        "folds": folds,
        "accuracy": float((y_pred == y_true).mean()),
        "majority_accuracy": float((y_maj == y_true).mean()),
        "confusion": cm,
    }
    if cm.shape == (2, 2):  # binary stage -> the P7 tier gate quantities ride along
        report.update(binary_rates(cm))
    return report


def binary_rates(confusion) -> dict:
    """{tpr, fp_rate} from 2x2 confusion matrix. O(1)."""
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.shape != (2, 2):
        raise ValueError(f"binary_rates expects a 2x2 confusion matrix, got {cm.shape}")
    pos = cm[1].sum()
    neg = cm[0].sum()
    return {
        "tpr": float(cm[1, 1] / pos) if pos else 0.0,      # correct weapon detections
        "fp_rate": float(cm[0, 1] / neg) if neg else 0.0,  # false alarms on the negative class
    }


def tier_verdict(reports, *, fp_max: float = 0.10, tpr_min: float = 0.90) -> dict:
    """Phase-7 tier gate: PASS iff EVERY report meets FP <= fp_max AND TPR >= tpr_min."""
    worst_tpr = min(r["tpr"] for r in reports.values())
    worst_fp = max(r["fp_rate"] for r in reports.values())
    reasons = []
    if worst_fp > fp_max:
        reasons.append(f"fp_rate {worst_fp:.3f} > {fp_max}")
    if worst_tpr < tpr_min:
        reasons.append(f"tpr {worst_tpr:.3f} < {tpr_min}")
    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "tpr": worst_tpr,
        "fp_rate": worst_fp,
        "fp_max": fp_max,
        "tpr_min": tpr_min,
        "reasons": reasons,
    }


def evaluate_weapon(
    X, y, *, session_ids, subject_ids, make_head,
    fp_max: float = 0.10, tpr_min: float = 0.90,
) -> dict:
    """Stage-E tier report: LOGO over session and subject + verdict. make_head returns unfitted WeaponHead."""
    reports = {
        "session": leave_one_group_out(X, y, session_ids, make_head),
        "subject": leave_one_group_out(X, y, subject_ids, make_head),
    }
    reports["verdict"] = tier_verdict(
        {k: reports[k] for k in ("session", "subject")}, fp_max=fp_max, tpr_min=tpr_min,
    )
    return reports


def evaluate_concealment_gap(
    X, y, is_concealed, groups, make_head, *,
    fp_max: float = 0.10, tpr_min: float = 0.90,
) -> dict:
    """Evaluate open->concealed transfer. Train on visible, test on concealed.

    is_concealed: mask for concealed samples.
    groups: visible reference folds.
    Returns metrics dict. Verdict based on concealed split."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    mask = np.asarray(is_concealed, dtype=bool)
    if not mask.any() or mask.all():
        raise ValueError("evaluate_concealment_gap: need both visible and concealed samples")

    Xv, yv, gv = X[~mask], y[~mask], np.asarray(groups)[~mask]
    Xc, yc = X[mask], y[mask]

    # concealed: fit on ALL visible, predict the held-out concealed set (honest transfer number)
    head = make_head().fit(Xv, yv)
    pred_c = head.predict(Xc)
    cm_c = confusion_matrix(yc, pred_c, labels=[0, 1])
    concealed = {"n": int(mask.sum()), "accuracy": float((pred_c == yc).mean()), **binary_rates(cm_c)}

    # visible reference: within-condition LOGO (same head recipe) — the "seen condition" ceiling
    visible = leave_one_group_out(Xv, yv, gv, make_head)

    reasons = []
    if concealed["fp_rate"] > fp_max:
        reasons.append(f"concealed fp_rate {concealed['fp_rate']:.3f} > {fp_max}")
    if concealed["tpr"] < tpr_min:
        reasons.append(f"concealed tpr {concealed['tpr']:.3f} < {tpr_min}")
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
    n, _, win = X_image.shape
    if cv_window > win:
        raise ValueError("cv_window must be <= the front-end window length")
    pred = np.zeros(n, dtype=np.int64)
    for i in range(n):
        seg = PresenceSegmenter(cv_window, enter_cv, exit_cv)
        cols = np.ascontiguousarray(X_image[i].T)  # (window, K): one frame's K magnitudes per row
        for f in range(win):
            if seg.push(cols[f]):
                pred[i] = 1
                break
    return pred


def evaluate_presence(
    X_features, y, *, session_ids, subject_ids, config: ModelConfig, X_image=None,
    segmenter_kwargs: dict | None = None,
) -> dict:
    """Phase-6 DoD report: LOGO over sessions and subjects + baselines."""
    make_head = lambda: PresenceHead(config)
    report = {
        "session": leave_one_group_out(X_features, y, session_ids, make_head),
        "subject": leave_one_group_out(X_features, y, subject_ids, make_head),
    }
    if X_image is not None:
        seg_pred = segmenter_baseline(X_image, **(segmenter_kwargs or {}))
        report["segmenter_accuracy"] = float((seg_pred == np.asarray(y)).mean())
    return report
