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


def leaveOneGroupOut(X, y, groups, make_head) -> dict:
    """Hold out one group per fold; fit a fresh head on the rest. Returns metrics dict."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    if np.unique(groups).size < 2:
        raise ValueError("leaveOneGroupOut: need >= 2 distinct groups")

    folds = []
    trueAll: list[np.ndarray] = []
    predAll: list[np.ndarray] = []
    majAll: list[np.ndarray] = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        head = make_head().fit(X[tr], y[tr])
        pred = head.predict(X[te])
        majority = int(np.bincount(y[tr]).argmax())
        majPred = np.full(te.size, majority, dtype=np.int64)
        folds.append({
            "group": str(groups[te[0]]),
            "n": int(te.size),
            "accuracy": float((pred == y[te]).mean()),
            "majority_accuracy": float((majPred == y[te]).mean()),
        })
        trueAll.append(y[te])
        predAll.append(pred)
        majAll.append(majPred)

    yTrue = np.concatenate(trueAll)
    yPred = np.concatenate(predAll)
    yMaj = np.concatenate(majAll)
    cm = confusion_matrix(yTrue, yPred, labels=np.unique(y))
    report = {
        "folds": folds,
        "accuracy": float((yPred == yTrue).mean()),
        "majority_accuracy": float((yMaj == yTrue).mean()),
        "confusion": cm,
    }
    if cm.shape == (2, 2):  # binary stage -> the P7 tier gate quantities ride along
        report.update(binaryRates(cm))
    return report


def binaryRates(confusion) -> dict:
    """{tpr, fp_rate} from 2x2 confusion matrix. O(1)."""
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.shape != (2, 2):
        raise ValueError(f"binaryRates expects a 2x2 confusion matrix, got {cm.shape}")
    pos = cm[1].sum()
    neg = cm[0].sum()
    return {
        "tpr": float(cm[1, 1] / pos) if pos else 0.0,      # correct weapon detections
        "fp_rate": float(cm[0, 1] / neg) if neg else 0.0,  # false alarms on the negative class
    }


def tierVerdict(reports, *, fp_max: float = 0.10, tpr_min: float = 0.90) -> dict:
    """Phase-7 tier gate: PASS iff EVERY report meets FP <= fp_max AND TPR >= tpr_min."""
    worstTpr = min(r["tpr"] for r in reports.values())
    worstFp = max(r["fp_rate"] for r in reports.values())
    reasons = []
    if worstFp > fp_max:
        reasons.append(f"fp_rate {worstFp:.3f} > {fp_max}")
    if worstTpr < tpr_min:
        reasons.append(f"tpr {worstTpr:.3f} < {tpr_min}")
    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "tpr": worstTpr,
        "fp_rate": worstFp,
        "fp_max": fp_max,
        "tpr_min": tpr_min,
        "reasons": reasons,
    }


def evaluateWeapon(
    X, y, *, session_ids, subject_ids, make_head,
    fp_max: float = 0.10, tpr_min: float = 0.90,
) -> dict:
    """Stage-E tier report: LOGO over session and subject + verdict. make_head returns unfitted WeaponHead."""
    reports = {
        "session": leaveOneGroupOut(X, y, session_ids, make_head),
        "subject": leaveOneGroupOut(X, y, subject_ids, make_head),
    }
    reports["verdict"] = tierVerdict(
        {k: reports[k] for k in ("session", "subject")}, fp_max=fp_max, tpr_min=tpr_min,
    )
    return reports


def evaluateConcealmentGap(
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
        raise ValueError("evaluateConcealmentGap: need both visible and concealed samples")

    Xv, yv, gv = X[~mask], y[~mask], np.asarray(groups)[~mask]
    Xc, yc = X[mask], y[mask]

    # concealed: fit on ALL visible, predict the held-out concealed set (honest transfer number)
    head = make_head().fit(Xv, yv)
    predC = head.predict(Xc)
    cmC = confusion_matrix(yc, predC, labels=[0, 1])
    concealed = {"n": int(mask.sum()), "accuracy": float((predC == yc).mean()), **binaryRates(cmC)}

    # visible reference: within-condition LOGO (same head recipe) — the "seen condition" ceiling
    visible = leaveOneGroupOut(Xv, yv, gv, make_head)

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


def segmenterBaseline(
    X_image, *, cv_window: int = 32, enter_cv: float = 0.08, exit_cv: float = 0.04
) -> np.ndarray:
    """No-train DSP baseline: per-window present/absent from PresenceSegmenter. O(n * window * cv_window)."""
    X_image = np.asarray(X_image, dtype=np.float32)
    if X_image.ndim != 3:
        raise ValueError(f"segmenterBaseline expects (n, K, window), got {X_image.shape}")
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


def evaluatePresence(
    X_features, y, *, session_ids, subject_ids, config: ModelConfig, X_image=None,
    segmenter_kwargs: dict | None = None,
) -> dict:
    """Phase-6 DoD report: LOGO over sessions and subjects + baselines."""
    make_head = lambda: PresenceHead(config)
    report = {
        "session": leaveOneGroupOut(X_features, y, session_ids, make_head),
        "subject": leaveOneGroupOut(X_features, y, subject_ids, make_head),
    }
    if X_image is not None:
        segPred = segmenterBaseline(X_image, **(segmenter_kwargs or {}))
        report["segmenter_accuracy"] = float((segPred == np.asarray(y)).mean())
    return report
