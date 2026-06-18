"""Model explainability for the observability UI.

Two flavours of antenna/channel importance:
  * STATIC weight (per model)   — L2 norm of the first Conv2d's filters per input channel.
  * DYNAMIC importance (per window) — channel ablation: zero each channel, measure the confidence
    drop on the winning class. This is 'which antenna drove THIS decision'.

Also: permutation importance for MLP/SVM feature heads, and a confusion-matrix helper.
All offline / UI-cadence — never on the <8 ms inference path.
"""

import numpy as np


def cnn_channel_weights(head) -> np.ndarray | None:
    """Per-input-channel static weight = L2 norm of the first Conv2d's filters for that input
    channel, normalized to sum 1. Returns (C,) float32, or None for non-CNN heads.

    For a WeaponHead with CNN backend, C = num nodes (multi-node images are node-as-channel).
    O(filters)."""
    net = getattr(head, "_net", None)
    if net is None:
        return None
    for m in net.modules():
        w = getattr(m, "weight", None)
        if w is not None and w.dim() == 4:           # (out_ch, in_ch, kH, kW)
            W = w.detach().cpu().numpy()
            per_in = np.sqrt((W ** 2).sum(axis=(0, 2, 3)))   # (in_ch,) energy
            s = per_in.sum()
            return (per_in / s).astype(np.float32) if s > 0 else per_in.astype(np.float32)
    return None


def ablation_importance(head, image, *, baseline_value=0.0) -> np.ndarray:
    """DYNAMIC per-channel importance for one window's image (C, K, W).
    For each channel c: zero it, re-predict, measure |p_full - p_ablated| on the winning class.
    Larger drop = that channel mattered more for this decision. Returns (C,) normalized float32.

    Cheap at UI cadence (a handful of channels). NOT for the per-window inference path.
    baseline_value: substitute amplitude when ablating (0 = absent signal)."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:                       # (K, W) single channel
        return np.array([1.0], dtype=np.float32)
    C = img.shape[0]
    full = head.predict_proba(img[np.newaxis])[0]
    cls = int(np.argmax(full))
    drops = np.empty(C, dtype=np.float64)
    for c in range(C):
        a = img.copy()
        a[c] = baseline_value
        p = head.predict_proba(a[np.newaxis])[0]
        drops[c] = abs(float(full[cls]) - float(p[cls]))
    s = drops.sum()
    return (drops / s).astype(np.float32) if s > 0 else np.full(C, 1.0 / C, np.float32)


def feature_node_importance(head, X, *, node_dim=9, k: int, n_nodes: int,
                             n_repeats=5, seed=0) -> np.ndarray:
    """Permutation importance per NODE for an MLP/SVM feature head (9·K-per-node concat).
    Shuffles each node's feature block, measures the prediction-probability shift. Returns (n_nodes,)
    normalized float64. Offline. O(n_nodes·n_repeats·predict)."""
    X = np.asarray(X, dtype=np.float32)
    block = node_dim * k
    rng = np.random.default_rng(seed)
    proba0 = head.predict_proba(X)
    imp = np.zeros(n_nodes)
    for node in range(n_nodes):
        sl = slice(node * block, (node + 1) * block)
        acc = 0.0
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, sl] = X[rng.permutation(X.shape[0])][:, sl]
            proba = head.predict_proba(Xp)
            acc += float(np.mean(np.abs(proba - proba0)))
        imp[node] = acc / n_repeats
    s = imp.sum()
    return (imp / s) if s > 0 else imp


def confusion(y_true, y_pred, n_classes=2) -> dict:
    """Confusion matrix (rows=true, cols=pred) + TPR/FPR for the binary weapon gate."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            M[t, p] += 1
    out: dict = {"matrix": M.tolist()}
    if n_classes == 2:
        tn, fp, fn, tp = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        out["tpr"] = round(tp / (tp + fn), 3) if (tp + fn) else 0.0
        out["fp_rate"] = round(fp / (fp + tn), 3) if (fp + tn) else 0.0
    return out
