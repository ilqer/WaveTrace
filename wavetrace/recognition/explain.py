"""What the model is paying attention to, for the dashboard.

Two views of a CNN: a static per-channel weight, the L2 norm of the first Conv2d's filters, and a
per-window importance from ablation - zero a channel and measure the confidence it costs. For
MLP/SVM there is permutation importance instead. UI cadence only, never the hot path.
"""

import numpy as np


def cnn_channel_weights(head) -> np.ndarray | None:
    """Static weight per input channel (normalized L2 norm of Conv2d filters). O(filters)."""
    network = getattr(head, "_net", None)
    if network is None:
        return None
    for module in network.modules():
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dim() == 4:           # (out_ch, in_ch, kH, kW)
            filter_weights = weight.detach().cpu().numpy()
            input_channel_energy = np.sqrt((filter_weights ** 2).sum(axis=(0, 2, 3)))
            total_energy = input_channel_energy.sum()
            return ((input_channel_energy / total_energy).astype(np.float32) if total_energy > 0
                    else input_channel_energy.astype(np.float32))
    return None


def ablation_importance(head, image, *, baseline_value=0.0) -> np.ndarray:
    """Per-channel importance: set one channel to baseline_value and measure how much the
    top class's confidence drops, normalized to sum 1. O(channels * predict)."""
    csi_image = np.asarray(image, dtype=np.float32)
    if csi_image.ndim == 2:                       # (K, W) single channel
        return np.array([1.0], dtype=np.float32)
    channel_count = csi_image.shape[0]
    full_probabilities = head.predict_proba(csi_image[np.newaxis])[0]
    predicted_class = int(np.argmax(full_probabilities))
    confidence_drops = np.empty(channel_count, dtype=np.float64)
    for channel in range(channel_count):
        ablated_image = csi_image.copy()
        ablated_image[channel] = baseline_value
        ablated_probabilities = head.predict_proba(ablated_image[np.newaxis])[0]
        confidence_drops[channel] = abs(float(full_probabilities[predicted_class])
                                        - float(ablated_probabilities[predicted_class]))
    total_drop = confidence_drops.sum()
    return ((confidence_drops / total_drop).astype(np.float32) if total_drop > 0
            else np.full(channel_count, 1.0 / channel_count, np.float32))


def feature_node_importance(
    head, X, *, features_per_subcarrier=9, subcarriers_per_node: int, num_nodes: int,
    num_repeats=5, seed=0,
) -> np.ndarray:
    """Permutation importance per node for MLP/SVM feature head. O(num_nodes*num_repeats*predict)."""
    X = np.asarray(X, dtype=np.float32)
    features_per_node = features_per_subcarrier * subcarriers_per_node
    rng = np.random.default_rng(seed)
    baseline_probabilities = head.predict_proba(X)
    importance = np.zeros(num_nodes)
    for node in range(num_nodes):
        node_columns = slice(node * features_per_node, (node + 1) * features_per_node)
        total_change = 0.0
        for _ in range(num_repeats):
            X_permuted = X.copy()
            X_permuted[:, node_columns] = X[rng.permutation(X.shape[0])][:, node_columns]
            permuted_probabilities = head.predict_proba(X_permuted)
            total_change += float(np.mean(np.abs(permuted_probabilities - baseline_probabilities)))
        importance[node] = total_change / num_repeats
    total_importance = importance.sum()
    return (importance / total_importance) if total_importance > 0 else importance


def confusion(y_true, y_pred, num_classes=2) -> dict:
    """Confusion matrix (rows=true, cols=pred) + TPR/FPR for the binary weapon gate. Labels outside
    [0, num_classes) are silently dropped."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= predicted_label < num_classes:
            matrix[true_label, predicted_label] += 1
    report: dict = {"matrix": matrix.tolist()}
    if num_classes == 2:
        true_negatives, false_positives, false_negatives, true_positives = (
            matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        )
        report["tpr"] = (round(true_positives / (true_positives + false_negatives), 3)
                         if (true_positives + false_negatives) else 0.0)
        report["fp_rate"] = (round(false_positives / (false_positives + true_negatives), 3)
                             if (false_positives + true_negatives) else 0.0)
    return report
