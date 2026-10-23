"""Static link priors from LOGO accuracy, and an offline replay of the `LinkVoter` blend."""

from dataclasses import dataclass

import numpy as np

from wavetrace.domain.recognition.link_vote import LinkVoter

__all__ = ["LinkFusionReport", "LinkVoter", "accuracy_weights", "evaluate_link_fusion"]


def accuracy_weights(balanced_accuracy: dict) -> dict:
    """LOGO balanced accuracy -> static priors: w = max(accuracy - 0.5, 0) * 2 (chance->0, perfect->1)."""
    return {node_id: max(float(accuracy) - 0.5, 0.0) * 2.0 for node_id, accuracy in balanced_accuracy.items()}


@dataclass(frozen=True, slots=True)
class LinkFusionReport:
    fused_accuracy: float
    per_link_accuracy: dict
    weights: dict
    sample_count: int


def evaluate_link_fusion(links, y, *, qualities=None) -> LinkFusionReport:
    """links: dict[node_id -> (proba, balanced_acc)]. proba is (n, C) from that link's own head;
    balanced_acc is its LOGO balanced accuracy, turned into a static prior by accuracy_weights.
    qualities: optional dict[node_id -> (n,) live quality], e.g. per-window max-proba margin.

    O(n·L·C). If every link is at/below chance (all weights 0) it falls back to a uniform blend so
    the vote is still defined."""
    y = np.asarray(y, dtype=np.int64)
    node_ids = list(links)
    weights = accuracy_weights({node_id: links[node_id][1] for node_id in node_ids})
    static_weights = weights if any(weight > 0 for weight in weights.values()) else None
    voter = LinkVoter(static_weights)
    fused_predictions = np.empty(y.size, dtype=np.int64)
    for i in range(y.size):
        for node_id in node_ids:
            quality = float(qualities[node_id][i]) if qualities and node_id in qualities else 1.0
            voter.add(node_id, links[node_id][0][i], quality=quality)
        fused_predictions[i] = voter.finalize()[0]
    per_link_accuracy = {
        node_id: float((np.argmax(links[node_id][0], axis=1) == y).mean()) for node_id in node_ids
    }
    return LinkFusionReport(
        fused_accuracy=float((fused_predictions == y).mean()),
        per_link_accuracy=per_link_accuracy,
        weights=weights,
        sample_count=int(y.size),
    )
