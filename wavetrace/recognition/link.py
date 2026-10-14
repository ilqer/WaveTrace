"""Blend one verdict per link into one verdict, weighting each link by prior and live quality.

This is where a blocked link recovers: its live quality collapses, so weight shifts to the links
still seeing the target. It is also the only level at which the 2.4 GHz mesh and the 5 GHz Pi link
can combine, since their feature spaces never share a tensor.
"""

from dataclasses import dataclass

import numpy as np


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


class LinkVoter:
    """Blend per-link class probabilities with static-prior x live-quality weights. O(C)/add."""

    def __init__(self, static_weights: dict | None = None, *, quality_floor: float = 0.05):
        self._static_weights = static_weights or {}
        self._quality_floor = float(quality_floor)
        self._weighted_probability_sum: np.ndarray | None = None
        self._total_weight: float = 0.0
        self._class_count: int | None = None

    def add(self, node_id: int, proba, quality: float = 1.0) -> None:
        probabilities = np.asarray(proba, dtype=np.float64)
        if probabilities.ndim != 1:
            raise ValueError(f"LinkVoter.add: proba must be 1-D, got shape {probabilities.shape}")
        class_count = int(probabilities.size)
        if self._class_count is None:
            self._class_count = class_count
            self._weighted_probability_sum = np.zeros(class_count, dtype=np.float64)
        elif class_count != self._class_count:
            raise ValueError(f"LinkVoter.add: C mismatch — expected {self._class_count}, got {class_count}")
        static_weight = float(self._static_weights.get(int(node_id), 1.0))
        weight = static_weight * max(float(quality), self._quality_floor)
        self._weighted_probability_sum += weight * probabilities
        self._total_weight += weight

    def finalize(self) -> tuple:
        """Returns (class_id, blended_proba) and resets all state for the next window."""
        if self._weighted_probability_sum is None or self._total_weight == 0.0:
            raise ValueError("LinkVoter.finalize: no probabilities added")
        blended = self._weighted_probability_sum / self._total_weight
        class_id = int(np.argmax(blended))
        result = (class_id, blended.astype(np.float32))
        self._weighted_probability_sum = None
        self._total_weight = 0.0
        self._class_count = None
        return result
