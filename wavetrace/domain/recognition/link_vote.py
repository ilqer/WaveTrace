"""Blend one verdict per link into one verdict, weighting each link by prior and live quality.

This is where a blocked link recovers: its live quality collapses, so weight shifts to the links
still seeing the target. It is also the only level at which the 2.4 GHz mesh and the 5 GHz Pi link
can combine, since their feature spaces never share a tensor.
"""

import numpy as np


class LinkVoter:
    """Blend per-link class probabilities with static-prior x live-quality weights. O(C)/add."""

    def __init__(self, static_weights: dict | None = None, *, quality_floor: float = 0.05):
        self._static_weights = static_weights or {}
        self._quality_floor = float(quality_floor)
        self._weighted_probability_sum: np.ndarray | None = None
        self._total_weight: float = 0.0
        self._class_count: int | None = None

    def add(self, link_id: int, proba, quality: float = 1.0) -> None:
        probabilities = np.asarray(proba, dtype=np.float64)
        if probabilities.ndim != 1:
            raise ValueError(f"LinkVoter.add: proba must be 1-D, got shape {probabilities.shape}")
        class_count = int(probabilities.size)
        if self._class_count is None:
            self._class_count = class_count
            self._weighted_probability_sum = np.zeros(class_count, dtype=np.float64)
        elif class_count != self._class_count:
            raise ValueError(f"LinkVoter.add: C mismatch — expected {self._class_count}, got {class_count}")
        static_weight = float(self._static_weights.get(int(link_id), 1.0))
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
