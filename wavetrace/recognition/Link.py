"""Blend one verdict per link into one verdict, weighting each link by prior and live quality.

This is where a blocked link recovers: its live quality collapses, so weight shifts to the links
still seeing the target. It is also the only level at which the 2.4 GHz mesh and the 5 GHz Pi link
can combine, since their feature spaces never share a tensor.

One trained head per link, then `add` per link per window, then `finalize`. `quality` comes from
the caller (a max-probability margin, window motion energy). Static priors come from
`accuracyWeights` over per-link LOGO results, or from the operator.
"""

from dataclasses import dataclass

import numpy as np


def accuracyWeights(balanced_acc: dict) -> dict:
    """LOGO balanced accuracy -> static priors: w = max(acc - 0.5, 0) * 2 (chance->0, perfect->1)."""
    return {k: max(float(v) - 0.5, 0.0) * 2.0 for k, v in balanced_acc.items()}


@dataclass(frozen=True, slots=True)
class LinkFusionReport:
    """Offline measurement of decision-level band fusion (`evaluateLinkFusion`): fused accuracy
    against each link's own accuracy and the static prior weight it was blended with."""

    fused_accuracy: float
    per_link_accuracy: dict
    weights: dict
    sample_count: int


def evaluateLinkFusion(links, y, *, qualities=None) -> LinkFusionReport:
    """Measure band fusion offline: blend each link's per-window class probabilities with
    accuracy-derived static priors, and report fused accuracy against the best single link.

    links: dict[node_id -> (proba, balanced_acc)]. proba is (n, C) from that link's own head, and
    balanced_acc is its LOGO balanced accuracy, turned into a static prior by accuracyWeights.
    qualities: optional dict[node_id -> (n,) live quality], e.g. per-window max-proba margin.

    O(n·L·C). If every link is at/below chance (all weights 0) it falls back to a uniform blend so
    the vote is still defined."""
    y = np.asarray(y, dtype=np.int64)
    ids = list(links)
    weights = accuracyWeights({nid: links[nid][1] for nid in ids})
    static = weights if any(w > 0 for w in weights.values()) else None  # uniform if all at chance
    voter = LinkVoter(static)
    fused = np.empty(y.size, dtype=np.int64)
    for i in range(y.size):
        for nid in ids:
            q = float(qualities[nid][i]) if qualities and nid in qualities else 1.0
            voter.add(nid, links[nid][0][i], quality=q)
        fused[i] = voter.finalize()[0]
    perLink = {nid: float((np.argmax(links[nid][0], axis=1) == y).mean()) for nid in ids}
    return LinkFusionReport(
        fused_accuracy=float((fused == y).mean()),
        per_link_accuracy=perLink,
        weights=weights,
        sample_count=int(y.size),
    )


class LinkVoter:
    """Blend per-link class probabilities with static-prior x live-quality weights. O(C)/add.

    Reusable per window: finalize() resets all state so a new round of add() is independent."""

    def __init__(self, static_weights: dict | None = None, *, quality_floor: float = 0.05):
        self._static = static_weights or {}
        self._quality_floor = float(quality_floor)
        self._wsum: np.ndarray | None = None
        self._total: float = 0.0
        self._C: int | None = None

    def add(self, node_id: int, proba, quality: float = 1.0) -> None:
        """Accumulate one link's probability vector with its combined weight. O(C)."""
        p = np.asarray(proba, dtype=np.float64)
        if p.ndim != 1:
            raise ValueError(f"LinkVoter.add: proba must be 1-D, got shape {p.shape}")
        C = int(p.size)
        if self._C is None:
            self._C = C
            self._wsum = np.zeros(C, dtype=np.float64)
        elif C != self._C:
            raise ValueError(f"LinkVoter.add: C mismatch — expected {self._C}, got {C}")
        static = float(self._static.get(int(node_id), 1.0))
        w = static * max(float(quality), self._quality_floor)
        self._wsum += w * p
        self._total += w

    def finalize(self) -> tuple:
        """Blend and return (class_id, blended_proba); reset all state for the next window."""
        if self._wsum is None or self._total == 0.0:
            raise ValueError("LinkVoter.finalize: no probabilities added")
        blended = self._wsum / self._total
        cls = int(np.argmax(blended))
        result = (cls, blended.astype(np.float32))
        self._wsum = None
        self._total = 0.0
        self._C = None
        return result
