"""T6/P10 — decision-level per-link fusion via weighted probability blending.

`LinkVoter`: blend per-link class probabilities with static-prior × live-quality weights.
Operator importance knob + blockage-recovery mechanism (a blocked link's live quality collapses →
weight shifts to links still seeing the target, plan §2.9.3) + the only level where the 2.4 GHz
mesh and 5 GHz Pi link can combine (different feature spaces).

Usage: one trained head per link → per-window `add` per link → `finalize`.
`quality` is caller-supplied (e.g. max(proba) margin or window motion energy).
Static priors from `accuracy_weights` of per-link LOGO results, or operator-set.
NOT wired into Cli — lands with multi-node serving in Phase 0+.
"""

import numpy as np


def accuracy_weights(balanced_acc: dict) -> dict:
    """LOGO balanced accuracy -> static priors: w = max(acc - 0.5, 0) * 2 (chance->0, perfect->1)."""
    return {k: max(float(v) - 0.5, 0.0) * 2.0 for k, v in balanced_acc.items()}


def evaluate_link_fusion(links, y, *, qualities=None) -> dict:
    """Measure decision-level band fusion offline — the ONLY level the 2.4 GHz mesh and the 5 GHz Pi
    combine (different feature spaces never share a tensor, plan §2.9.3). Blend each link's per-window
    class probabilities with accuracy-derived static priors and report fused vs best-single accuracy.

    links: dict[node_id -> (proba, balanced_acc)] — proba is (n, C) from that link's OWN head, and
    balanced_acc is its LOGO balanced accuracy (→ static prior via accuracy_weights; chance→0).
    qualities: optional dict[node_id -> (n,) live quality], e.g. per-window max-proba margin.

    Returns {fused_accuracy, per_link_accuracy, weights, n}. O(n·L·C). If every link is at/below
    chance (all weights 0) it falls back to a uniform blend so the vote is still defined."""
    y = np.asarray(y, dtype=np.int64)
    ids = list(links)
    weights = accuracy_weights({nid: links[nid][1] for nid in ids})
    static = weights if any(w > 0 for w in weights.values()) else None  # uniform if all at chance
    voter = LinkVoter(static)
    fused = np.empty(y.size, dtype=np.int64)
    for i in range(y.size):
        for nid in ids:
            q = float(qualities[nid][i]) if qualities and nid in qualities else 1.0
            voter.add(nid, links[nid][0][i], quality=q)
        fused[i] = voter.finalize()[0]
    per_link = {nid: float((np.argmax(links[nid][0], axis=1) == y).mean()) for nid in ids}
    return {
        "fused_accuracy": float((fused == y).mean()),
        "per_link_accuracy": per_link,
        "weights": weights,
        "n": int(y.size),
    }


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
        # reset for reuse
        self._wsum = None
        self._total = 0.0
        self._C = None
        return result
