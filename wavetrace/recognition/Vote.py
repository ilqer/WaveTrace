"""Phase 7p-e: Soft majority voting over presence event.

Accumulates class probabilities during an active segment; verdict is argmax of mean.
Voting helps moving subjects, not static windows.
Options: mid-segment extraction and window decimation.

O(1) per add, O(votes) finalize."""

import numpy as np


class SegmentVoter:
    """Accumulate per-window probabilities for ONE segment; finalize() votes and resets."""

    def __init__(self, *, middle_fraction: float = 1.0, decimate: int = 1,
                 confidence_weighted: bool = False):
        if not 0.0 < middle_fraction <= 1.0:
            raise ValueError("middle_fraction must be in (0, 1]")
        if decimate < 1:
            raise ValueError("decimate must be >= 1")
        self._mid = float(middle_fraction)
        self._step = int(decimate)
        self._weighted = bool(confidence_weighted)
        self._probas: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._probas)

    def add(self, proba) -> None:
        """Add class-probability vector."""
        p = np.asarray(proba, dtype=np.float64).ravel()
        if self._probas and p.size != self._probas[0].size:
            raise ValueError(f"class count changed mid-segment: {p.size} vs {self._probas[0].size}")
        self._probas.append(p)

    def finalize(self) -> tuple[int, np.ndarray]:
        """Close segment: returns (argmax class index, mean probability vector)."""
        if not self._probas:
            raise ValueError("SegmentVoter: no votes in this segment")
        n = len(self._probas)
        keep = max(1, int(round(self._mid * n)))
        start = (n - keep) // 2
        votes = self._probas[start:start + keep:self._step]  # middle slice, then decimate
        V = np.asarray(votes)                      # (m, C)
        if self._weighted:
            w = V.max(axis=1, keepdims=True)       # per-window confidence as weight
            total = float(w.sum())
            mean = (V * w).sum(axis=0) / max(total, 1e-9)
        else:
            mean = V.mean(axis=0)
        self._probas.clear()
        return int(np.argmax(mean)), mean

    def reset(self) -> None:
        self._probas.clear()
