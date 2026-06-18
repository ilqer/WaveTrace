"""Phase 7p-e — soft majority voting over one presence event (Zhou, "Detection of Suspicious
Objects Concealed by Walking Pedestrians").

During one PresenceSegmenter-bounded active segment, every emitted window gets classified and the
head's CLASS PROBABILITIES are accumulated; at segment close the verdict is argmax of the MEAN
probability (soft vote — same cost as hard majority, strictly better when calibrated). Zhou's
per-snapshot CNN was 51.1%; voting over the walk lifted it to 93.3%. ⚠️ The gain is large only in
the MOVING regime (decorrelated per-window views); near-identical static windows have correlated
errors and voting helps little (rev-5 note — expect the lift at tier 7c, not 7a/7b).

Zhou options reproduced: mid-segment extraction (middle_fraction — the pedestrian is crossing the
link in the middle of the segment, the edges are approach/leave) and every-other-window decimation
(decimate — adjacent windows overlap by window-hop frames, decimation de-correlates votes).

O(1) per add (running per-class sums would lose mid-segment selection, so windows are kept:
O(votes) memory, O(votes) finalize — votes per segment is small, seconds × emit rate).
"""

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
        """One window's class-probability vector (C,) from the mode session's head."""
        p = np.asarray(proba, dtype=np.float64).ravel()
        if self._probas and p.size != self._probas[0].size:
            raise ValueError(f"class count changed mid-segment: {p.size} vs {self._probas[0].size}")
        self._probas.append(p)

    def finalize(self) -> tuple[int, np.ndarray]:
        """Segment closed: (argmax class INDEX, mean probability vector) over the selected votes,
        then reset for the next segment. Map the index through the head's classes_.

        When confidence_weighted=True, each window's vote is weighted by its own peak probability
        (max over classes), so borderline windows count less than confident ones."""
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
