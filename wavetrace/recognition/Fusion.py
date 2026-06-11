"""Phase 6e — multi-RX feature-level fusion: concatenate per-node feature vectors.

Separate RX nodes have INDEPENDENT clocks, so cross-node conjugate multiplication is physically
meaningless (§2.8) — fusion happens at the FEATURE level only: each node runs its own front-end
(gain-lock → NBVI → FeatureExtractor), `NodeAggregator` (Phase 2) stages the time-sync so one emit
holds the latest window per node, and `fuse` concatenates those vectors in a STABLE node order into
the single (Σ 9·Kᵢ,) vector the head consumes. O(m total features) per emit.

NOTE (user): Python glue for now — move to C++/a faster library when the multi-RX rig exists.
"""

import numpy as np


def fuse(node_features, out: np.ndarray | None = None) -> np.ndarray:
    """Concat per-node 1-D feature vectors [feat_node0, feat_node1, ...] -> (Σ dᵢ,) float32.

    Caller fixes the node order (e.g. sorted node id) and must keep it identical between training
    and inference — the head's weights are positional. `out` lets the per-emit call reuse one buffer
    (no hot-path allocation)."""
    arrs = [np.asarray(v, dtype=np.float32) for v in node_features]
    if not arrs:
        raise ValueError("fuse: no node features")
    for i, a in enumerate(arrs):
        if a.ndim != 1:
            raise ValueError(f"fuse: node {i} feature vector must be 1-D, got shape {a.shape}")
    total = sum(a.size for a in arrs)
    if out is not None and (out.ndim != 1 or out.size != total or out.dtype != np.float32):
        raise ValueError(f"fuse: out must be float32 (={total},), got {out.dtype} {out.shape}")
    return np.concatenate(arrs, out=out)
