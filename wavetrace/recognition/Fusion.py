"""Phase 6e: Multi-RX feature-level fusion.

RX nodes have independent clocks; fuse at feature level. Concatenate per-node feature vectors in stable order. O(features) per emit."""

import numpy as np


def fuse(node_features, out: np.ndarray | None = None) -> np.ndarray:
    """Concatenate per-node 1-D feature vectors. Keep node order consistent."""
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
