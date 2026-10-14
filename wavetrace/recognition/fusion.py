"""Fuse several RX nodes at the feature level: concatenate their vectors in a stable order.

The nodes have independent clocks, so anything finer than feature-level concatenation would be
combining clock noise. O(features) per emit.
"""

import numpy as np


def fuse(node_features, out: np.ndarray | None = None) -> np.ndarray:
    """`node_features` order must match training; `out` (float32, exact size) avoids an allocation."""
    vectors = [np.asarray(features, dtype=np.float32) for features in node_features]
    if not vectors:
        raise ValueError("fuse: no node features")
    for i, vector in enumerate(vectors):
        if vector.ndim != 1:
            raise ValueError(f"fuse: node {i} feature vector must be 1-D, got shape {vector.shape}")
    total = sum(vector.size for vector in vectors)
    if out is not None and (out.ndim != 1 or out.size != total or out.dtype != np.float32):
        raise ValueError(f"fuse: out must be float32 (={total},), got {out.dtype} {out.shape}")
    return np.concatenate(vectors, out=out)
