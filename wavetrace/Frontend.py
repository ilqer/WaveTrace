"""Phase 8 — the single source of front-end truth, shared by training (build_dataset) and serving
(Cli.run) so the served model sees EXACTLY the features it trained on.

`iter_windows` streams a CSI recording through the P4 front-end and yields one tuple per emitted
window. The per-frame logic mirrors the dual-block contract: the inter-carrier block always sees
RAW (pre-lock) magnitudes (the gain lock cancels the cross-subcarrier flatness the metal signature
lives in), while the feature/image path sees the gain-locked magnitudes. Emit cadence: once the
window fills, then every `hop` frames — all three extractors run in lockstep.

The yielded arrays are the extractors' REUSED zero-copy buffers: copy before advancing the iterator
if you retain them (build_dataset does; run consumes each in place). O(n log n) per emit.
"""

import numpy as np

from wavetrace import FeatureExtractor, InterCarrierExtractor, SpectrogramBuilder


def iter_windows(frames, subcarriers, gain_lock, *, window=128, hop=32, intercarrier=False):
    """Yield (t, features, image, ic) per emitted window over `frames`.

    subcarriers: NBVI subcarrier indices (the K feature/image series).
    gain_lock: locked GainLock applied in-place to the feature/image path, or None to skip it.
    intercarrier: also emit the (n, 27) InterCarrierExtractor block over ALL subcarriers (from raw
      magnitudes); `ic` is None when False. Set True for the σ²[p] / fusion weapon paths.

    Yields: t (float window-END timestamp), features (9·K,), image (K, window), ic (27,) or None.
    """
    subc = np.asarray(subcarriers, dtype=np.intp)
    K = int(subc.size)
    fe = FeatureExtractor(num_series=K, window=window, hop=hop)
    sg = SpectrogramBuilder(num_subcarriers=K, time_steps=window, hop=hop)
    ic = InterCarrierExtractor(window=window, hop=hop) if intercarrier else None

    for fr in frames:
        if ic is not None:
            # IC must see raw (pre-lock) magnitudes — push before gain_lock.apply
            ic_emitted = ic.push(np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32))
        if gain_lock is not None:
            gain_lock.apply(fr)  # in-place amplitude rescale to the locked reference (phase preserved)
        mags = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)  # (S,) antenna-averaged
        vals = np.ascontiguousarray(mags[subc])                            # (K,) NBVI subcarriers
        emitted = fe.push(vals)
        sg_emitted = sg.push(vals)  # same window/hop -> emits in lockstep with fe
        if ic is None:
            ic_emitted = emitted
        if emitted:
            assert sg_emitted and ic_emitted, "front-end emit cadence diverged"
            yield (float(fr.timestamp), fe.features, sg.image, ic.features if ic is not None else None)
