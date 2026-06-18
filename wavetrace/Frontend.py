"""Phase 8 — the single source of front-end truth, shared by training (build_dataset) and serving
(Cli.run) so the served model sees EXACTLY the features it trained on.

`iter_windows` streams a CSI recording through the P4 front-end and yields one tuple per emitted
window. The per-frame logic mirrors the dual-block contract: the inter-carrier block always sees
RAW (pre-lock) magnitudes (the gain lock cancels the cross-subcarrier flatness the metal signature
lives in), while the feature/image path sees the gain-locked magnitudes. Emit cadence: once the
window fills, then every `hop` frames — all three extractors run in lockstep.

`frame_average` (T2/P10): non-overlapping decimating mean (LUMS moving-metal trick). M=1 (default)
is special-cased to the original byte-identical path. For M>1 each group of M real frames is
collapsed to one virtual frame; incomplete tail groups are dropped; effective fs = fs/M.

`image_subcarriers` (T1/P10): when set, the image path uses these subcarrier indices (all valid,
frequency-ordered) instead of the NBVI set. Feature path always uses NBVI `subcarriers`. Dual
SpectrogramBuilder when the two sets differ. `image_baseline` (T3/P10, in the image path's basis)
is subtracted per virtual frame when provided (image-path only; features and IC untouched).

The yielded arrays for frame_average=1 are the extractors' REUSED zero-copy buffers: copy before
advancing the iterator if you retain them. For frame_average>1, features and IC are also reused
buffers; image is reused. build_dataset copies each. O(n log n) per emit.

`iter_windows_stacked` (T4/P10): lockstep-zip one `iter_windows` per node, yield channel-stacked
(N, K_img, window) images and (N·9·K,) feature vectors. `demux_by_node` splits an interleaved
stream by node_id.
"""

import numpy as np

from wavetrace import FeatureExtractor, InterCarrierExtractor, SpectrogramBuilder


def iter_windows(frames, subcarriers, gain_lock, *, window=128, hop=32, intercarrier=False,
                 image_subcarriers=None, frame_average=1, image_baseline=None):
    """Yield (t, features, image, ic) per emitted window over `frames`.

    subcarriers: NBVI subcarrier indices (K, the feature series).
    gain_lock: locked GainLock or None (no rescale).
    intercarrier: emit the 27-feature IC block from raw mags; ic=None when False.
    image_subcarriers: if set, image rows use these subcarrier indices (all-valid, freq order) instead
      of the NBVI set. None -> image uses `subcarriers` (byte-identical to pre-T1 behavior).
    frame_average: M>=1 decimating mean (M=1 -> existing path, byte-identical). Effective fs=fs/M.
    image_baseline: (S,) float32 baseline in the image path's amplitude basis. When set, subtracted
      from each virtual frame's image values before pushing. Features and IC are not affected. Must be
      pre-computed by the caller via Calibration.image_baseline(). Applied after frame averaging.

    Yields: t (float window-END timestamp), features (9·K,), image (K_img, window), ic (27,) or None.
    """
    if frame_average < 1:
        raise ValueError("frame_average must be >= 1")

    subc = np.asarray(subcarriers, dtype=np.intp)
    K = int(subc.size)
    fe = FeatureExtractor(num_series=K, window=window, hop=hop)

    if image_subcarriers is not None:
        img_subc = np.asarray(image_subcarriers, dtype=np.intp)
        K_img = int(img_subc.size)
        sg = SpectrogramBuilder(num_subcarriers=K_img, time_steps=window, hop=hop)
    else:
        img_subc = subc
        K_img = K
        sg = SpectrogramBuilder(num_subcarriers=K, time_steps=window, hop=hop)

    ic = InterCarrierExtractor(window=window, hop=hop) if intercarrier else None

    # precompute baseline slice (once, before the loop) to avoid repeated indexing
    img_base = (np.ascontiguousarray(image_baseline[img_subc], dtype=np.float32)
                if image_baseline is not None else None)
    sub_buf = np.empty(K_img, dtype=np.float32) if img_base is not None else None

    if frame_average == 1:
        # M=1: exact original code path — byte-identical to pre-P10 behavior.
        for fr in frames:
            if ic is not None:
                ic_emitted = ic.push(np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32))
            if gain_lock is not None:
                gain_lock.apply(fr)
            mags = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)
            vals = np.ascontiguousarray(mags[subc])
            if image_subcarriers is not None:
                vals_img = np.ascontiguousarray(mags[img_subc])
            else:
                vals_img = vals
            emitted = fe.push(vals)
            if img_base is not None:
                np.subtract(vals_img, img_base, out=sub_buf)
                sg_emitted = sg.push(sub_buf)
            else:
                sg_emitted = sg.push(vals_img)
            if ic is None:
                ic_emitted = emitted
            if emitted:
                assert sg_emitted and ic_emitted, "front-end emit cadence diverged"
                yield (float(fr.timestamp), fe.features, sg.image,
                       ic.features if ic is not None else None)
    else:
        # M>1: non-overlapping decimating mean (LUMS temporal averaging).
        # Two preallocated (S,) float32 accumulators; np.add(..., out=...) avoids per-frame alloc.
        # Incomplete tail group (F % M frames) is dropped. Effective fs = fs/M.
        raw_acc = locked_acc = None
        count = 0
        last_ts = 0.0

        for fr in frames:
            raw_mags = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)
            if raw_acc is None:
                S = raw_mags.size
                raw_acc = np.zeros(S, dtype=np.float32)
                locked_acc = np.zeros(S, dtype=np.float32)
            if ic is not None:
                np.add(raw_acc, raw_mags, out=raw_acc)
            if gain_lock is not None:
                gain_lock.apply(fr)
            locked_mags = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)
            np.add(locked_acc, locked_mags, out=locked_acc)
            count += 1
            last_ts = float(fr.timestamp)

            if count == frame_average:
                locked_acc /= frame_average
                if ic is not None:
                    raw_acc /= frame_average

                vals = np.ascontiguousarray(locked_acc[subc])
                if image_subcarriers is not None:
                    vals_img = np.ascontiguousarray(locked_acc[img_subc])
                else:
                    vals_img = vals
                emitted = fe.push(vals)
                if img_base is not None:
                    np.subtract(vals_img, img_base, out=sub_buf)
                    sg_emitted = sg.push(sub_buf)
                else:
                    sg_emitted = sg.push(vals_img)
                if ic is not None:
                    ic_emitted = ic.push(raw_acc)
                else:
                    ic_emitted = emitted
                if emitted:
                    assert sg_emitted and ic_emitted, "front-end emit cadence diverged"
                    yield (last_ts, fe.features, sg.image,
                           ic.features if ic is not None else None)

                raw_acc[:] = 0.0
                locked_acc[:] = 0.0
                count = 0


def demux_by_node(frames) -> dict:
    """Split an interleaved CsiFrame stream by fr.node_id, capture order preserved. O(F)."""
    result: dict = {}
    for fr in frames:
        nid = int(fr.node_id)
        if nid not in result:
            result[nid] = []
        result[nid].append(fr)
    return result


def iter_windows_stacked(per_node_frames, per_node_calib, *, window=128, hop=32,
                         intercarrier=False, frame_average=1, node_tolerance=0.05):
    """Lockstep-zip one iter_windows per node; yield channel-stacked windows. Offline. O(N·n log n).

    per_node_frames: dict[node_id -> frame iterable].
    per_node_calib: dict[node_id -> (subcarriers, image_subcarriers, gain_lock, image_baseline|None)].
    All nodes must share K and K_img (ValueError otherwise).
    Node/channel order = sorted node ids.

    Yields (t, features, image, ic):
      t = window-end timestamp of the LOWEST node id.
      features = (N·9·K,) float32 concatenation across nodes.
      image = (N, K_img, window) float32 np.stack across nodes.
      ic = (N·27,) float32 concatenation or None when intercarrier=False.

    Stacked outputs are NEW arrays (np.stack/concatenate copy) — safe to retain, unlike iter_windows
    whose buffers are reused per emit. Stops at the shortest node stream; no error on unequal lengths.
    Raises ValueError when timestamps diverge > node_tolerance (node de-sync) or K/K_img mismatch.
    """
    node_ids = sorted(per_node_calib.keys())
    N = len(node_ids)
    if N == 0:
        return

    # Validate K and K_img consistency before starting iteration
    k_list, k_img_list = [], []
    for nid in node_ids:
        subc, img_subc, _, _ = per_node_calib[nid]
        k_list.append(len(subc))
        img = img_subc if img_subc is not None else subc
        k_img_list.append(len(img))
    if len(set(k_list)) != 1:
        raise ValueError(f"iter_windows_stacked: nodes have different K: {k_list}")
    if len(set(k_img_list)) != 1:
        raise ValueError(f"iter_windows_stacked: nodes have different K_img: {k_img_list}")

    gens = []
    for nid in node_ids:
        subc, img_subc, lock, base = per_node_calib[nid]
        gens.append(iter_windows(
            per_node_frames[nid], subc, lock, window=window, hop=hop,
            intercarrier=intercarrier, image_subcarriers=img_subc,
            frame_average=frame_average, image_baseline=base,
        ))

    while True:
        items = []
        for g in gens:
            try:
                items.append(next(g))
            except StopIteration:
                return

        ts = [item[0] for item in items]
        if max(ts) - min(ts) > node_tolerance:
            raise ValueError(
                f"recording is not node-synced: max timestamp gap "
                f"{max(ts) - min(ts):.4f}s > tolerance {node_tolerance}s"
            )

        t = items[0][0]  # lowest node id's timestamp
        features = np.concatenate([np.asarray(item[1], dtype=np.float32) for item in items])
        image = np.stack([np.asarray(item[2], dtype=np.float32) for item in items])
        ic = (np.concatenate([np.asarray(item[3], dtype=np.float32) for item in items])
              if intercarrier else None)
        yield (t, features, image, ic)
