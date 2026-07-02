"""Phase 8 — the single source of front-end truth, shared by training (buildDataset) and serving
(Cli.run) so the served model sees EXACTLY the features it trained on.

`iterWindows` streams a CSI recording through the P4 front-end and yields one tuple per emitted
window. The per-frame logic mirrors the dual-block contract: the inter-carrier block always sees
RAW (pre-lock) magnitudes (the gain lock cancels the cross-subcarrier flatness the metal signature
lives in), while the feature/image path sees the gain-locked magnitudes. Emit cadence: once the
window fills, then every `hop` frames — all three extractors run in lockstep.

`frame_average` (T2/P10): non-overlapping decimating mean (LUMS moving-metal trick). M=1 (default)
is special-cased to the original byte-identical path. For M>1 each group of M real frames is
collapsed to one virtual frame; incomplete tail groups are dropped; effective fs = fs/M.

`image_subcarriers` (T1/P10): when set, the image path uses these subcarrier indices (all valid,
frequency-ordered) instead of the NBVI set. Feature path always uses NBVI `subcarriers`. Dual
SpectrogramBuilder when the two sets differ. `imageBaseline` (T3/P10, in the image path's basis)
is subtracted per virtual frame when provided (image-path only; features and IC untouched).

The yielded arrays for frame_average=1 are the extractors' REUSED zero-copy buffers: copy before
advancing the iterator if you retain them. For frame_average>1, features and IC are also reused
buffers; image is reused. buildDataset copies each. O(n log n) per emit.

`iterWindowsStacked` (T4/P10): lockstep-zip one `iterWindows` per node, yield channel-stacked
(N, K_img, window) images and (N·9·K,) feature vectors. `demuxByNode` splits an interleaved
stream by node_id.
"""

import numpy as np

from wavetrace import FeatureExtractor, InterCarrierExtractor, SpectrogramBuilder


def iterWindows(frames, subcarriers, gainLock, *, window=128, hop=32, intercarrier=False,
                 image_subcarriers=None, frame_average=1, imageBaseline=None, ic_baseline=None):
    """Yield (t, features, image, ic) per emitted window over `frames`.

    subcarriers: NBVI subcarrier indices (K, the feature series).
    gainLock: locked GainLock or None (no rescale).
    intercarrier: emit the 27-feature IC block from raw mags; ic=None when False.
    ic_baseline: (S,) float32 quiet-room |H| per subcarrier (calibration.baseline_mag). When set, it
      is subtracted from each frame's raw magnitude BEFORE the IC block (diagnosis CAUSE 2B weapon
      background subtraction): σ²[p] then measures the variance of the PERTURBATION (h_live−h_baseline,
      static room nulled), not of the full channel. Per-subcarrier (not a scalar), so it genuinely
      reshapes σ²[p]. Only affects the IC path; features/image are unchanged. Must match training.
    image_subcarriers: if set, image rows use these subcarrier indices (all-valid, freq order) instead
      of the NBVI set. None -> image uses `subcarriers` (byte-identical to pre-T1 behavior).
    frame_average: M>=1 decimating mean (M=1 -> existing path, byte-identical). Effective fs=fs/M.
    imageBaseline: (S,) float32 baseline in the image path's amplitude basis. When set, subtracted
      from each virtual frame's image values before pushing. Features and IC are not affected. Must be
      pre-computed by the caller via Calibration.imageBaseline(). Applied after frame averaging.

    Yields: t (float window-END timestamp), features (9·K,), image (K_img, window), ic (27,) or None.
    """
    if frame_average < 1:
        raise ValueError("frame_average must be >= 1")

    subc = np.asarray(subcarriers, dtype=np.intp)
    K = int(subc.size)
    fe = FeatureExtractor(num_series=K, window=window, hop=hop)

    if image_subcarriers is not None:
        imgSubc = np.asarray(image_subcarriers, dtype=np.intp)
        KImg = int(imgSubc.size)
        sg = SpectrogramBuilder(num_subcarriers=KImg, time_steps=window, hop=hop)
    else:
        imgSubc = subc
        KImg = K
        sg = SpectrogramBuilder(num_subcarriers=K, time_steps=window, hop=hop)

    ic = InterCarrierExtractor(window=window, hop=hop) if intercarrier else None

    # precompute baseline slice (once, before the loop) to avoid repeated indexing
    imgBase = (np.ascontiguousarray(imageBaseline[imgSubc], dtype=np.float32)
                if imageBaseline is not None else None)
    subBuf = np.empty(KImg, dtype=np.float32) if imgBase is not None else None
    # full-S IC baseline (not sliced to subc — the IC block consumes the whole frame, Frontend §76)
    icBase = np.ascontiguousarray(ic_baseline, dtype=np.float32) if ic_baseline is not None else None

    if frame_average == 1:
        # M=1: exact original code path — byte-identical to pre-P10 behavior.
        for fr in frames:
            if ic is not None:
                icMag = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)
                if icBase is not None:
                    if icMag.shape != icBase.shape:
                        raise ValueError(f"ic_baseline width {icBase.shape} != frame width "
                                         f"{icMag.shape}; calibration and capture must share width")
                    icMag = icMag - icBase  # null the static room before σ²[p]
                icEmitted = ic.push(icMag)
            if gainLock is not None:
                gainLock.apply(fr)
            mags = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)
            vals = np.ascontiguousarray(mags[subc])
            if image_subcarriers is not None:
                valsImg = np.ascontiguousarray(mags[imgSubc])
            else:
                valsImg = vals
            emitted = fe.push(vals)
            if imgBase is not None:
                np.subtract(valsImg, imgBase, out=subBuf)
                sgEmitted = sg.push(subBuf)
            else:
                sgEmitted = sg.push(valsImg)
            if ic is None:
                icEmitted = emitted
            if emitted:
                assert sgEmitted and icEmitted, "front-end emit cadence diverged"
                yield (float(fr.timestamp), fe.features, sg.image,
                       ic.features if ic is not None else None)
    else:
        # M>1: non-overlapping decimating mean (LUMS); preallocated accumulators avoid per-frame alloc.
        rawAcc = lockedAcc = None
        count = 0
        lastTs = 0.0

        for fr in frames:
            rawMags = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)
            if rawAcc is None:
                S = rawMags.size
                rawAcc = np.zeros(S, dtype=np.float32)
                lockedAcc = np.zeros(S, dtype=np.float32)
            if ic is not None:
                np.add(rawAcc, rawMags, out=rawAcc)
            if gainLock is not None:
                gainLock.apply(fr)
            lockedMags = np.abs(np.asarray(fr.grid)).mean(axis=0).astype(np.float32)
            np.add(lockedAcc, lockedMags, out=lockedAcc)
            count += 1
            lastTs = float(fr.timestamp)

            if count == frame_average:
                lockedAcc /= frame_average
                if ic is not None:
                    rawAcc /= frame_average

                vals = np.ascontiguousarray(lockedAcc[subc])
                if image_subcarriers is not None:
                    valsImg = np.ascontiguousarray(lockedAcc[imgSubc])
                else:
                    valsImg = vals
                emitted = fe.push(vals)
                if imgBase is not None:
                    np.subtract(valsImg, imgBase, out=subBuf)
                    sgEmitted = sg.push(subBuf)
                else:
                    sgEmitted = sg.push(valsImg)
                if ic is not None:
                    if icBase is not None and rawAcc.shape != icBase.shape:
                        raise ValueError(f"ic_baseline width {icBase.shape} != frame width "
                                         f"{rawAcc.shape}; calibration and capture must share width")
                    icEmitted = ic.push(rawAcc - icBase if icBase is not None else rawAcc)
                else:
                    icEmitted = emitted
                if emitted:
                    assert sgEmitted and icEmitted, "front-end emit cadence diverged"
                    yield (lastTs, fe.features, sg.image,
                           ic.features if ic is not None else None)

                rawAcc[:] = 0.0
                lockedAcc[:] = 0.0
                count = 0


def demuxByNode(frames) -> dict:
    """Split an interleaved CsiFrame stream by fr.node_id, capture order preserved. O(F)."""
    result: dict = {}
    for fr in frames:
        nid = int(fr.node_id)
        if nid not in result:
            result[nid] = []
        result[nid].append(fr)
    return result


def iterWindowsStacked(per_node_frames, per_node_calib, *, window=128, hop=32,
                         intercarrier=False, frame_average=1, node_tolerance=0.05):
    """Lockstep-zip one iterWindows per node; yield channel-stacked windows. Offline. O(N·n log n).

    per_node_frames: dict[node_id -> frame iterable].
    per_node_calib: dict[node_id -> (subcarriers, image_subcarriers, gainLock, imageBaseline|None)].
    All nodes must share K and K_img (ValueError otherwise).
    Node/channel order = sorted node ids.

    Yields (t, features, image, ic):
      t = window-end timestamp of the LOWEST node id.
      features = (N·9·K,) float32 concatenation across nodes.
      image = (N, K_img, window) float32 np.stack across nodes.
      ic = (N·27,) float32 concatenation or None when intercarrier=False.

    Stacked outputs are NEW arrays (np.stack/concatenate copy) — safe to retain, unlike iterWindows
    whose buffers are reused per emit. Stops at the shortest node stream; no error on unequal lengths.
    Raises ValueError when timestamps diverge > node_tolerance (node de-sync) or K/K_img mismatch.
    """
    nodeIds = sorted(per_node_calib.keys())
    N = len(nodeIds)
    if N == 0:
        return

    kList, kImgList = [], []
    for nid in nodeIds:
        subc, imgSubc, _, _ = per_node_calib[nid]
        kList.append(len(subc))
        img = imgSubc if imgSubc is not None else subc
        kImgList.append(len(img))
    if len(set(kList)) != 1:
        raise ValueError(f"iterWindowsStacked: nodes have different K: {kList}")
    if len(set(kImgList)) != 1:
        raise ValueError(f"iterWindowsStacked: nodes have different K_img: {kImgList}")

    gens = []
    for nid in nodeIds:
        subc, imgSubc, lock, base = per_node_calib[nid]
        gens.append(iterWindows(
            per_node_frames[nid], subc, lock, window=window, hop=hop,
            intercarrier=intercarrier, image_subcarriers=imgSubc,
            frame_average=frame_average, imageBaseline=base,
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
