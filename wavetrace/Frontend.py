"""One front-end for training and for serving, so a served model sees the features it trained on.

`iterWindows` streams a CSI recording and yields one tuple per emitted window: the window fills,
then one emit every `hop` frames, with the feature, image and inter-carrier extractors in lockstep.
O(n log n) per emit.

The inter-carrier block always reads RAW, pre-lock magnitudes - the gain lock cancels the
cross-subcarrier flatness the metal signature lives in - while the feature and image paths read
gain-locked magnitudes.

The arrays yielded are the extractors' reused zero-copy buffers: copy them before advancing the
iterator if you keep them. `iterWindowsStacked` yields fresh arrays instead.
"""

import numpy as np

from wavetrace import FeatureExtractor, InterCarrierExtractor, SpectrogramBuilder


def iterWindows(frames, subcarriers, gainLock, *, window=128, hop=32, intercarrier=False,
                 image_subcarriers=None, frame_average=1, imageBaseline=None, ic_baseline=None):
    """Yield (t, features, image, ic) per emitted window over `frames`.

    subcarriers: NBVI subcarrier indices (K, the feature series).
    gainLock: a locked GainLock, or None to leave amplitudes alone.
    intercarrier: emit the 27-feature inter-carrier block from raw magnitudes; ic=None when False.
    ic_baseline: (S,) quiet-room |H| per subcarrier, subtracted from each raw magnitude before the
      inter-carrier block, so σ²[p] measures the variance of the perturbation rather than of the
      whole channel with the static room still in it. Must match what training used.
    image_subcarriers: subcarrier indices for the image rows; None reuses `subcarriers`.
    frame_average: M >= 1 non-overlapping decimating mean. Each group of M frames collapses into one
      virtual frame, an incomplete tail group is dropped, and the effective rate is fs/M.
    imageBaseline: (S,) baseline in the image path's amplitude basis, subtracted per virtual frame
      after averaging. Image path only; build it with Calibration.build_image_baseline().

    Yields: t (window-END timestamp), features (9·K,), image (K_img, window), ic (27,) or None.
    """
    if frame_average < 1:
        raise ValueError("frame_average must be >= 1")

    subcarrier_indices = np.asarray(subcarriers, dtype=np.intp)
    num_feature_subcarriers = int(subcarrier_indices.size)
    feature_extractor = FeatureExtractor(num_series=num_feature_subcarriers, window=window, hop=hop)

    if image_subcarriers is not None:
        image_subcarrier_indices = np.asarray(image_subcarriers, dtype=np.intp)
        num_image_subcarriers = int(image_subcarrier_indices.size)
        spectrogram = SpectrogramBuilder(num_subcarriers=num_image_subcarriers, time_steps=window, hop=hop)
    else:
        image_subcarrier_indices = subcarrier_indices
        num_image_subcarriers = num_feature_subcarriers
        spectrogram = SpectrogramBuilder(num_subcarriers=num_feature_subcarriers, time_steps=window, hop=hop)

    intercarrier_extractor = InterCarrierExtractor(window=window, hop=hop) if intercarrier else None

    # slice the baseline once here rather than per frame
    image_baseline_rows = (
        np.ascontiguousarray(imageBaseline[image_subcarrier_indices], dtype=np.float32)
        if imageBaseline is not None else None
    )
    image_row_buffer = (np.empty(num_image_subcarriers, dtype=np.float32)
                        if image_baseline_rows is not None else None)
    # not sliced to subcarrier_indices: the inter-carrier block consumes the whole frame
    ic_baseline_values = (np.ascontiguousarray(ic_baseline, dtype=np.float32)
                          if ic_baseline is not None else None)

    if frame_average == 1:
        for frame in frames:
            if intercarrier_extractor is not None:
                raw_magnitudes = np.abs(np.asarray(frame.grid)).mean(axis=0).astype(np.float32)
                if ic_baseline_values is not None:
                    if raw_magnitudes.shape != ic_baseline_values.shape:
                        raise ValueError(f"ic_baseline width {ic_baseline_values.shape} != frame width "
                                         f"{raw_magnitudes.shape}; calibration and capture must share width")
                    # null the static room before σ²[p]
                    raw_magnitudes = raw_magnitudes - ic_baseline_values
                intercarrier_emitted = intercarrier_extractor.push(raw_magnitudes)
            if gainLock is not None:
                gainLock.apply(frame)
            magnitudes = np.abs(np.asarray(frame.grid)).mean(axis=0).astype(np.float32)
            feature_values = np.ascontiguousarray(magnitudes[subcarrier_indices])
            if image_subcarriers is not None:
                image_values = np.ascontiguousarray(magnitudes[image_subcarrier_indices])
            else:
                image_values = feature_values
            feature_emitted = feature_extractor.push(feature_values)
            if image_baseline_rows is not None:
                np.subtract(image_values, image_baseline_rows, out=image_row_buffer)
                image_emitted = spectrogram.push(image_row_buffer)
            else:
                image_emitted = spectrogram.push(image_values)
            if intercarrier_extractor is None:
                intercarrier_emitted = feature_emitted
            if feature_emitted:
                assert image_emitted and intercarrier_emitted, "front-end emit cadence diverged"
                yield (float(frame.timestamp), feature_extractor.features, spectrogram.image,
                       intercarrier_extractor.features if intercarrier_extractor is not None else None)
    else:
        # accumulators are preallocated: this runs per frame
        raw_accumulator = locked_accumulator = None
        frames_in_group = 0
        last_timestamp = 0.0

        for frame in frames:
            raw_magnitudes = np.abs(np.asarray(frame.grid)).mean(axis=0).astype(np.float32)
            if raw_accumulator is None:
                num_subcarriers = raw_magnitudes.size
                raw_accumulator = np.zeros(num_subcarriers, dtype=np.float32)
                locked_accumulator = np.zeros(num_subcarriers, dtype=np.float32)
            if intercarrier_extractor is not None:
                np.add(raw_accumulator, raw_magnitudes, out=raw_accumulator)
            if gainLock is not None:
                gainLock.apply(frame)
            locked_magnitudes = np.abs(np.asarray(frame.grid)).mean(axis=0).astype(np.float32)
            np.add(locked_accumulator, locked_magnitudes, out=locked_accumulator)
            frames_in_group += 1
            last_timestamp = float(frame.timestamp)

            if frames_in_group == frame_average:
                locked_accumulator /= frame_average
                if intercarrier_extractor is not None:
                    raw_accumulator /= frame_average

                feature_values = np.ascontiguousarray(locked_accumulator[subcarrier_indices])
                if image_subcarriers is not None:
                    image_values = np.ascontiguousarray(locked_accumulator[image_subcarrier_indices])
                else:
                    image_values = feature_values
                feature_emitted = feature_extractor.push(feature_values)
                if image_baseline_rows is not None:
                    np.subtract(image_values, image_baseline_rows, out=image_row_buffer)
                    image_emitted = spectrogram.push(image_row_buffer)
                else:
                    image_emitted = spectrogram.push(image_values)
                if intercarrier_extractor is not None:
                    if ic_baseline_values is not None and raw_accumulator.shape != ic_baseline_values.shape:
                        raise ValueError(f"ic_baseline width {ic_baseline_values.shape} != frame width "
                                         f"{raw_accumulator.shape}; calibration and capture must share width")
                    intercarrier_emitted = intercarrier_extractor.push(
                        raw_accumulator - ic_baseline_values
                        if ic_baseline_values is not None else raw_accumulator
                    )
                else:
                    intercarrier_emitted = feature_emitted
                if feature_emitted:
                    assert image_emitted and intercarrier_emitted, "front-end emit cadence diverged"
                    yield (last_timestamp, feature_extractor.features, spectrogram.image,
                           intercarrier_extractor.features if intercarrier_extractor is not None else None)

                raw_accumulator[:] = 0.0
                locked_accumulator[:] = 0.0
                frames_in_group = 0


def demuxByNode(frames) -> dict:
    """Split an interleaved CsiFrame stream by node_id, capture order preserved. O(F)."""
    result: dict = {}
    for frame in frames:
        node_id = int(frame.node_id)
        if node_id not in result:
            result[node_id] = []
        result[node_id].append(frame)
    return result


def iterWindowsStacked(per_node_frames, per_node_calib, *, window=128, hop=32,
                         intercarrier=False, frame_average=1, node_tolerance=0.05):
    """Lockstep-zip one iterWindows per node; yield channel-stacked windows. O(N·n log n).

    per_node_frames: dict[node_id -> frame iterable].
    per_node_calib: dict[node_id -> (subcarriers, image_subcarriers, gainLock, imageBaseline|None)].
    Channel order is the sorted node ids, and every node must share K and K_img.

    Yields (t, features, image, ic):
      t = window-end timestamp of the lowest node id.
      features = (N·9·K,) concatenated across nodes.
      image = (N, K_img, window) stacked across nodes.
      ic = (N·27,) concatenated, or None when intercarrier=False.

    Stops at the shortest node stream. Raises ValueError when the nodes' timestamps diverge by more
    than node_tolerance (de-sync) or their widths disagree.
    """
    node_ids = sorted(per_node_calib.keys())
    num_nodes = len(node_ids)
    if num_nodes == 0:
        return

    feature_widths, image_widths = [], []
    for node_id in node_ids:
        subcarrier_indices, image_subcarrier_indices, _, _ = per_node_calib[node_id]
        feature_widths.append(len(subcarrier_indices))
        image_widths.append(len(image_subcarrier_indices if image_subcarrier_indices is not None
                                else subcarrier_indices))
    if len(set(feature_widths)) != 1:
        raise ValueError(f"iterWindowsStacked: nodes have different K: {feature_widths}")
    if len(set(image_widths)) != 1:
        raise ValueError(f"iterWindowsStacked: nodes have different K_img: {image_widths}")

    node_windows = []
    for node_id in node_ids:
        (subcarrier_indices, image_subcarrier_indices,
         node_gain_lock, node_image_baseline) = per_node_calib[node_id]
        node_windows.append(iterWindows(
            per_node_frames[node_id], subcarrier_indices, node_gain_lock, window=window, hop=hop,
            intercarrier=intercarrier, image_subcarriers=image_subcarrier_indices,
            frame_average=frame_average, imageBaseline=node_image_baseline,
        ))

    while True:
        windows = []
        for windows_of_one_node in node_windows:
            try:
                windows.append(next(windows_of_one_node))
            except StopIteration:
                return

        timestamps = [window[0] for window in windows]
        if max(timestamps) - min(timestamps) > node_tolerance:
            raise ValueError(
                f"recording is not node-synced: max timestamp gap "
                f"{max(timestamps) - min(timestamps):.4f}s > tolerance {node_tolerance}s"
            )

        timestamp = windows[0][0]
        features = np.concatenate([np.asarray(window[1], dtype=np.float32) for window in windows])
        image = np.stack([np.asarray(window[2], dtype=np.float32) for window in windows])
        intercarrier_features = (
            np.concatenate([np.asarray(window[3], dtype=np.float32) for window in windows])
            if intercarrier else None
        )
        yield (timestamp, features, image, intercarrier_features)
