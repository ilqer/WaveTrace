"""The collect use-case: a source + a label source -> a serialized labeled dataset."""

from wavetrace.calibration import load_calibration
from wavetrace.domain.contracts import DEFAULT_HOP_FRAMES, DEFAULT_WINDOW_FRAMES
from wavetrace.groundtruth import (
    build_dataset,
    presence_label_fn,
    save_dataset,
    weapon_label_fn,
)
from wavetrace.groundtruth.camera_labeler import ScriptedLabeler


def collect_source(source, calibration_dir, out_dir, spans, *, stage="presence",
                    window=DEFAULT_WINDOW_FRAMES, hop=DEFAULT_HOP_FRAMES,
                    session_id="", subject_id="", frame_average=1, subtract_baseline=False,
                    subtract_ic_baseline=False, labeler=None, tier=""):
    """Build + serialize a labeled dataset from a source + a label source. weapon stage emits the
    dual-block (intercarrier) dataset; presence emits the feature path.

    labeler: explicit label source (list[Label] / callable / Labeler) to use INSTEAD of the default
    scripted spans — pass a SegmentationLabeler/YoloSegLabeler (or a list of camera-produced Labels)
    to collect a mask-bearing, camera-supervised dataset that feeds the heatmap head. Defaults to a
    ScriptedLabeler over `spans` (the no-camera path).
    tier: 'open' | 'wrapped' | 'concealed', stamped into meta['tier'] so a concealed collection can be
    held out by evaluate_concealment_gap (the open->concealed transfer measurement)."""
    result, gain_lock = load_calibration(calibration_dir)
    if labeler is None:
        label_fn = weapon_label_fn if stage == "weapon" else presence_label_fn
        labeler = ScriptedLabeler([(start, end, True) for start, end in spans], label_fn=label_fn)
    intercarrier = stage == "weapon"
    # weapon IC/CNN paths need raw magnitudes: gain-locking cancels sigma2[p] (the metal discriminator)
    effective_lock = None if intercarrier else gain_lock
    dataset = build_dataset(list(source.frames()), result, effective_lock, labeler, window=window, hop=hop,
                            session_id=session_id, subject_id=subject_id, intercarrier=intercarrier,
                            frame_average=frame_average, subtract_baseline=subtract_baseline,
                            subtract_ic_baseline=subtract_ic_baseline)
    if tier:
        dataset.meta["tier"] = tier
    return save_dataset(dataset, out_dir), dataset
