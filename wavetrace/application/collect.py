from wavetrace.calibration import load_calibration
from wavetrace.domain.contracts import DEFAULT_HOP_FRAMES, DEFAULT_WINDOW_FRAMES
from wavetrace.groundtruth import (
    build_dataset,
    presence_label_function,
    save_dataset,
    weapon_label_function,
)
from wavetrace.groundtruth.camera_labeler import ScriptedLabeler


def collect_source(source, calibration_dir, out_dir, spans, *, stage="presence",
                    window=DEFAULT_WINDOW_FRAMES, hop=DEFAULT_HOP_FRAMES,
                    session_id="", subject_id="", frame_average=1, subtract_baseline=False,
                    subtract_ic_baseline=False, labeler=None, tier=""):
    """The weapon stage emits the dual-block (intercarrier) dataset; presence emits the feature path.

    labeler: replaces the default ScriptedLabeler over `spans`; a camera labeler yields the mask-bearing
    dataset that feeds the heatmap head.
    tier: 'open' | 'wrapped' | 'concealed', stamped into meta['tier'] so evaluate_concealment_gap can
    hold a concealed collection out."""
    calibration_result, gain_lock = load_calibration(calibration_dir)
    if labeler is None:
        label_function = weapon_label_function if stage == "weapon" else presence_label_function
        labeler = ScriptedLabeler([(start, end, True) for start, end in spans], label_function=label_function)
    intercarrier = stage == "weapon"
    # weapon IC/CNN paths need raw magnitudes: gain-locking cancels sigma2[p] (the metal discriminator)
    effective_lock = None if intercarrier else gain_lock
    dataset = build_dataset(list(source.frames()), calibration_result, effective_lock, labeler,
                            window=window, hop=hop, session_id=session_id, subject_id=subject_id,
                            intercarrier=intercarrier, frame_average=frame_average,
                            subtract_baseline=subtract_baseline,
                            subtract_ic_baseline=subtract_ic_baseline)
    if tier:
        dataset.meta["tier"] = tier
    return save_dataset(dataset, out_dir), dataset
