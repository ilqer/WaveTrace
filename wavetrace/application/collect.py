from wavetrace import GainLock
from wavetrace.application.ports import CsiSource
from wavetrace.domain.calibration import CalibrationResult
from wavetrace.domain.groundtruth.dataset import Dataset, build_dataset
from wavetrace.domain.groundtruth.labeler import (
    ScriptedLabeler,
    presence_label_function,
    weapon_label_function,
)


def collect_dataset(source: CsiSource, calibration_result: CalibrationResult, gain_lock: GainLock | None, spans, *,
                    stage, window, hop, session_id, subject_id, frame_average, subtract_baseline,
                    subtract_ic_baseline, labeler, tier) -> Dataset:
    """The weapon stage emits the dual-block (intercarrier) dataset and ignores `gain_lock`; presence
    emits the feature path.

    labeler: replaces the default ScriptedLabeler over `spans`; a camera labeler yields the mask-bearing
    dataset that feeds the heatmap head.
    tier: 'open' | 'wrapped' | 'concealed', stamped into meta['tier'] so evaluate_concealment_gap can
    hold a concealed collection out."""
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
    return dataset
