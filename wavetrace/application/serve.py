from collections.abc import Sequence

from wavetrace import GainLock, RecognitionResult
from wavetrace.application.behaviours import WindowBehaviour
from wavetrace.application.ports import CsiSource
from wavetrace.domain.calibration import CalibrationResult, build_image_baseline
from wavetrace.domain.dsp.frontend import iter_windows
from wavetrace.domain.recognition.input_plan import plan_input
from wavetrace.domain.recognition.verdict import decide_class


def infer_windows(source: CsiSource, calibration_result: CalibrationResult, gain_lock: GainLock | None, session,
                  mode, publisher, *, behaviours: Sequence[WindowBehaviour] = ()) -> list[RecognitionResult]:
    """session: has `.head` (a presence or weapon head) and `.predict_window_proba(row) -> (C,)`.
    Publishes one verdict per window (one forward pass each) and each behaviour's events right after it,
    then each behaviour's extra rows after the last window.
    O(windows)."""
    should_apply_lock, intercarrier, pick = plan_input(mode, session.head)
    config = session.head.config
    classes = session.head.classes_
    image_baseline = None
    if config.subtract_baseline:
        image_baseline = build_image_baseline(calibration_result,
                                              locked=(should_apply_lock and gain_lock is not None))
    # the raw baseline, inter-carrier path only, exactly as training subtracted it
    intercarrier_baseline = (calibration_result.baseline_mag
                             if getattr(config, "subtract_ic_baseline", False) else None)

    results = []
    for timestamp, features, image, intercarrier_features in iter_windows(
        source.frames(), calibration_result.subcarriers, gain_lock if should_apply_lock else None,
        window=config.window, hop=config.hop, intercarrier=intercarrier,
        image_subcarriers=getattr(calibration_result, "image_subcarriers", None),
        frame_average=config.frame_average, image_baseline=image_baseline,
        intercarrier_baseline=intercarrier_baseline,
    ):
        proba = session.predict_window_proba(pick(features, image, intercarrier_features))
        class_id, confidence = decide_class(proba, classes)
        result_row = RecognitionResult()
        result_row.class_id = class_id
        result_row.confidence = confidence
        result_row.timestamp = timestamp
        publisher.publish(result_row)
        results.append(result_row)
        for behaviour in behaviours:
            event = behaviour.observe_window(timestamp, class_id, proba)
            if event:
                publisher.publish_event(event)
    for behaviour in behaviours:
        extra_row = behaviour.finish()
        if extra_row is not None:
            publisher.publish(extra_row)
            results.append(extra_row)
    return results
