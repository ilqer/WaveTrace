from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from wavetrace import GainLock, RecognitionResult
from wavetrace.application.behaviours import WindowBehaviour
from wavetrace.application.ports import CsiSource
from wavetrace.domain.calibration import CalibrationResult, build_image_baseline
from wavetrace.domain.dsp.frontend import iter_windows
from wavetrace.domain.recognition.input_plan import plan_input
from wavetrace.domain.recognition.verdict import decide_class


@dataclass(frozen=True, slots=True)
class WindowView:
    """One window's front-end output, as `iter_windows` yields it."""
    features: np.ndarray  # (9·K,)
    image: np.ndarray  # (K_img, window)
    intercarrier_features: np.ndarray | None  # (27,), None when the head reads no inter-carrier block


@dataclass(frozen=True, slots=True)
class WindowVerdict:
    timestamp_s: float  # window end
    class_id: int
    confidence: float
    proba: np.ndarray  # columns ordered by the head's `classes_`
    window_view: WindowView  # `iter_windows`' reused buffers: valid only until the iterator advances


def iter_window_verdicts(source: CsiSource, calibration_result: CalibrationResult, gain_lock: GainLock | None,
                         session, mode) -> Iterator[WindowVerdict]:
    """session: has `.head` (a presence or weapon head) and `.predict_window_proba(row) -> (C,)`. The
    head's contract picks the input, gain lock, frame averaging and baselines. One forward pass per
    window, O(windows)."""
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

    for timestamp, features, image, intercarrier_features in iter_windows(
        source.frames(), calibration_result.subcarriers, gain_lock if should_apply_lock else None,
        window=config.window, hop=config.hop, intercarrier=intercarrier,
        image_subcarriers=getattr(calibration_result, "image_subcarriers", None),
        frame_average=config.frame_average, image_baseline=image_baseline,
        intercarrier_baseline=intercarrier_baseline,
    ):
        proba = session.predict_window_proba(pick(features, image, intercarrier_features))
        class_id, confidence = decide_class(proba, classes)
        yield WindowVerdict(timestamp, class_id, confidence, proba,
                            WindowView(features, image, intercarrier_features))


def infer_windows(source: CsiSource, calibration_result: CalibrationResult, gain_lock: GainLock | None, session,
                  mode, publisher, *, behaviours: Sequence[WindowBehaviour] = ()) -> list[RecognitionResult]:
    """Publishes one verdict per `iter_window_verdicts` window and each behaviour's events right after it,
    then each behaviour's extra rows after the last window. O(windows)."""
    results = []
    for verdict in iter_window_verdicts(source, calibration_result, gain_lock, session, mode):
        result_row = RecognitionResult()
        result_row.class_id = verdict.class_id
        result_row.confidence = verdict.confidence
        result_row.timestamp = verdict.timestamp_s
        publisher.publish(result_row)
        results.append(result_row)
        for behaviour in behaviours:
            event = behaviour.observe_window(verdict.timestamp_s, verdict.class_id, verdict.proba)
            if event:
                publisher.publish_event(event)
    for behaviour in behaviours:
        extra_row = behaviour.finish()
        if extra_row is not None:
            publisher.publish(extra_row)
            results.append(extra_row)
    return results
