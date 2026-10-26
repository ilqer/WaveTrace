import collections
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from wavetrace import GainLock, RecognitionResult
from wavetrace.application.behaviours import RuleViolationCounter, WindowBehaviour, iter_frames_satisfying_width
from wavetrace.application.ports import CsiSource
from wavetrace.domain.calibration import CalibrationResult, build_image_baseline
from wavetrace.domain.contracts import (
    DEFAULT_MIN_SAMPLE_RATE_FRACTION, PipelineContract, require_valid_min_sample_rate_fraction,
)
from wavetrace.domain.dsp.frontend import iter_windows
from wavetrace.domain.recognition.input_plan import plan_input
from wavetrace.domain.recognition.verdict import decide_class
from wavetrace.domain.rules import (
    BusinessRule, SampleRateMustBeWithinTolerance, SubcarrierWidthMustCoverCalibration,
    SubcarrierWidthMustMatchCalibration, SubcarrierWidthMustMatchContract, measure_delivered_rate_hz, require,
)


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


def _require_calibration_fits_head(calibration_result: CalibrationResult, contract: PipelineContract) -> BusinessRule | None:
    """The rule the stream's frames must satisfy, None when neither records a width. When both do, the
    calibration must have been made from the head's width, so the head's rule covers the calibration's."""
    calibrated_width = calibration_result.subcarrier_width
    if contract.subcarrier_width is None:
        return None if calibrated_width is None else SubcarrierWidthMustMatchCalibration(calibrated_width)
    require(SubcarrierWidthMustCoverCalibration.for_calibration(calibration_result), contract.subcarrier_width,
            "the model's capture width")
    if calibrated_width is not None:
        require(SubcarrierWidthMustMatchContract(contract.subcarrier_width), calibrated_width,
                "the calibration's capture width")
    return SubcarrierWidthMustMatchContract(contract.subcarrier_width)


def _iter_frames_remembering_timestamps(frames: Iterable, recent_timestamps_s: collections.deque) -> Iterator:
    """Yields `frames` unchanged, keeping the last `recent_timestamps_s.maxlen` timestamps. O(1) per frame."""
    for frame in frames:
        recent_timestamps_s.append(frame.timestamp)
        yield frame


def iter_window_verdicts(source: CsiSource, calibration_result: CalibrationResult, gain_lock: GainLock | None,
                         session, mode, *,
                         min_sample_rate_fraction: float = DEFAULT_MIN_SAMPLE_RATE_FRACTION
                         ) -> Iterator[WindowVerdict | None]:
    """session: has `.head` (a presence or weapon head) and `.predict_window_proba(row) -> (C,)`. The
    head's contract picks the input, gain lock, frame averaging and baselines. Frames of another width
    than the head's recorded capture width (else the calibration's) are dropped and counted. A calibration
    indexing past the head's width, or made from another one, raises `BusinessRuleViolationError` before
    the source is read. A window whose frames were delivered below `min_sample_rate_fraction` of the head's
    target rate is skipped and counted under `SampleRateMustBeWithinTolerance`. One forward pass per
    window, O(windows). Yields None for a skipped window so a caller that stops on a flag can check it
    while a slow stream scores nothing."""
    require_valid_min_sample_rate_fraction(min_sample_rate_fraction)
    contract = session.head.contract
    stream_width_rule = _require_calibration_fits_head(calibration_result, contract)
    rule_types = (SampleRateMustBeWithinTolerance,) if stream_width_rule is None else (
        type(stream_width_rule), SampleRateMustBeWithinTolerance)
    violations = RuleViolationCounter(rule_types)
    frames = source.frames()
    if stream_width_rule is not None:
        frames = iter_frames_satisfying_width(frames, stream_width_rule, violations)
    should_apply_lock, intercarrier, pick = plan_input(mode, session.head)
    config = session.head.config
    # a window spans window * frame_average raw frames, so their timestamps give its delivered rate
    recent_timestamps_s = collections.deque(maxlen=config.window * config.frame_average)
    frames = _iter_frames_remembering_timestamps(frames, recent_timestamps_s)
    sample_rate_rule = SampleRateMustBeWithinTolerance(contract.target_sample_rate_hz, min_sample_rate_fraction)
    classes = session.head.classes_
    image_baseline = None
    if config.subtract_baseline:
        image_baseline = build_image_baseline(calibration_result,
                                              locked=(should_apply_lock and gain_lock is not None))
    # the raw baseline, inter-carrier path only, exactly as training subtracted it
    intercarrier_baseline = (calibration_result.baseline_mag
                             if getattr(config, "subtract_ic_baseline", False) else None)

    for timestamp, features, image, intercarrier_features in iter_windows(
        frames, calibration_result.subcarriers, gain_lock if should_apply_lock else None,
        window=config.window, hop=config.hop, intercarrier=intercarrier,
        image_subcarriers=getattr(calibration_result, "image_subcarriers", None),
        frame_average=config.frame_average, image_baseline=image_baseline,
        intercarrier_baseline=intercarrier_baseline,
    ):
        delivered_rate_hz = measure_delivered_rate_hz(len(recent_timestamps_s), recent_timestamps_s[0],
                                                      recent_timestamps_s[-1])
        if not sample_rate_rule.is_satisfied(delivered_rate_hz):
            violations.record(sample_rate_rule, delivered_rate_hz, "the served stream", timestamp)
            yield None
            continue
        proba = session.predict_window_proba(pick(features, image, intercarrier_features))
        class_id, confidence = decide_class(proba, classes)
        yield WindowVerdict(timestamp, class_id, confidence, proba,
                            WindowView(features, image, intercarrier_features))


def infer_windows(source: CsiSource, calibration_result: CalibrationResult, gain_lock: GainLock | None, session,
                  mode, publisher, *, behaviours: Sequence[WindowBehaviour] = (),
                  min_sample_rate_fraction: float = DEFAULT_MIN_SAMPLE_RATE_FRACTION) -> list[RecognitionResult]:
    """Publishes one verdict per `iter_window_verdicts` window and each behaviour's events right after it,
    then each behaviour's extra rows after the last window. O(windows)."""
    results = []
    for verdict in iter_window_verdicts(source, calibration_result, gain_lock, session, mode,
                                        min_sample_rate_fraction=min_sample_rate_fraction):
        if verdict is None:
            continue
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
