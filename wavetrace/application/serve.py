"""The run/serve use-case: stream a source through the front-end and publish one verdict per
window."""

import numpy as np

from wavetrace.Calibration import loadCalibration, build_image_baseline
from wavetrace.Frontend import iterWindows
from wavetrace.recognition import SegmentVoter, modeSession, planInferenceInput
from wavetrace import RecognitionResult


def run_inference(source, calib_dir, model_path, mode, publisher, *, vote=False, guard=False):
    """Stream a source through the front-end and publish one verdict per window (+ a final soft-vote
    verdict when vote=True). When guard=True, wires AlertGuard+DriftMonitor for debounce and drift
    advisory (O(S)/frame extra — acceptable on the Pi serving side). O(windows)."""
    result, gain_lock = loadCalibration(calib_dir)
    session = modeSession(mode, model_path)
    apply_lock, intercarrier, pick = planInferenceInput(mode, session.head)
    config = session.head.config

    image_subcarriers = getattr(result, "image_subcarriers", None)
    image_baseline = None
    if config.subtract_baseline:
        image_baseline = build_image_baseline(result, locked=(apply_lock and gain_lock is not None))
    # the raw baseline, inter-carrier path only, exactly as training subtracted it
    ic_baseline = result.baseline_mag if getattr(config, "subtract_ic_baseline", False) else None

    frames_iter = source.frames()
    if guard:
        from wavetrace.output.Guard import AlertGuard, DriftMonitor
        drift_monitor = DriftMonitor(result.baseline_mag)
        alert_guard = AlertGuard()
        # tee the raw, pre-lock magnitudes to DriftMonitor without disturbing the frame stream
        def _tee_drift(frames, monitor, publisher):
            import numpy as _np
            for frame in frames:
                event = monitor.update(float(frame.timestamp),
                                        _np.abs(_np.asarray(frame.grid)).mean(axis=0).astype(_np.float32))
                if event:
                    publisher.publishEvent(event)
                yield frame
        frames_iter = _tee_drift(frames_iter, drift_monitor, publisher)

    voter = SegmentVoter() if vote else None
    results = []
    for timestamp, features, image, intercarrier_features in iterWindows(
        frames_iter, result.subcarriers, gain_lock if apply_lock else None,
        window=config.window, hop=config.hop, intercarrier=intercarrier,
        image_subcarriers=image_subcarriers,
        frame_average=config.frame_average,
        imageBaseline=image_baseline,
        ic_baseline=ic_baseline,
    ):
        predicted_class, confidence = session.predictWindow(pick(features, image, intercarrier_features))
        result_row = RecognitionResult()
        result_row.class_id = predicted_class
        result_row.confidence = confidence
        result_row.timestamp = timestamp
        publisher.publish(result_row)
        results.append(result_row)
        if guard:
            event = alert_guard.update(timestamp, predicted_class)
            if event:
                publisher.publishEvent(event)
        if voter is not None:
            voter.add(session.head.predict_proba(np.asarray(pick(features, image, intercarrier_features),
                                                             dtype=np.float32).reshape(1, -1))[0])
    if voter is not None and len(voter):
        voted_class, voted_mean = voter.finalize()
        vote_row = RecognitionResult()
        vote_row.class_id = int(voted_class)
        vote_row.confidence = float(voted_mean[voted_class])
        vote_row.timestamp = results[-1].timestamp if results else 0.0
        publisher.publish(vote_row)
        results.append(vote_row)
    return results
