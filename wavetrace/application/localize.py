import numpy as np

from wavetrace import RecognitionResult
from wavetrace.application.ports import CsiSource
from wavetrace.domain.localize import Localization, Localizer, Tracker


def _spatial_result(timestamp_s, x_m, y_m, angle_deg, range_m, confidence, is_located):
    """Wire schema: location rides in bbox [x, y, 0, 0], azimuth + range in keypoints. class_id = 1 for
    a measured/confident fix, 0 for a coasted/low-confidence one. nan range -> -1 (JSON-safe)."""
    result = RecognitionResult()
    result.class_id = 1 if is_located else 0
    result.confidence = float(confidence)
    result.timestamp = float(timestamp_s)
    result.bbox = [float(x_m), float(y_m), 0.0, 0.0]
    result.keypoints = [float(angle_deg), (-1.0 if np.isnan(range_m) else float(range_m))]
    return result


def localize(source: CsiSource, localizer: Localizer, tracker: Tracker | None, publisher) -> Localization:
    """Publishes one fix per frame (none when publisher is None) and returns the aggregate room map.

    tracker: the published track goes through this constant-velocity Kalman filter (confidence-weighted
    fusion, gates impossible jumps); None publishes the raw per-frame fixes. The room map is always the
    raw aggregate. O(F·(A²S + A·G) + (A·S)³)."""
    frames = list(source.frames())
    if publisher is not None:
        for localization in localizer.locate_stream(frames):
            if tracker is not None:
                track_state = tracker.update(localization)
                publisher.publish(_spatial_result(track_state.timestamp, track_state.x_m, track_state.y_m,
                                                   track_state.angle_deg, track_state.range_m,
                                                   track_state.confidence, track_state.measured))
            else:
                publisher.publish(_spatial_result(localization.timestamp, localization.x_m, localization.y_m,
                                                   localization.peak_angle_deg, localization.peak_range_m,
                                                   localization.confidence, localization.confidence >= 0.5))
    return localizer.aggregate(frames)
