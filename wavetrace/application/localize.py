"""The localize use-case: stream a source through the AoA Localizer -> a published track +
a persisted room map."""

import numpy as np

from wavetrace.Localize import Localizer, Tracker, saveLocalization
from wavetrace import RecognitionResult


def _spatial_result(timestamp, x_m, y_m, angle_deg, range_m, confidence, located):
    """A spatial fix -> RecognitionResult on the wire schema: location rides in bbox [x, y, 0, 0]
    (Publisher.resultToDict emits it), azimuth + range in keypoints. class_id = 1 when this frame
    carries a real (measured/confident) fix, 0 when it is a coasted/low-confidence estimate. nan
    range -> -1 (JSON-safe)."""
    result = RecognitionResult()
    result.class_id = 1 if located else 0
    result.confidence = float(confidence)
    result.timestamp = float(timestamp)
    result.bbox = [float(x_m), float(y_m), 0.0, 0.0]
    result.keypoints = [float(angle_deg), (-1.0 if np.isnan(range_m) else float(range_m))]
    return result


def localize_source(source, out_dir, *, num_antennas, antenna_spacing_wavelengths=0.5,
                     method="music", num_sources=1, num_angles=181,
                     subcarrier_spacing_hz=312.5e3, max_range_m=12.0, num_ranges=64,
                     range_enabled=True, filter_track=True, publisher=None):
    """Stream a source through the AoA Localizer: publish the per-frame track as RecognitionResults
    (Publisher wire schema) and persist the aggregate joint-2-D room map. Returns (path, aggregate
    Localization). Needs >= 2 RX antennas (2-antenna ESP32 / Pi NIC).

    filter_track (default on): smooth the raw per-frame measurements with a constant-velocity Kalman
    `Tracker` — predict from motion (no teleporting), fuse each measurement weighted by its confidence,
    and gate impossible jumps. The PUBLISHED track is the filtered one; the saved room map is the
    raw aggregate. O(F·(A²S + A·G) + (A·S)³)."""
    localizer = Localizer(num_antennas, spacing=antenna_spacing_wavelengths, method=method,
                           num_sources=num_sources, num_angles=num_angles,
                           subcarrier_spacing_hz=subcarrier_spacing_hz, max_range_m=max_range_m,
                           num_ranges=num_ranges, range_enabled=range_enabled)
    tracker = Tracker(range_enabled=range_enabled) if filter_track else None
    frames = list(source.frames())
    for localization in localizer.locateStream(frames):
        if publisher is None:
            continue
        if tracker is not None:
            track_state = tracker.update(localization)
            publisher.publish(_spatial_result(track_state.timestamp, track_state.x_m, track_state.y_m,
                                               track_state.angle_deg, track_state.range_m,
                                               track_state.confidence, track_state.measured))
        else:
            publisher.publish(_spatial_result(localization.timestamp, localization.x_m, localization.y_m,
                                               localization.peak_angle_deg, localization.peak_range_m,
                                               localization.confidence, localization.confidence >= 0.5))
    aggregate_localization = localizer.aggregate(frames)
    return saveLocalization(aggregate_localization, out_dir), aggregate_localization
