from types import SimpleNamespace

import numpy as np
import pytest

from wavetrace.application.behaviours import (
    AlertGuardBehaviour, DriftMonitoredSource, SegmentVoteBehaviour,
)
from wavetrace.domain.guard import (
    AlertClearedEvent, AlertGuard, DriftMonitor, RecalibrationAdvisoryEvent, WeaponAlertEvent,
)
from wavetrace.domain.recognition.vote import SegmentVoter

UNUSED_PROBA = np.array([0.5, 0.5])


class _RecordingPublisher:
    def __init__(self):
        self.events = []

    def publish_event(self, event):
        self.events.append(event)


def test_alert_guard_alerts_on_the_fourth_positive_window_and_clears_on_the_sixth_negative():
    behaviour = AlertGuardBehaviour(AlertGuard())
    events = [behaviour.observe_window(float(i), 1, UNUSED_PROBA) for i in range(4)]
    events += [behaviour.observe_window(float(4 + i), 0, UNUSED_PROBA) for i in range(6)]
    assert events[:3] == [None] * 3 and events[3] == WeaponAlertEvent(t=3.0)
    assert events[4:9] == [None] * 5 and events[9] == AlertClearedEvent(t=9.0)
    assert behaviour.finish() is None


def test_segment_vote_adds_one_row_with_the_mean_argmax_at_the_last_window():
    behaviour = SegmentVoteBehaviour(SegmentVoter())
    probas = [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]
    for timestamp_s, proba in zip([1.0, 1.5, 2.0], probas):
        assert behaviour.observe_window(timestamp_s, 0, np.array(proba)) is None
    vote_row = behaviour.finish()
    assert vote_row.class_id == 1
    assert vote_row.confidence == pytest.approx((0.1 + 0.8 + 0.7) / 3)
    assert vote_row.timestamp == 2.0


def test_segment_vote_with_no_windows_adds_nothing():
    assert SegmentVoteBehaviour(SegmentVoter()).finish() is None


def test_drift_monitored_source_passes_frames_through_and_publishes_the_advisory():
    frames = [SimpleNamespace(timestamp=float(i), grid=np.full((2, 4), 2.0 + 0j)) for i in range(3)]
    publisher = _RecordingPublisher()
    monitor = DriftMonitor(np.ones(4, dtype=np.float32), min_frames=1, cooldown_s=10.0)
    source = DriftMonitoredSource(SimpleNamespace(frames=lambda: iter(frames)), monitor, publisher)
    assert list(source.frames()) == frames
    assert publisher.events == [RecalibrationAdvisoryEvent(t=0.0, drift=1.0)]
