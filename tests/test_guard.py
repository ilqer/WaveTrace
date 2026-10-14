import json

import numpy as np
import pytest

from wavetrace.output.guard import (
    AlertClearedEvent,
    AlertGuard,
    DriftMonitor,
    RecalibrationAdvisoryEvent,
    WeaponAlertEvent,
)


def test_alert_guard_fires_after_n_on():
    guard = AlertGuard(positives_to_alert=3, negatives_to_clear=5, cooldown_s=0.0, positive_class=1)
    assert guard.update(0.0, 1) is None   # count=1
    assert guard.update(0.1, 1) is None   # count=2
    alert = guard.update(0.2, 1)          # count=3 → fires
    assert alert is not None
    assert isinstance(alert, WeaponAlertEvent)
    assert alert.event == "weapon_alert"
    assert alert.t == pytest.approx(0.2)
    # A negative resets the counter
    second_guard = AlertGuard(positives_to_alert=3, negatives_to_clear=5, cooldown_s=0.0)
    second_guard.update(0.0, 1)
    second_guard.update(0.1, 0)
    second_guard.update(0.2, 1)
    assert second_guard.update(0.3, 1) is None  # only 2 consecutive, not 3


def test_alert_guard_clear_after_n_off():
    guard = AlertGuard(positives_to_alert=2, negatives_to_clear=3, cooldown_s=0.0)
    guard.update(0.0, 1)
    guard.update(0.1, 1)  # alert fires
    assert guard.update(0.2, 0) is None   # neg=1
    assert guard.update(0.3, 0) is None   # neg=2
    cleared = guard.update(0.4, 0)           # neg=3 → clear
    assert cleared is not None
    assert isinstance(cleared, AlertClearedEvent)
    assert cleared.event == "clear"
    # A positive mid-sequence resets the neg counter
    second_guard = AlertGuard(positives_to_alert=2, negatives_to_clear=3, cooldown_s=0.0)
    second_guard.update(0.0, 1); second_guard.update(0.1, 1)
    second_guard.update(0.2, 0); second_guard.update(0.3, 1)  # positive resets neg_count
    assert second_guard.update(0.4, 0) is None  # only 1 neg after reset, not 3


def test_alert_guard_cooldown_suppresses_second_alert():
    guard = AlertGuard(positives_to_alert=2, negatives_to_clear=3, cooldown_s=10.0, positive_class=1)

    guard.update(0.9, 1)
    first_alert = guard.update(1.0, 1)  # pos_count=2 → alert, _last_alert_timestamp_s=1.0
    assert first_alert is not None and first_alert.event == "weapon_alert"

    # Clear via 3 negatives
    guard.update(2.0, 0); guard.update(3.0, 0); guard.update(4.0, 0)

    # Second burst within cooldown (5.1 - 1.0 = 4.1 < 10.0)
    guard.update(5.0, 1)
    suppressed_event = guard.update(5.1, 1)  # pos_count=2 >= positives_to_alert, but in cooldown
    assert suppressed_event is None

    # Past cooldown: a fresh burst fires
    guard.update(5.2, 0)        # reset pos_count to 0
    guard.update(12.0, 1)       # pos_count=1
    second_alert = guard.update(12.1, 1) # pos_count=2, 12.1-1.0=11.1 >= 10.0 → alert
    assert second_alert is not None and second_alert.event == "weapon_alert"


def test_drift_monitor_advisory():
    baseline = np.ones(16, dtype=np.float32)
    # alpha=1.0 makes EMA = most-recent frame (instant convergence)
    monitor = DriftMonitor(baseline, alpha=1.0, drift_threshold=0.4, min_frames=3, cooldown_s=0.0)

    # First frame initialises EMA (no update formula)
    assert monitor.update(0.0, baseline * 2.0) is None   # frame 1, count < 3
    assert monitor.update(1.0, baseline * 2.0) is None   # frame 2, count < 3
    # Frame 3 = min_frames: EMA=2*baseline, drift=|2/1-1|=1.0 >= 0.4 → advisory
    advisory = monitor.update(2.0, baseline * 2.0)
    assert advisory is not None
    assert isinstance(advisory, RecalibrationAdvisoryEvent)
    assert advisory.event == "recalibrate_advisory"
    assert advisory.drift == pytest.approx(1.0, abs=0.05)

    # Cooldown=0 → advisory fires again on next frame if still drifted
    repeat_advisory = monitor.update(3.0, baseline * 2.0)
    assert repeat_advisory is not None

    # If drift drops below threshold → no advisory
    low_drift_monitor = DriftMonitor(baseline, alpha=1.0, drift_threshold=0.5, min_frames=2, cooldown_s=0.0)
    low_drift_monitor.update(0.0, baseline * 1.1)  # EMA ≈ baseline*1.1, drift ≈ 0.1 < 0.5
    no_advisory = low_drift_monitor.update(1.0, baseline * 1.1)
    assert no_advisory is None


def test_guard_events_serialize_to_pinned_jsonl_bytes():
    """A JSONL consumer (the dashboard) reads these bytes: key names and order are pinned."""
    assert json.dumps(WeaponAlertEvent(t=1.5).to_dict()) == '{"event": "weapon_alert", "t": 1.5}'
    assert json.dumps(AlertClearedEvent(t=2.25).to_dict()) == '{"event": "clear", "t": 2.25}'
    assert (json.dumps(RecalibrationAdvisoryEvent(t=3.0, drift=0.75).to_dict())
            == '{"event": "recalibrate_advisory", "t": 3.0, "drift": 0.75}')
