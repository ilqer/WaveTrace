from dataclasses import asdict, dataclass, field

import numpy as np


# Event field names are JSONL/WebSocket wire keys (via to_dict); t is in seconds.
@dataclass(frozen=True, slots=True)
class WeaponAlertEvent:
    event: str = field(default="weapon_alert", init=False)
    t: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlertClearedEvent:
    event: str = field(default="clear", init=False)
    t: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecalibrationAdvisoryEvent:
    event: str = field(default="recalibrate_advisory", init=False)
    t: float
    drift: float

    def to_dict(self) -> dict:
        return asdict(self)


GuardEvent = WeaponAlertEvent | AlertClearedEvent | RecalibrationAdvisoryEvent


class AlertGuard:
    """N-on/M-off debounce with cooldown. O(1)/window.

    State machine:
      inactive: consecutive_positives increments on the positive class, resets on any other; at
        positives_to_alert AND past cooldown -> WeaponAlertEvent, go active. If positives_to_alert is
        reached inside the cooldown: no event, stay inactive, the counter holds its value (fires when
        the cooldown passes).
      active: consecutive_negatives increments on non-positive; a positive resets it; at
        negatives_to_clear -> AlertClearedEvent, go inactive, reset both counters.
    """

    def __init__(self, *, positives_to_alert: int = 4, negatives_to_clear: int = 6,
                 cooldown_s: float = 3.0, positive_class: int = 1):
        self._positives_to_alert = positives_to_alert
        self._negatives_to_clear = negatives_to_clear
        self._cooldown_s = cooldown_s
        self._positive_class = positive_class
        self._is_active = False
        self._consecutive_positives = 0
        self._consecutive_negatives = 0
        self._last_alert_timestamp_s = -1e18

    def update(self, timestamp_s: float, class_id: int) -> GuardEvent | None:
        if not self._is_active:
            if class_id == self._positive_class:
                self._consecutive_positives += 1
            else:
                self._consecutive_positives = 0
            if (self._consecutive_positives >= self._positives_to_alert
                    and (timestamp_s - self._last_alert_timestamp_s) >= self._cooldown_s):
                self._is_active = True
                self._consecutive_negatives = 0
                self._last_alert_timestamp_s = timestamp_s
                return WeaponAlertEvent(t=timestamp_s)
        else:
            if class_id == self._positive_class:
                self._consecutive_negatives = 0
            else:
                self._consecutive_negatives += 1
                if self._consecutive_negatives >= self._negatives_to_clear:
                    self._is_active = False
                    self._consecutive_positives = 0
                    self._consecutive_negatives = 0
                    return AlertClearedEvent(t=timestamp_s)
        return None


class DriftMonitor:
    """Advises recalibration when a slow EMA of raw per-subcarrier |H| drifts from the baseline.
    O(S)/frame, no per-frame allocation. EMA seeded from the first frame; no advisory before min_frames."""

    def __init__(self, baseline_magnitudes, *, alpha: float = 1e-4, drift_threshold: float = 0.5,
                 min_frames: int = 2000, cooldown_s: float = 600.0):
        self._baseline_magnitudes = np.asarray(baseline_magnitudes, dtype=np.float32)
        self._alpha = float(alpha)
        self._drift_threshold = float(drift_threshold)
        self._min_frames = int(min_frames)
        self._cooldown_s = float(cooldown_s)
        self._ema_magnitudes: np.ndarray | None = None
        self._frame_count = 0
        self._last_advisory_timestamp_s = -1e18

    def update(self, timestamp_s: float, raw_magnitudes: np.ndarray) -> RecalibrationAdvisoryEvent | None:
        magnitudes = np.asarray(raw_magnitudes, dtype=np.float32)
        if self._ema_magnitudes is None:
            self._ema_magnitudes = magnitudes.copy()
        else:
            # ema += alpha·(x - ema), updating the stored array in place
            self._ema_magnitudes += self._alpha * (magnitudes - self._ema_magnitudes)
        self._frame_count += 1
        if self._frame_count < self._min_frames:
            return None
        drift = float(np.median(np.abs(
            self._ema_magnitudes / np.maximum(self._baseline_magnitudes, 1e-12) - 1.0)))
        if (drift >= self._drift_threshold
                and (timestamp_s - self._last_advisory_timestamp_s) >= self._cooldown_s):
            self._last_advisory_timestamp_s = timestamp_s
            return RecalibrationAdvisoryEvent(t=timestamp_s, drift=drift)
        return None
