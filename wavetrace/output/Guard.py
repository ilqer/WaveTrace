"""T5/P10 — production serving guards: AlertGuard debounce + DriftMonitor recalibrate advisory.

AlertGuard: N-consecutive-positive → alert, M-consecutive-negative → clear (with cooldown).
DriftMonitor: slow EMA of raw per-subcarrier |H| vs quiet-room baseline → recalibrate advisory.
Both are pure-Python, O(1)/window and O(S)/frame respectively; zero effect when not instantiated.
"""

import numpy as np


class AlertGuard:
    """N-on/M-off debounce + cooldown over per-window verdicts -> alert/clear events. O(1)/window.

    State machine:
      inactive: pos_count increments on positive class, resets on other; at n_on AND past cooldown
        -> {"event": "weapon_alert", "t": t}, go active.  If n_on reached but in cooldown: no event,
        stay inactive, counter holds its value (fires when cooldown passes).
      active: neg_count increments on non-positive; a positive resets it; at n_off
        -> {"event": "clear", "t": t}, go inactive, reset all counters.
    """

    def __init__(self, *, n_on: int = 4, n_off: int = 6, cooldown_s: float = 3.0,
                 positive_class: int = 1):
        self._n_on = n_on
        self._n_off = n_off
        self._cooldown_s = cooldown_s
        self._pos_cls = positive_class
        self._active = False
        self._pos_count = 0
        self._neg_count = 0
        self._last_alert_t = -1e18

    def update(self, t: float, class_id: int) -> dict | None:
        """Process one window verdict; return an event dict or None."""
        if not self._active:
            if class_id == self._pos_cls:
                self._pos_count += 1
            else:
                self._pos_count = 0
            if self._pos_count >= self._n_on and (t - self._last_alert_t) >= self._cooldown_s:
                self._active = True
                self._neg_count = 0
                self._last_alert_t = t
                return {"event": "weapon_alert", "t": t}
        else:
            if class_id == self._pos_cls:
                self._neg_count = 0
            else:
                self._neg_count += 1
                if self._neg_count >= self._n_off:
                    self._active = False
                    self._pos_count = 0
                    self._neg_count = 0
                    return {"event": "clear", "t": t}
        return None


class DriftMonitor:
    """Slow EMA of raw per-subcarrier |H| vs the calibration baseline -> recalibrate advisory.
    O(S)/frame, no per-frame allocation. EMA initialized from first frame; advisory after min_frames."""

    def __init__(self, baseline_mag, *, alpha: float = 1e-4, drift_thresh: float = 0.5,
                 min_frames: int = 2000, cooldown_s: float = 600.0):
        self._baseline = np.asarray(baseline_mag, dtype=np.float32)
        self._alpha = float(alpha)
        self._drift_thresh = float(drift_thresh)
        self._min_frames = int(min_frames)
        self._cooldown_s = float(cooldown_s)
        self._ema: np.ndarray | None = None
        self._frame_count = 0
        self._last_advisory_t = -1e18

    def update(self, t: float, raw_mags: np.ndarray) -> dict | None:
        """Push one frame's raw per-subcarrier magnitudes; return advisory dict or None."""
        mags = np.asarray(raw_mags, dtype=np.float32)
        if self._ema is None:
            self._ema = mags.copy()
        else:
            self._ema += self._alpha * (mags - self._ema)  # in-place update, no new allocation
        self._frame_count += 1
        if self._frame_count < self._min_frames:
            return None
        driftVal = float(np.median(np.abs(self._ema / np.maximum(self._baseline, 1e-12) - 1.0)))
        if driftVal >= self._drift_thresh and (t - self._last_advisory_t) >= self._cooldown_s:
            self._last_advisory_t = t
            return {"event": "recalibrate_advisory", "t": t, "drift": driftVal}
        return None
