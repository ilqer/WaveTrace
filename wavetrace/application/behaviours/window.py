from typing import Protocol

import numpy as np

from wavetrace import RecognitionResult
from wavetrace.domain.guard import AlertGuard, GuardEvent
from wavetrace.domain.recognition.vote import SegmentVoter


class WindowBehaviour(Protocol):
    def observe_window(self, timestamp_s: float, class_id: int, proba: np.ndarray) -> GuardEvent | None:
        """Called once per window after its verdict is published; a returned event is published."""

    def finish(self) -> RecognitionResult | None:
        """Called once after the last window; a returned row is published and appended."""


class AlertGuardBehaviour:
    """Debounced weapon alert / clear events. O(1)/window."""

    def __init__(self, alert_guard: AlertGuard):
        self._alert_guard = alert_guard

    def observe_window(self, timestamp_s: float, class_id: int, proba: np.ndarray) -> GuardEvent | None:
        return self._alert_guard.update(timestamp_s, class_id)

    def finish(self) -> None:
        return None


def build_alert_behaviours(mode: str) -> list[WindowBehaviour]:
    """The debounced weapon alert belongs to weapon mode only; a presence or count positive is no threat."""
    return [AlertGuardBehaviour(AlertGuard())] if mode == "weapon" else []


class SegmentVoteBehaviour:
    """One soft-vote verdict over the whole stream, stamped with the last window's timestamp.
    O(1)/window, O(windows) memory and finish."""

    def __init__(self, voter: SegmentVoter):
        self._voter = voter
        self._last_timestamp_s = 0.0

    def observe_window(self, timestamp_s: float, class_id: int, proba: np.ndarray) -> None:
        self._voter.add(proba)
        self._last_timestamp_s = timestamp_s

    def finish(self) -> RecognitionResult | None:
        if not len(self._voter):
            return None
        # the vote's class id is the proba column index, not a `classes_` label
        voted_index, voted_mean = self._voter.finalize()
        vote_row = RecognitionResult()
        vote_row.class_id = int(voted_index)
        vote_row.confidence = float(voted_mean[voted_index])
        vote_row.timestamp = self._last_timestamp_s
        return vote_row
