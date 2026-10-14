"""Result sinks: `JsonlPublisher` writes one JSON line per result; a network transport subclasses
`Publisher` and reuses `result_to_dict`. O(1) per result.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
import sys

from wavetrace.domain.guard import GuardEvent


def result_to_dict(result, *, mode: str = "") -> dict:
    """bbox/keypoints ride along only for spatial results; presence/weapon leave them null/empty."""
    bounding_box = getattr(result, "bbox", None)
    return {
        "t": float(result.timestamp),
        "class": int(result.class_id),
        "conf": float(result.confidence),
        "mode": mode,
        "bbox": list(bounding_box) if bounding_box is not None else None,
        "keypoints": list(getattr(result, "keypoints", []) or []),
    }


class Publisher(ABC):
    """Event lines carry an "event" key; result lines never do. publish_event is a no-op by default so
    subclasses without an override do not crash."""

    def __init__(self, *, mode: str = ""):
        self.mode = mode  # stamped on every message so a consumer knows presence from weapon

    @abstractmethod
    def publish(self, result) -> None:
        """O(1)."""

    def publish_event(self, event: GuardEvent) -> None:
        """No-op by default."""

    def close(self) -> None:
        """Flush/close the transport. No-op by default."""

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        self.close()


class JsonlPublisher(Publisher):
    """Default sink is stdout; a str/Path sink is opened for writing and closed by `close`."""

    def __init__(self, sink=None, *, mode: str = ""):
        super().__init__(mode=mode)
        if sink is None:
            self._stream, self._is_owned = sys.stdout, False
        elif isinstance(sink, (str, Path)):
            sink_path = Path(sink)
            sink_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream, self._is_owned = open(sink_path, "w"), True
        else:
            self._stream, self._is_owned = sink, False

    def publish(self, result) -> None:
        self._stream.write(json.dumps(result_to_dict(result, mode=self.mode)) + "\n")
        self._stream.flush()  # a downstream tail should see each verdict as it happens

    def publish_event(self, event: GuardEvent) -> None:
        self._stream.write(json.dumps(event.to_dict()) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._is_owned:
            self._stream.close()
