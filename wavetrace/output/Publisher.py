"""Where a verdict goes: one RecognitionResult per inference, serialized and sent.

`Publisher` is the transport-agnostic sink. `JsonlPublisher` is the default: one JSON line per
result to stdout or a file, with no broker or server to run. A network transport subclasses
Publisher and reuses `resultToDict`. O(1) per result.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
import sys

from wavetrace.output.Guard import GuardEvent


def resultToDict(result, *, mode: str = "") -> dict:
    """RecognitionResult -> the wire schema. bbox/keypoints ride along only when the head is spatial
    (the ladder's location/posture heads); presence/weapon leave them null/empty."""
    boxVal = getattr(result, "bbox", None)
    return {
        "t": float(result.timestamp),
        "class": int(result.class_id),
        "conf": float(result.confidence),
        "mode": mode,
        "bbox": list(boxVal) if boxVal is not None else None,
        "keypoints": list(getattr(result, "keypoints", []) or []),
    }


class Publisher(ABC):
    """Backend-agnostic result sink. Subclasses implement the transport; callers see publish/close.

    Event lines carry an "event" key; result lines never do. publishEvent has a no-op default so
    subclasses without an override do not crash (non-breaking for existing subclasses)."""

    def __init__(self, *, mode: str = ""):
        self.mode = mode  # stamped on every message so a consumer knows presence from weapon

    @abstractmethod
    def publish(self, result) -> None:
        """Serialize and emit one RecognitionResult. O(1)."""

    def publishEvent(self, event: GuardEvent) -> None:
        """Emit one guard/advisory event. Default is a no-op; override to transport it."""

    def close(self) -> None:
        """Flush/close the transport. No-op by default."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class JsonlPublisher(Publisher):
    """One JSON line per result to a stream or file (default stdout). Zero-dependency."""

    def __init__(self, sink=None, *, mode: str = ""):
        super().__init__(mode=mode)
        if sink is None:
            self._fh, self._owned = sys.stdout, False
        elif isinstance(sink, (str, Path)):
            sinkPath = Path(sink)
            sinkPath.parent.mkdir(parents=True, exist_ok=True)
            self._fh, self._owned = open(sinkPath, "w"), True
        else:
            self._fh, self._owned = sink, False

    def publish(self, result) -> None:
        self._fh.write(json.dumps(resultToDict(result, mode=self.mode)) + "\n")
        self._fh.flush()  # a downstream tail should see each verdict as it happens

    def publishEvent(self, event: GuardEvent) -> None:
        self._fh.write(json.dumps(event.to_dict()) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._owned:
            self._fh.close()
