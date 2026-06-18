"""Phase 8 — real-time result publisher.

The deployment path emits one RecognitionResult per inference (window verdict, or a per-segment
vote). `Publisher` is the backend-agnostic sink; `JsonlPublisher` is the zero-dependency default —
one JSON line per result to stdout or a file, always testable with no broker/server.

Concrete network backends are SEAMS (built when a real consumer exists, behind optional deps):
  * MqttPublisher  — paho-mqtt; publish each line to a broker topic (pip install wavetrace[mqtt]).
  * WsPublisher    — websockets; push each line to connected clients (pip install wavetrace[ws]).
Both would subclass Publisher and reuse `result_to_dict` — only the transport differs.

O(1) serialize + publish per result (plan §2.6).
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
import sys


def result_to_dict(result, *, mode: str = "") -> dict:
    """RecognitionResult -> the wire schema. bbox/keypoints ride along only when the head is spatial
    (the ladder's location/posture heads); presence/weapon leave them null/empty."""
    bbox = getattr(result, "bbox", None)
    return {
        "t": float(result.timestamp),
        "class": int(result.class_id),
        "conf": float(result.confidence),
        "mode": mode,
        "bbox": list(bbox) if bbox is not None else None,
        "keypoints": list(getattr(result, "keypoints", []) or []),
    }


class Publisher(ABC):
    """Backend-agnostic result sink. Subclasses implement the transport; callers see publish/close.

    Event lines carry an "event" key; result lines never do. publish_event has a no-op default so
    subclasses without an override do not crash (non-breaking for existing subclasses)."""

    def __init__(self, *, mode: str = ""):
        self.mode = mode  # stamped on every message so a consumer knows presence vs weapon

    @abstractmethod
    def publish(self, result) -> None:
        """Serialize and emit one RecognitionResult. O(1)."""

    def publish_event(self, event: dict) -> None:
        """Emit one guard/advisory event dict. Default is a no-op; override to transport it."""

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
        # sink: a path (opened for append), an open text stream, or None -> stdout
        if sink is None:
            self._fh, self._owned = sys.stdout, False
        elif isinstance(sink, (str, Path)):
            p = Path(sink)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh, self._owned = open(p, "w"), True
        else:
            self._fh, self._owned = sink, False

    def publish(self, result) -> None:
        self._fh.write(json.dumps(result_to_dict(result, mode=self.mode)) + "\n")
        self._fh.flush()  # real-time: a downstream tail should see verdicts as they happen

    def publish_event(self, event: dict) -> None:
        self._fh.write(json.dumps(event) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._owned:
            self._fh.close()
