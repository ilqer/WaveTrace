"""Real-time result publishers. JSONL is the default sink."""

from wavetrace.output.Publisher import JsonlPublisher, Publisher, resultToDict

__all__ = ["Publisher", "JsonlPublisher", "resultToDict"]
