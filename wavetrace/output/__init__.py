"""Real-time result publishers. JSONL is the default sink."""

from wavetrace.output.publisher import JsonlPublisher, Publisher, result_to_dict

__all__ = ["Publisher", "JsonlPublisher", "result_to_dict"]
