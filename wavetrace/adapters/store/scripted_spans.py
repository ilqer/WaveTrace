"""Scripted-span sidecar: JSON `{"spans": [{"start": s, "end": e, "present": bool}, ...]}`; `present` defaults to true."""

import json


def load_scripted_spans(path) -> list[tuple[float, float, bool]]:
    with open(path) as spans_file:
        manifest = json.load(spans_file)
    return [(span["start"], span["end"], span.get("present", True)) for span in manifest["spans"]]
