import json

from wavetrace.adapters.store.scripted_spans import load_scripted_spans


def test_load_scripted_spans_defaults_present_to_true(tmp_path):
    path = tmp_path / "spans.json"
    path.write_text(json.dumps({"spans": [{"start": 1.0, "end": 2.0}, {"start": 3.0, "end": 4.0, "present": False}]}))
    assert load_scripted_spans(path) == [(1.0, 2.0, True), (3.0, 4.0, False)]
