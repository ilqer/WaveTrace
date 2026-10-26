from wavetrace.entrypoints.api.camera_feed import CameraFeed


def test_each_weights_file_loads_once():
    loaded = []
    feed = CameraFeed(load_model=lambda weights: loaded.append(weights) or f"model:{weights}")
    assert feed.load_model("a.pt") == "model:a.pt"
    assert feed.load_model("a.pt") == "model:a.pt"
    assert feed.load_model("b.pt") == "model:b.pt"
    assert loaded == ["a.pt", "b.pt"]


def test_a_failed_load_is_cached_as_no_model(capsys):
    attempts = []

    def fail(weights):
        attempts.append(weights)
        raise FileNotFoundError(weights)

    feed = CameraFeed(load_model=fail)
    assert feed.load_model("missing.pt") is None
    assert feed.load_model("missing.pt") is None
    assert attempts == ["missing.pt"]
    assert capsys.readouterr().out == "[YOLO] load failed: missing.pt\n"


def test_two_feeds_share_no_state():
    first, second = CameraFeed(load_model=str), CameraFeed(load_model=str)
    first.is_streaming = True
    assert second.is_streaming is False
