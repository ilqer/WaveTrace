import pytest

from wavetrace import pipeline


class _UnloadableMode:
    def format_banner(self, port):
        raise ValueError("per-node heads disagree on class ordering")


def test_a_banner_that_raises_binds_no_socket(monkeypatch):
    bound_ports = []
    monkeypatch.setattr(pipeline, "UdpLinkBatchReceiver",
                        lambda port, *, timeout_s: bound_ports.append(port))
    with pytest.raises(ValueError):
        pipeline.serve_mesh(_UnloadableMode(), 5599)
    assert bound_ports == []



class _BannerMode:
    def format_banner(self, port):
        return f"serving udp/{port}\n"


class _RecordingReceiver:
    def __init__(self):
        self.is_closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        self.close()

    def receive(self):
        return {}

    def close(self):
        self.is_closed = True


def test_a_banner_that_cannot_be_printed_still_closes_the_socket(monkeypatch):
    def _print_to_closed_pipe(*args, **kwargs):
        raise BrokenPipeError

    receiver = _RecordingReceiver()
    monkeypatch.setattr(pipeline, "UdpLinkBatchReceiver", lambda port, *, timeout_s: receiver)
    monkeypatch.setattr(pipeline, "print", _print_to_closed_pipe, raising=False)
    with pytest.raises(BrokenPipeError):
        pipeline.serve_mesh(_BannerMode(), 5599)
    assert receiver.is_closed
