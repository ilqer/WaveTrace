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
