"""Unit tests for web.ServerOptions.WebServerOptions (§1.5 tests/unit/, mirroring
web/ServerOptions.py). Split out of tests/unit/test_config.py when WebServerOptions moved out of
wavetrace.Config — its only consumer is web/app.py, so it has nothing to do with the CSI domain."""

import pytest

from web.ServerOptions import WebServerOptions


def test_web_server_options_defaults_match_the_previous_uvicorn_literals():
    options = WebServerOptions()
    assert options.host == "0.0.0.0" and options.port == 8000 and options.reload is True
    assert options.reload_dirs == ("web", "wavetrace") and options.reload_includes == ("*.py",)


def test_web_server_options_rejects_out_of_range_port():
    with pytest.raises(ValueError, match="port"):
        WebServerOptions(port=70000)


def test_web_server_options_are_frozen():
    options = WebServerOptions()
    with pytest.raises(AttributeError):
        options.port = 1
