"""Each live-serving entrypoint passes `--rate-gate` and `--min-rate-fraction` through to the use case that
applies them."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_count, run_live_mesh, run_weapon
from wavetrace import pipeline
from wavetrace.application import serve
from wavetrace.application.serve_mesh import RateGate
from wavetrace.entrypoints.cli import main as cli_main

SERVE_MODE_CLASS_NAME = {run_live_mesh: "PresenceServeMode", run_count: "CountServeMode",
                         run_weapon: "WeaponServeMode"}


class _StubServeMode:
    nodes = {0: object()}

    def format_banner(self, port):
        return ""


class _RecordingUseCase:
    instances = []

    def __init__(self, mode, **options):
        self.options = options
        self.num_runs = 0
        _RecordingUseCase.instances.append(self)

    def run(self, receiver):
        self.num_runs += 1


class _NullReceiver:
    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        pass


@pytest.fixture
def use_case_options(monkeypatch):
    _RecordingUseCase.instances = []
    monkeypatch.setattr(pipeline, "ServeMeshUseCase", _RecordingUseCase)
    monkeypatch.setattr(pipeline, "UdpLinkBatchReceiver", lambda port, *, timeout_s: _NullReceiver())

    def run_script(script, *flags):
        stub_class = SimpleNamespace(load=lambda *load_args: _StubServeMode())
        monkeypatch.setattr(script, SERVE_MODE_CLASS_NAME[script], stub_class)
        monkeypatch.setattr(sys, "argv", [Path(script.__file__).stem, *flags])
        script.main()
        assert len(_RecordingUseCase.instances) == 1 and _RecordingUseCase.instances[0].num_runs == 1
        return _RecordingUseCase.instances[0].options

    return run_script


@pytest.mark.parametrize("script", [run_live_mesh, run_count, run_weapon])
def test_a_serving_script_passes_the_rate_gate_flags_to_the_use_case(use_case_options, script):
    options = use_case_options(script, "--rate-gate", "peers", "--min-rate-fraction", "0.6")
    assert options["rate_gate"] is RateGate.PEERS
    assert options["min_sample_rate_fraction"] == 0.6


@pytest.mark.parametrize("script", [run_live_mesh, run_count, run_weapon])
def test_a_serving_script_gates_on_the_target_rate_at_eighty_percent_by_default(use_case_options, script):
    options = use_case_options(script)
    assert options["rate_gate"] is RateGate.TARGET
    assert options["min_sample_rate_fraction"] == 0.8


@pytest.mark.parametrize("script", [run_live_mesh, run_count, run_weapon])
def test_a_serving_script_refuses_an_unknown_rate_gate(use_case_options, script):
    with pytest.raises(SystemExit):
        use_case_options(script, "--rate-gate", "median")


def test_the_cli_run_mode_passes_the_minimum_rate_fraction_to_the_window_stream(monkeypatch, tmp_path):
    served_fractions = []

    def fake_iter_window_verdicts(source, calibration_result, gain_lock, session, mode, *,
                                  min_sample_rate_fraction):
        served_fractions.append(min_sample_rate_fraction)
        return iter(())

    monkeypatch.setattr(pipeline, "load_calibration", lambda calibration_dir: (None, None))
    monkeypatch.setattr(pipeline, "open_session", lambda mode, model_path: None)
    monkeypatch.setattr(serve, "iter_window_verdicts", fake_iter_window_verdicts)
    monkeypatch.setattr(cli_main, "build_csi_source", lambda args: None)
    run_flags = ["run", "--calibration", "cal", "--model", "model", "--out", str(tmp_path / "out.jsonl")]
    assert cli_main.main(run_flags + ["--min-rate-fraction", "0.5"]) == 0
    assert cli_main.main(run_flags) == 0
    assert served_fractions == [0.5, 0.8]
