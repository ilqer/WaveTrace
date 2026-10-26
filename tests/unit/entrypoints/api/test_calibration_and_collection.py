import pytest

from wavetrace.entrypoints.api.routers.pipeline import StartRequest


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path, monkeypatch):
    """Collection writes to the cwd-relative output/dataset_ui; this keeps it out of the repo's output/."""
    monkeypatch.chdir(tmp_path)


def _synthetic_request(**overrides):
    return StartRequest(synthetic=True, duration=2.0, baseline_packets=100, **overrides)


def test_a_calibration_that_cannot_save_logs_the_error_and_still_ends_the_pipeline(tmp_path,
                                                                                    build_recording_runner):
    (tmp_path / "a_file").write_text("")
    runner, messages = build_recording_runner()
    runner.run_calibration_request(_synthetic_request(action="calib", cal_out=str(tmp_path / "a_file" / "cal")))
    assert any(line.startswith("ERROR Calibration failed: ") for line in messages.logs())
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]
    assert not runner.is_running


def test_a_collection_without_a_calibration_logs_the_error_and_still_ends_the_pipeline(tmp_path,
                                                                                       build_recording_runner):
    runner, messages = build_recording_runner()
    runner.run_collection_request(_synthetic_request(action="collect", calibration=str(tmp_path / "missing_cal")))
    assert any(line.startswith("ERROR Collection failed: FileNotFoundError: ") for line in messages.logs())
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]
    assert not runner.is_running


@pytest.mark.parametrize(("verb", "action"), [("run_calibration_request", "calib"),
                                              ("run_collection_request", "collect")])
def test_a_stop_while_the_source_opens_still_ends_the_pipeline_and_writes_nothing(tmp_path, build_recording_runner,
                                                                                   verb, action):
    def stop_once_the_source_is_announced(message):
        if "[SIM]" in message:
            runner.stop()

    runner, messages = build_recording_runner({"logs": stop_once_the_source_is_announced})
    getattr(runner, verb)(_synthetic_request(action=action, cal_out=str(tmp_path / "cal")))
    assert messages.logs() == ["[SIM] Synthetic source (no hardware connected).", "System halt requested."]
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]
    assert not runner.is_running
    assert list(tmp_path.iterdir()) == []


def test_a_synthetic_calibration_logs_its_path_then_ends_the_pipeline(tmp_path, build_recording_runner):
    runner, messages = build_recording_runner()
    runner.run_calibration_request(_synthetic_request(action="calib", cal_out=str(tmp_path / "cal")))
    assert messages.logs() == ["[SIM] Synthetic source (no hardware connected).",
                               f"Starting Calibration -> {tmp_path / 'cal'}",
                               f"Calibration complete: {tmp_path / 'cal'}"]
    assert messages.payloads("inference") == [{"event": "pipeline_done"}]
    assert not runner.is_running
