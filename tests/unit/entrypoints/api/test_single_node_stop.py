import numpy as np

from fixtures.synthetic_csi import calibrate_quiet_baseline
from wavetrace.entrypoints.api import streamer
from wavetrace.domain.model_config import ModelConfig
from wavetrace.domain.contracts import derive_pipeline_contract
from wavetrace.synthetic import generate_paired_recording
from types import SimpleNamespace

NUM_ANTENNAS, NUM_SUBCARRIERS, SAMPLE_RATE_HZ = 2, 32, 100.0
MAX_FRAMES_BEFORE_HANG = 20_000


class _SlowSource:
    """Delivers at half the target rate forever; the runner is stopped after `stop_after` frames."""

    def __init__(self, runner, stop_after):
        self._runner = runner
        self._stop_after = stop_after
        self.num_delivered = 0

    def frames(self):
        recording, _, _ = generate_paired_recording(
            num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
            duration_s=10.0, camera_frame_rate_hz=30.0, seed=300)
        timestamps_s = [frame.timestamp for frame in recording]
        for cycle in range(MAX_FRAMES_BEFORE_HANG // len(recording)):
            for frame, timestamp_s in zip(recording, timestamps_s):
                frame.timestamp = 2.0 * (timestamp_s + cycle * 10.0)
                self.num_delivered += 1
                if self.num_delivered == self._stop_after:
                    self._runner.is_running = False
                yield frame


class _Session:
    def __init__(self):
        config = ModelConfig(stage="presence", k=4, window=32, hop=16, frame_average=1)
        self.head = SimpleNamespace(config=config, contract=derive_pipeline_contract(config),
                                    classes_=np.array([0, 1]))

    def predict_window_proba(self, feature_vector):
        raise AssertionError("a slow stream must score nothing")


def test_stop_ends_the_dashboard_loop_while_every_window_is_skipped_as_too_slow(build_recording_runner,
                                                                              monkeypatch):
    runner, _ = build_recording_runner()
    runner.is_running = True
    monkeypatch.setattr(streamer, "load_calibration", lambda calibration_dir: calibrate_quiet_baseline(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=SAMPLE_RATE_HZ,
        num_frames=60, seed=7, baseline_packets=50))
    monkeypatch.setattr(streamer, "open_session", lambda mode, model_path: _Session())
    monkeypatch.setattr(runner, "_build_dashboard", lambda *arguments: None)
    source = _SlowSource(runner, stop_after=300)
    runner._serve_single_node(source, "cal", "model", "presence", None)
    assert source.num_delivered < 2 * 300
