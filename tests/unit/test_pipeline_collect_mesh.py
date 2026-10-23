import pytest

from wavetrace import pipeline


class _RecordedUseCase:
    def __init__(self, runs, capture):
        self._runs = runs
        self._capture = capture

    def run(self, node_ids, num_sessions):
        self._runs.append((self._capture, node_ids, num_sessions))


class _RecordedCapture:
    def __init__(self, port, node_ids, *, max_frames_per_link):
        self.options = (port, node_ids, max_frames_per_link)


class _Mode:
    def __init__(self, model_root):
        self.model_root = model_root


@pytest.fixture
def recorded(monkeypatch):
    runs = []
    monkeypatch.setattr(pipeline, "CollectMeshUseCase",
                        lambda plan, training_policy, *, capture, **options: _RecordedUseCase(runs, capture))
    monkeypatch.setattr(pipeline, "UdpLinkCapture", _RecordedCapture)
    return runs


def _collect(tmp_path, node_id=None):
    pipeline.collect_mesh(_Mode(str(tmp_path / "model")), port=5599, calibration_root=str(tmp_path / "cal"),
                          node_id=node_id, num_sessions=2, max_frames_per_link=64)


def test_missing_calibration_is_an_error_on_stderr_and_nothing_is_captured(tmp_path, recorded, capsys):
    _collect(tmp_path)
    assert capsys.readouterr().err == (f"\n[ERROR] no per-node calibrations in {tmp_path}/cal/node*. "
                                       "Run collect_baseline.py first.\n")
    assert recorded == [] and not (tmp_path / "model").exists()


def test_an_uncalibrated_node_filter_is_an_error_on_stderr(tmp_path, recorded, capsys):
    (tmp_path / "cal" / "node1").mkdir(parents=True)
    _collect(tmp_path, node_id=9)
    assert capsys.readouterr().err == f"\n[ERROR] node 9 has no calibration in {tmp_path}/cal.\n"
    assert recorded == []


@pytest.mark.parametrize(("node_id", "expected_node_ids"), [(None, [1, 2]), (2, [2])])
def test_captures_every_calibrated_node_or_only_the_filtered_one(tmp_path, recorded, node_id,
                                                                  expected_node_ids):
    for name in ("node1", "node2"):
        (tmp_path / "cal" / name).mkdir(parents=True)
    _collect(tmp_path, node_id=node_id)
    [(capture, node_ids, num_sessions)] = recorded
    assert (capture.options, node_ids, num_sessions) == ((5599, expected_node_ids, 64), expected_node_ids, 2)
    assert (tmp_path / "model").is_dir()
