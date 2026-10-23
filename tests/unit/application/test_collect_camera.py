import threading

import numpy as np
import pytest

from wavetrace import CsiFrame, Label
from wavetrace.application import collect_camera
from wavetrace.application.collect_camera import CollectCameraUseCase, HeatmapDatasetReport
from wavetrace.application.collect_mesh import HeadTrainingReport

SAMPLE_RATE_HZ = 128.0  # binary-exact frame spacing, so resampling keeps the frame count
WINDOW_FRAMES = 4


def _frames(num_frames, amplitude=1.0, rx_node_id=1):
    frames = []
    for i in range(num_frames):
        frame = CsiFrame(1, 4)
        frame.grid[:, :] = amplitude
        frame.timestamp = i / SAMPLE_RATE_HZ
        frame.node_id = rx_node_id
        frames.append(frame)
    return frames


def _labels(class_ids):
    labels = []
    for i, class_id in enumerate(class_ids):
        label = Label()
        label.class_id = class_id
        label.timestamp = i / SAMPLE_RATE_HZ
        labels.append(label)
    return labels


class _FakeCapture:
    def __init__(self, links=None, error=None):
        self._links = links or {}
        self._error = error
        self.stop = None

    def capture_for(self, duration_s, *, stop):
        self.stop = stop
        if self._error is not None:
            raise self._error
        return self._links


class _FakeCamera:
    def __init__(self, labels=None, error=None):
        self._labels = labels if labels is not None else _labels([0, 1, 1])
        self._error = error
        self.stop = None

    def record_labels(self, duration_s, *, stop, on_label):
        self.stop = stop
        if self._error is not None:
            raise self._error
        for label in self._labels:
            on_label(label)
        return self._labels


class _FakePlan:
    positive_class_name = "present"

    def describe_link_dataset(self, link_key, frames, labels):
        return link_key, frames, labels


class _FakeTraining:
    def __init__(self):
        self.dataset_dirs_by_node = {}
        self.heatmap_dataset_dirs = []

    def format_training_banner(self):
        return "training..."

    def train_node(self, node_id, dataset_dirs):
        self.dataset_dirs_by_node[node_id] = dataset_dirs
        return [HeadTrainingReport(f"node {node_id}: {len(dataset_dirs)} datasets", node_id)]

    def format_trained_summary(self, trained_head_keys):
        return f"trained {trained_head_keys}"

    def format_untrained_error(self):
        return "nothing trained"

    def train_heatmap(self, dataset_dir):
        self.heatmap_dataset_dirs.append(dataset_dir)
        return HeadTrainingReport(f"heatmap from {dataset_dir}", "heatmap")


class _Run:
    def __init__(self, links, *, camera=None, training=None, capture=None, node_ids=(1,)):
        self.capture = capture or _FakeCapture(links)
        self.camera = camera or _FakeCamera()
        self.messages = []
        self.written = []
        self.heatmap_frames_by_node = None
        self.num_capture_ends = 0
        self.stop = threading.Event()

        def count_capture_end():
            self.num_capture_ends += 1

        def write_link_dataset(request):
            self.written.append(request)
            return f"ds{len(self.written)}"

        def build_heatmap(frames_by_node, labels):
            self.heatmap_frames_by_node = frames_by_node
            return HeatmapDatasetReport(dataset_dir="heatmap_ds", num_windows=3, num_masked_windows=2)

        CollectCameraUseCase(_FakePlan(), training, capture=self.capture, camera=self.camera,
                             write_link_dataset=write_link_dataset, build_heatmap=build_heatmap,
                             log=self.messages.append, on_capture_end=count_capture_end,
                             target_sample_rate_hz=SAMPLE_RATE_HZ,
                             min_window_frames=WINDOW_FRAMES).run(list(node_ids), 1.0, self.stop)


def test_each_tx_link_is_its_own_dataset_never_merged_with_another_link_of_its_node():
    links = {("aa", 1): _frames(8, amplitude=1.0), ("bb", 1): _frames(8, amplitude=5.0)}
    run = _Run(links)
    assert [link_key for link_key, _, _ in run.written] == [("aa", 1), ("bb", 1)]
    for (_, frames, _), amplitude in zip(run.written, (1.0, 5.0)):
        assert len(frames) == 8
        assert np.allclose([np.abs(frame.grid) for frame in frames], amplitude)


def test_every_link_dataset_is_labelled_by_the_camera():
    labels = _labels([0, 1, 1, 0])
    run = _Run({("aa", 1): _frames(8)}, camera=_FakeCamera(labels))
    [(_, _, request_labels)] = run.written
    assert request_labels is labels


def test_a_link_under_one_window_or_off_the_calibrated_nodes_is_not_written():
    links = {("aa", 1): _frames(WINDOW_FRAMES - 1), ("bb", 1): _frames(8), ("cc", 9): _frames(8, rx_node_id=9)}
    run = _Run(links)
    assert [link_key for link_key, _, _ in run.written] == [("bb", 1)]
    assert f"   [SKIP] link aa->1: {WINDOW_FRAMES - 1} frames (< {WINDOW_FRAMES})" in run.messages


def test_the_heatmap_stacks_each_nodes_links_merged_into_one_stream():
    links = {("aa", 1): _frames(8), ("bb", 1): _frames(8), ("aa", 2): _frames(8, rx_node_id=2)}
    run = _Run(links, node_ids=(1, 2))
    assert sorted(run.heatmap_frames_by_node) == [1, 2]
    assert len(run.heatmap_frames_by_node[2]) == 8
    assert {frame.node_id for frame in run.heatmap_frames_by_node[1]} == {1}


def test_the_capture_end_is_signalled_exactly_once_and_never_logged():
    run = _Run({("aa", 1): _frames(8)}, training=_FakeTraining())
    assert run.num_capture_ends == 1
    assert not any("\a" in message for message in run.messages)


def test_training_gets_this_runs_dataset_dirs_per_node_then_trains_the_heatmap():
    training = _FakeTraining()
    run = _Run({("aa", 1): _frames(8), ("bb", 1): _frames(8), ("aa", 2): _frames(8, rx_node_id=2)},
               training=training, node_ids=(1, 2))
    assert training.dataset_dirs_by_node == {1: ["ds1", "ds2"], 2: ["ds3"]}
    assert training.heatmap_dataset_dirs == ["heatmap_ds"]
    assert run.messages[-6:-1] == ["training...", "node 1: 2 datasets", "node 2: 1 datasets", "trained [1, 2]",
                                   "heatmap from heatmap_ds"]


def test_without_a_training_policy_nothing_is_trained():
    run = _Run({("aa", 1): _frames(8)})
    assert not any("train" in message for message in run.messages)


def test_the_capture_gets_the_callers_stop_and_its_end_stops_the_camera():
    run = _Run({("aa", 1): _frames(8)})
    assert run.capture.stop is run.stop
    assert run.camera.stop is not run.stop and run.camera.stop.is_set()


def test_a_capture_error_stops_the_camera_and_propagates():
    camera = _FakeCamera()
    with pytest.raises(OSError, match="address already in use"):
        _Run({}, camera=camera, capture=_FakeCapture(error=OSError("address already in use")))
    assert camera.stop.is_set()


@pytest.mark.parametrize(("camera", "error_message"), [
    (_FakeCamera(error=RuntimeError("cannot open webcam index 3")), "[ERROR] webcam: cannot open webcam index 3"),
    (_FakeCamera(labels=[]), "[ERROR] no webcam frames labeled — check the camera permission and index."),
])
def test_a_camera_without_labels_is_an_error_and_nothing_is_written(camera, error_message):
    run = _Run({("aa", 1): _frames(8)}, camera=camera, training=_FakeTraining())
    assert run.messages[-1] == error_message
    assert run.written == [] and run.heatmap_frames_by_node is None


def test_no_usable_csi_is_an_error_and_nothing_is_trained():
    training = _FakeTraining()
    run = _Run({}, training=training)
    assert run.messages[-1] == "[ERROR] no usable CSI — is the mesh streaming on this port?"
    assert training.dataset_dirs_by_node == {}


class _StuckCamera:
    """Ignores its stop until the test releases it."""

    def __init__(self):
        self.release = threading.Event()

    def record_labels(self, duration_s, *, stop, on_label):
        self.release.wait()
        return _labels([1])


def test_a_camera_still_labelling_after_the_join_timeout_is_an_error_and_nothing_is_written(monkeypatch):
    monkeypatch.setattr(collect_camera, "_MIN_CAMERA_JOIN_TIMEOUT_S", 0.05)
    camera = _StuckCamera()
    try:
        run = _Run({("aa", 1): _frames(8)}, camera=camera, training=_FakeTraining())
    finally:
        camera.release.set()
    assert run.messages[-1].startswith("[ERROR] the camera did not stop within")
    assert run.written == [] and run.heatmap_frames_by_node is None
