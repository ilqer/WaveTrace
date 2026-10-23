"""Camera-labelled mesh collection: record every (tx_short, rx_node_id) link's CSI while a camera labels
the scene, write one camera-labelled dataset per usable link through a `CameraCollectPlan`, stack every
node into one heatmap dataset, then optionally train through a `CameraTrainingPolicy`."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from wavetrace import Label
from wavetrace.application.collect_mesh import (
    HeadTrainingReport, LinkDatasetRequest, TrainingPolicy, train_every_node,
)
from wavetrace.application.ports import CameraLabelCapture, TimedLinkCapture
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.domain.link_key import LinkKey

_NUM_LABELS_PER_PROGRESS_LINE = 30
_MIN_CAMERA_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class HeatmapDatasetReport:
    dataset_dir: str
    num_windows: int
    num_masked_windows: int


class CameraCollectPlan(Protocol):
    positive_class_name: str  # what a class-1 camera label means, e.g. 'present'

    def describe_link_dataset(self, link_key: LinkKey, frames: list, labels: list[Label]) -> LinkDatasetRequest:
        """frames: the link's resampled recording, at least one window long."""


class CameraTrainingPolicy(TrainingPolicy, Protocol):
    def train_heatmap(self, dataset_dir: str) -> HeadTrainingReport: ...


class CollectCameraUseCase:
    """write_link_dataset: saves one request's recording and dataset, returns the dataset dir.
    build_heatmap: stacks {rx_node_id: that node's resampled frames} and the labels into the heatmap dataset
    and saves it. training_policy None: nothing is trained. log: receives every message. on_capture_end: called
    once when the capture and the camera have both stopped, before anything is written."""

    def __init__(self, plan: CameraCollectPlan, training_policy: CameraTrainingPolicy | None, *,
                 capture: TimedLinkCapture, camera: CameraLabelCapture,
                 write_link_dataset: Callable[[LinkDatasetRequest], str],
                 build_heatmap: Callable[[dict[int, list], list[Label]], HeatmapDatasetReport],
                 log: Callable[[str], None], on_capture_end: Callable[[], None], target_sample_rate_hz: float,
                 min_window_frames: int):
        self._plan = plan
        self._training_policy = training_policy
        self._capture = capture
        self._camera = camera
        self._write_link_dataset = write_link_dataset
        self._build_heatmap = build_heatmap
        self._log = log
        self._on_capture_end = on_capture_end
        self._target_sample_rate_hz = target_sample_rate_hz
        self._min_window_frames = min_window_frames

    def run(self, node_ids: list[int], duration_s: float, stop: threading.Event) -> None:
        """Peak memory is every captured frame, O(L·F) over L links of F frames, plus one resampled copy
        per node for the heatmap and one link's at a time; resampling is O(L·F·A·S). A capture error stops
        the camera and propagates."""
        captured = self._capture_with_camera(duration_s, stop)
        if captured is None:
            return
        links, labels = captured
        num_positive_labels = sum(label.class_id == 1 for label in labels)
        self._log(f"labeled {num_positive_labels}/{len(labels)} frames positive ({self._plan.positive_class_name}).")
        dataset_dirs_by_node = self._write_link_datasets(node_ids, links, labels)
        heatmap_report = self._stack_heatmap(node_ids, links, labels)
        if heatmap_report is None and not any(dataset_dirs_by_node.values()):
            self._log("[ERROR] no usable CSI — is the mesh streaming on this port?")
            return
        if self._training_policy is not None:
            self._train(node_ids, dataset_dirs_by_node, heatmap_report)
        written_node_ids = [node_id for node_id in node_ids if dataset_dirs_by_node[node_id]]
        self._log(f"\ndone. link datasets for nodes {written_node_ids}; "
                  f"heatmap {'built' if heatmap_report is not None else 'skipped'}.")

    def _capture_with_camera(self, duration_s, stop):
        """(links, labels), or None after logging why the camera gave no labels. The camera labels on a
        worker thread while this one records the CSI; the capture's end stops it."""
        camera_result = {}
        camera_stop = threading.Event()
        progress = {"num_labels": 0, "num_positive": 0}

        def report_progress(label):
            progress["num_labels"] += 1
            progress["num_positive"] += int(label.class_id == 1)
            if progress["num_labels"] % _NUM_LABELS_PER_PROGRESS_LINE == 0:
                self._log(f"   live: {progress['num_positive']}/{progress['num_labels']} frames "
                          f"{self._plan.positive_class_name}")

        def record_labels():
            try:
                camera_result["labels"] = self._camera.record_labels(duration_s, stop=camera_stop,
                                                                     on_label=report_progress)
            except Exception as error:  # camera permission or busy; reported after the join
                camera_result["error"] = error

        self._log(f"   [CAPTURING] {duration_s:g}s of camera labels and mesh CSI...")
        camera_join_timeout_s = max(_MIN_CAMERA_JOIN_TIMEOUT_S, 0.1 * duration_s)
        camera_thread = threading.Thread(target=record_labels, daemon=True)
        camera_thread.start()
        try:
            links = self._capture.capture_for(duration_s, stop=stop)
        finally:
            camera_stop.set()
            camera_thread.join(timeout=camera_join_timeout_s)
        self._on_capture_end()
        if camera_thread.is_alive():
            self._log(f"[ERROR] the camera did not stop within {camera_join_timeout_s:g}s of the capture.")
            return None
        if "error" in camera_result:
            self._log(f"[ERROR] webcam: {camera_result['error']}")
            return None
        if not camera_result["labels"]:
            self._log("[ERROR] no webcam frames labeled — check the camera permission and index.")
            return None
        return links, camera_result["labels"]

    def _write_link_datasets(self, node_ids, links, labels):
        dataset_dirs_by_node = {node_id: [] for node_id in node_ids}
        for link_key in sorted(links, key=lambda link_key: (link_key[1], link_key[0])):
            tx_short, rx_node_id = link_key
            if rx_node_id not in dataset_dirs_by_node:
                continue
            frames = resample_uniform(links[link_key], self._target_sample_rate_hz)
            if len(frames) < self._min_window_frames:
                self._log(f"   [SKIP] link {tx_short}->{rx_node_id}: {len(frames)} frames "
                          f"(< {self._min_window_frames})")
                continue
            dataset_dir = self._write_link_dataset(self._plan.describe_link_dataset(link_key, frames, labels))
            dataset_dirs_by_node[rx_node_id].append(dataset_dir)
            self._log(f"   [OK]   link {tx_short}->{rx_node_id} -> {dataset_dir}")
        return dataset_dirs_by_node

    def _stack_heatmap(self, node_ids, links, labels):
        # the heatmap head takes one channel per node, so each node's links are merged and resampled as one stream
        frames_by_node = {}
        for node_id in node_ids:
            node_frames = [frame for link_key, frames in links.items() if link_key[1] == node_id
                           for frame in frames]
            if node_frames:
                frames_by_node[node_id] = resample_uniform(node_frames, self._target_sample_rate_hz)
        if not frames_by_node:
            return None
        report = self._build_heatmap(frames_by_node, labels)
        self._log(f"   [OK]   heatmap: stacked {len(frames_by_node)} nodes -> {report.dataset_dir} "
                  f"({report.num_windows} windows, {report.num_masked_windows} with masks)")
        return report

    def _train(self, node_ids, dataset_dirs_by_node, heatmap_report):
        train_every_node(self._training_policy, node_ids, dataset_dirs_by_node, log=self._log, log_error=self._log,
                         should_stop=lambda: False)
        if heatmap_report is not None:
            self._log(self._training_policy.train_heatmap(heatmap_report.dataset_dir).line)
