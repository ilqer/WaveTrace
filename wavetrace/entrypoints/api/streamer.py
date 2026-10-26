import asyncio
import contextlib
import json
import os
import threading
import time
import traceback
from collections import deque

import numpy as np

from wavetrace.adapters.recognition import is_sklearn_backend
from wavetrace.adapters.recognition.session import open_session
from wavetrace.adapters.source import UdpLinkBatchReceiver, build_synthetic_source, parse_time_spans
from wavetrace.adapters.store.calibration import find_node_calibration_dirs, load_calibration
from wavetrace.adapters.store.dataset import load_dataset_labels
from wavetrace.adapters.store.model_runs import HEATMAP_FILE_NAME, resolve_run_dir
from wavetrace.application.behaviours import build_alert_behaviours
from wavetrace.application.collect_mesh import HeadTrainingReport, train_every_node
from wavetrace.application.ports import LinkBatchReceiver
from wavetrace.application.serve import WindowView, iter_window_verdicts
from wavetrace.application.serve_mesh import FusedVerdict, ServeMeshUseCase
from wavetrace.diagnostics import NodeHealthMeter, cluster_sync, dataset_report
from wavetrace.domain.guard import WeaponAlertEvent
from wavetrace.domain.localize import Localizer
from wavetrace.domain.recognition.verdict import decide_class
from wavetrace.groundtruth.webcam import COCO_WEAPON_CLASSES, WebcamLabelCapture
from wavetrace.pipeline import BUFFER_S, CHUNK_S, LINK_TIMEOUT_S, calibrate_source, collect_camera, collect_source
from wavetrace.recognition import link_health
from wavetrace.recognition.camera_collect_modes import (
    CameraPresenceTrainingPolicy, default_camera_model_root, find_heatmap_capture_root, train_heatmap_head,
)
from wavetrace.recognition.collect_modes import (
    CountCollectMode, PresenceCollectMode, WeaponTrainingPolicy, default_model_root, find_collected_max_count,
)
from wavetrace.recognition.serve_modes import CountServeMode, PresenceServeMode, WeaponServeMode

_OCCUPANCY_GRID_SIZE = 16
# labels change slowly next to the 1.28 s CSI window, so 5 Hz is enough and costs
# a third of the CPU that 15 Hz does
_CAMERA_FRAME_RATE_HZ = 5.0


def _occupancy_from_variance(image: np.ndarray, grid_size: int = _OCCUPANCY_GRID_SIZE) -> np.ndarray:
    """Tiles the variance when K < grid_size², because zero-padding it would show as black bars."""
    variance = image.var(axis=1).astype(np.float32)  # (K,)
    cell_count = grid_size * grid_size
    if variance.size < cell_count:
        repeat_count = (cell_count + variance.size - 1) // variance.size
        variance = np.tile(variance, repeat_count)[:cell_count]
    elif variance.size > cell_count:
        step = variance.size // cell_count
        variance = variance[:step * cell_count].reshape(cell_count, step).mean(axis=1)
    lowest, highest = variance.min(), variance.max()
    if highest > lowest:
        variance = (variance - lowest) / (highest - lowest)
    return variance.astype(np.float32)


def _predict_occupancy_grid(head, image: np.ndarray) -> np.ndarray:
    if head is None:
        return _occupancy_from_variance(image)
    try:
        image_batch = image[np.newaxis]   # (1, K, W)
        return head.predict_heatmap(image_batch)[0].flatten().astype(np.float32)
    except Exception:
        return _occupancy_from_variance(image)


def _class_name(mode: str, class_id: int) -> str:
    class_id = int(class_id)
    if mode == "presence":
        return {0: "empty", 1: "present"}.get(class_id, str(class_id))
    if mode == "weapon":
        return {0: "no weapon", 1: "weapon"}.get(class_id, str(class_id))
    return str(class_id)


_SERVE_MODES = {"presence": PresenceServeMode, "count": CountServeMode, "weapon": WeaponServeMode}


def _is_mesh_root(model_path: str) -> bool:
    return os.path.isdir(model_path) and any(
        name.startswith("node") and os.path.isdir(os.path.join(model_path, name)) for name in os.listdir(model_path))


def _read_antenna_weights(head) -> list | None:
    try:
        from wavetrace.recognition.explain import cnn_channel_weights
        channel_weights = cnn_channel_weights(head)
    except Exception:
        return None
    return channel_weights.tolist() if channel_weights is not None else None


class HealthObservedReceiver(LinkBatchReceiver):
    """Shows every received frame to `health_meter` before passing the batch on. O(frames·S) per receive."""

    def __init__(self, receiver: LinkBatchReceiver, health_meter: NodeHealthMeter):
        self._receiver = receiver
        self._health_meter = health_meter

    def receive(self):
        frames_by_link = self._receiver.receive()
        for frames in frames_by_link.values():
            for frame in frames:
                self._health_meter.observe(frame)
        return frames_by_link

    def close(self):
        self._receiver.close()


class _DashboardPublisher:
    """Publishes one verdict as the dashboard's messages, in this order: alert events, the verdict, the
    stream frame, then at most one telemetry snapshot per second of verdict time. class_ids: the proba's
    column order."""

    def __init__(self, emit_inference, emit_stream, emit_telemetry, *, mode: str, class_ids: list[int],
                 health_meter: NodeHealthMeter, heatmap_head, antenna_weights: list | None):
        self._emit_inference = emit_inference
        self._emit_stream = emit_stream
        self._emit_telemetry = emit_telemetry
        self._mode = mode
        self._class_ids = class_ids
        # a count verdict has no single positive class to trace
        self._positive_class_index = class_ids.index(1) if mode != "count" and 1 in class_ids else -1
        self._alert_behaviours = build_alert_behaviours(mode)
        self._health_meter = health_meter
        self._heatmap_head = heatmap_head
        self._antenna_weights = antenna_weights
        self._voter_trace: deque[float] = deque(maxlen=60)
        self._is_alert_active = False
        self._last_telemetry_time_s = 0.0

    def publish(self, timestamp_s: float, proba: np.ndarray, window_view: WindowView, node_power: dict[int, float],
                *, location=None, link_stats: list[dict] | None = None) -> None:
        """O(K·W) for the stream frame, plus the health snapshot on a telemetry tick."""
        predicted_class, confidence = decide_class(proba, self._class_ids)
        inference_payload = {"t": float(timestamp_s), "class": predicted_class, "conf": confidence,
                             "mode": self._mode}
        for alert_behaviour in self._alert_behaviours:
            alert_event = alert_behaviour.observe_window(timestamp_s, predicted_class, proba)
            if alert_event:
                self._is_alert_active = isinstance(alert_event, WeaponAlertEvent)
                self._emit_inference({**inference_payload, **alert_event.to_dict()})
        if self._positive_class_index >= 0:
            self._voter_trace.append(float(proba[self._positive_class_index]))
        if location is not None:
            inference_payload["pos"] = [float(location.x_m), 0.0, float(location.y_m)]
        self._emit_inference(inference_payload)

        occupancy_grid = _predict_occupancy_grid(self._heatmap_head, window_view.image)
        node_power_items = sorted(node_power.items())
        intercarrier_features = window_view.intercarrier_features
        stream_payload = {
            "t": float(timestamp_s), "image": window_view.image.tolist(),
            "features": window_view.features.tolist(),
            "ic": None if intercarrier_features is None else intercarrier_features.tolist(),
            "antennas": [power for _, power in node_power_items],
            "node_ids": [int(node_id) for node_id, _ in node_power_items],
            "heatmap_grid": occupancy_grid.tolist(), "grid_size": _OCCUPANCY_GRID_SIZE,
        }
        if location is not None:
            stream_payload["spatial"] = {"x": float(location.x_m), "y": float(location.y_m),
                                         "conf": float(location.confidence),
                                         "heatmap": location.heatmap.flatten().tolist()}
        self._emit_stream(stream_payload)

        if timestamp_s - self._last_telemetry_time_s < 1.0:
            return
        self._last_telemetry_time_s = timestamp_s
        telemetry_payload = {
            "nodes": self._health_meter.snapshot(),
            "sync": cluster_sync(self._health_meter),
            "heatmap": occupancy_grid.tolist(),
            "grid": _OCCUPANCY_GRID_SIZE,
            "antenna_weights": self._antenna_weights,
            "alert_active": self._is_alert_active,
            "drift_ratio": 0.0,
            "voter_trace": list(self._voter_trace),
            "contribution": {_class_name(self._mode, class_id): round(float(proba[j]), 3)
                             for j, class_id in enumerate(self._class_ids)},
        }
        if link_stats is not None:
            telemetry_payload["links"] = link_stats
        self._emit_telemetry(telemetry_payload)


def _publish_fused_verdict(dashboard: _DashboardPublisher, fused_verdict: FusedVerdict,
                           rx_node_ids: tuple[int, ...]) -> None:
    """Shows the last scored link's window; a node's power is the mean |H| of its last live link's frames.
    O(frames·S) over the live links."""
    node_power = dict.fromkeys(rx_node_ids, 0.0)
    link_stats = []
    window_view = None
    for link_report in fused_verdict.link_reports:
        tx_mac_suffix, rx_node_id = link_report.link_key
        node_power[rx_node_id] = float(np.mean([np.abs(frame.grid).mean() for frame in link_report.frames]))
        delivered_rate_hz, missing_fraction = link_health(link_report.frames)
        link_stats.append({"tx": tx_mac_suffix, "rx": rx_node_id,
                           "hz": round(delivered_rate_hz, 1), "miss": round(missing_fraction, 3)})
        if link_report.score is not None:
            window_view = link_report.score.window_view
    dashboard.publish(fused_verdict.now_s, fused_verdict.blended_proba, window_view, node_power,
                      link_stats=link_stats)


class FrameSnooper:
    def __init__(self, source, health_meter=None):
        self._source = source
        self.latest_grid = None
        self.latest_frame = None
        # each UDP/mesh frame carries one antenna, so this power is per RX board
        self.node_power: dict[int, float] = {}
        self._meter = health_meter

    def frames(self):
        for frame in self._source.frames():
            self.latest_grid = np.asarray(frame.grid)
            self.latest_frame = frame
            self.node_power[getattr(frame, "node_id", 0)] = float(np.abs(self.latest_grid).mean())
            if self._meter is not None:
                self._meter.observe(frame)
            yield frame


class WaveTraceRunner:
    def __init__(self, loop: asyncio.AbstractEventLoop,
                 inference_queue: asyncio.Queue, stream_queue: asyncio.Queue,
                 logs_queue: asyncio.Queue, training_queue: asyncio.Queue,
                 telemetry_queue: asyncio.Queue | None = None):
        self.loop = loop
        self.inference_queue = inference_queue
        self.stream_queue = stream_queue
        self.logs_queue = logs_queue
        self.training_queue = training_queue
        self.telemetry_queue = telemetry_queue
        self.is_running = False
        self._stop_event = threading.Event()
        self.localizer = None

    def log(self, message: str):
        time_of_day = time.strftime('%H:%M:%S')
        asyncio.run_coroutine_threadsafe(self.logs_queue.put(f"[{time_of_day}] {message}"), self.loop)

    def _emit_inference(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.inference_queue.put(json.dumps(payload)), self.loop)

    def _build_source(self, request):
        self.localizer = Localizer(request.antennas, range_enabled=True) if request.antennas >= 2 else None

        if not request.synthetic:
            port = request.udp_port
            self.log(f"[HW] UDP listener on :{port} — nodes push CSI here (PC_IP set in firmware)")
            self.log(f"[HW] Camera: {request.cam_url}")
            from wavetrace.adapters.source import UdpSource, UdpSourceOptions
            return UdpSource(UdpSourceOptions(port=port, timeout_seconds=60.0))

        self.log("[SIM] Synthetic source (no hardware connected).")
        presence_spec = request.col_spans if request.action == "collect" else "0:5,10:15,20:25,30:35,40:45,50:55"
        return build_synthetic_source(
            num_antennas=request.antennas, num_subcarriers=request.subcarriers, sample_rate_hz=request.fs,
            duration_s=request.duration, presence_spans=parse_time_spans(presence_spec),
            weapon_spans=parse_time_spans("2:5,12:15,22:25,32:35,42:45,52:55"), weapon_signature_depth=0.5,
            seed=request.seed)

    def run_inference_request(self, request):
        source = self._build_source(request)
        self.log(f"Loading Calibration: {request.calibration}")
        self.log(f"Loading Model: {request.model}")
        self.run_inference(source, request.calibration, request.model, request.mode, port=request.udp_port)

    def run_calibration_request(self, request):
        """Always ends with `pipeline_done` and `is_running` cleared, also on a stop or an error."""
        self.is_running = True
        try:
            source = self._build_source(request)
            if not self.is_running: return
            self.log(f"Starting Calibration -> {request.cal_out}")
            path, _ = calibrate_source(source, request.cal_out, baseline_packets=request.baseline_packets,
                                       use_gain_lock=request.gainLock)
            self.log(f"Calibration complete: {path}")
        except Exception as error:
            self.log(f"ERROR Calibration failed: {type(error).__name__}: {error}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False

    def run_collection_request(self, request):
        """Always ends with `pipeline_done` and `is_running` cleared, also on a stop or an error."""
        self.is_running = True
        try:
            source = self._build_source(request)
            if not self.is_running: return
            self.log(f"Collecting {request.col_stage} dataset "
                     f"(window={request.col_window}, hop={request.col_hop})")
            spans = parse_time_spans(request.col_spans)
            path, dataset = collect_source(source, request.calibration, "output/dataset_ui", spans,
                                           stage=request.col_stage, window=request.col_window,
                                           hop=request.col_hop,
                                           subtract_ic_baseline=request.subtract_ic_baseline)
            self.log(f"Dataset saved ({dataset.y.size} samples) -> {path}")
        except Exception as error:
            self.log(f"ERROR Collection failed: {type(error).__name__}: {error}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False

    def _emit_training(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.training_queue.put(json.dumps(payload)), self.loop)

    def run_training_request(self, request):
        """Trains what the matching collect script trains, on that script's pool under the capture-profile root
        `request.train_data`, into its default model root (`request.train_out` is ignored); backend 'heatmap'
        trains the heatmap head on the one camera session dir `request.train_data` instead. Sends one `done`
        with the last trained head's metrics, then always `pipeline_done`; `stop` ends a cnn or heatmap fit at
        its next epoch, and any other training before its next head."""
        self.is_running = True
        try:
            self.log(f"Training {request.col_stage}/{request.train_backend}...")
            if request.train_backend == "heatmap":
                trained_reports = self._train_heatmap(request)
            else:
                trained_reports = self._train_stage_heads(request)
            if trained_reports:
                self._emit_training({"type": "done", "metrics": trained_reports[-1].metrics})
        except KeyboardInterrupt:
            self.log("Training stopped by UI.")
        except Exception as error:
            self.log(f"ERROR Training failed: {type(error).__name__}: {error}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False

    def _log_each_line(self, message: str) -> None:
        """The training policies' messages carry console newlines; each non-blank line logs on its own."""
        for line in message.splitlines():
            if line.strip():
                self.log(line)

    def _train_stage_heads(self, request) -> list[HeadTrainingReport]:
        if request.camera_collect and request.col_stage == "count":
            self.log("ERROR: no camera count pool; untick camera (camera sessions are labelled for presence only).")
            return []
        training_policy = self._build_training_policy(request)
        dataset_dirs_by_node = training_policy.find_dataset_dirs_by_node()
        if not dataset_dirs_by_node:
            self.log(f"No {request.col_stage} datasets under {request.train_data}: the dataset path must be a "
                     "capture-profile root such as data/2g4_ht40 (run 'collect' first if it is one).")
            return []
        self._emit_train_init([dataset_dir for dataset_dirs in dataset_dirs_by_node.values()
                               for dataset_dir in dataset_dirs])
        return train_every_node(training_policy, sorted(dataset_dirs_by_node), dataset_dirs_by_node,
                                log=self._log_each_line, log_error=self._log_each_line,
                                should_stop=lambda: not self.is_running)

    def _build_training_policy(self, request):
        """The collect script's training policy for the request's stage, camera toggle and backend."""
        root, stage = request.train_data, request.col_stage
        if stage == "weapon":
            return WeaponTrainingPolicy(root=root, model_root=default_model_root(root, stage),
                                        is_per_link=request.per_link, backend=request.train_backend,
                                        report=self._report_epoch)
        backend = request.train_backend
        if not is_sklearn_backend(backend):
            self.log(f"{stage.capitalize()} only supports mlp/svm backends; ignoring '{backend}', using mlp.")
            backend = "mlp"
        if request.camera_collect:
            return CameraPresenceTrainingPolicy(root=root, model_root=default_camera_model_root(root, "presence"),
                                                backend=backend)
        if stage == "count":
            return CountCollectMode(root=root, model_root=default_model_root(root, stage),
                                    max_count=find_collected_max_count(root), backend=backend)
        return PresenceCollectMode(root=root, model_root=default_model_root(root, stage), backend=backend)

    def _train_heatmap(self, request) -> list[HeadTrainingReport]:
        dataset_dir = request.train_data
        root = find_heatmap_capture_root(dataset_dir)
        if root is None:
            self.log(f"{dataset_dir} is not a camera session's heatmap dataset (<root>/cam_ds/heatmap/<session>).")
            return []
        self._emit_train_init([dataset_dir])
        report = train_heatmap_head(dataset_dir, default_camera_model_root(root, request.col_stage),
                                    report=self._report_epoch)
        self._log_each_line(report.line)
        return [report] if report.trained_head_key is not None else []

    def _emit_train_init(self, dataset_dirs: list) -> None:
        """Reads only the manifests, O(windows of the pool)."""
        self._emit_training({"type": "train_init",
                             **dataset_report(load_dataset_labels(dataset_dir) for dataset_dir in dataset_dirs)})

    def _report_epoch(self, epoch: int, metrics: dict) -> None:
        if not self.is_running:
            raise KeyboardInterrupt("training stopped by UI")
        self._emit_training({
            "type": "epoch",
            "epoch": epoch,
            "loss": float(metrics.get("loss", 0.0)),
            "loss_std": float(metrics.get("loss_std", 0.0)),  # within-epoch batch spread; the client draws it as a band
            "accuracy": float(metrics.get("acc", metrics.get("accuracy", 0.0))),
        })
        self.log(f"Epoch {epoch}: loss={metrics.get('loss', 0):.4f}")

    def _emit_telemetry(self, payload: dict) -> None:
        if self.telemetry_queue is not None:
            asyncio.run_coroutine_threadsafe(
                self.telemetry_queue.put(json.dumps(payload)), self.loop)

    def _emit_stream(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(self.stream_queue.put(json.dumps(payload)), self.loop)

    def run_inference(self, source, calibration_dir: str, model_path: str, mode: str, port: int = 9876):
        """Serves `mode` ('presence', 'count' or 'weapon') until the stream ends or `stop`: a mesh model root
        (node*/ dirs) from udp/`port`, any other model as one node over `source`."""
        self.is_running = True
        try:
            heatmap_head = self._load_heatmap_head(model_path)
            if _is_mesh_root(model_path):
                self._serve_mesh(calibration_dir, model_path, mode, port, heatmap_head)
            else:
                self._serve_single_node(source, calibration_dir, model_path, mode, heatmap_head)
        except Exception as error:
            self.log(f"ERROR Runner: {type(error).__name__}: {error}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False

    def _load_heatmap_head(self, model_path: str):
        model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        if resolve_run_dir(model_dir, HEATMAP_FILE_NAME) is None:
            return None
        try:
            from wavetrace.recognition.heatmap import HeatmapHead
            heatmap_head = HeatmapHead.load(os.path.join(model_dir, HEATMAP_FILE_NAME))
        except Exception as heatmap_error:
            self.log(f"[HM] WARNING: heatmap load failed ({heatmap_error}); using fallback")
            return None
        self.log(f"[HM] Heatmap head loaded ({heatmap_head.grid}×{heatmap_head.grid})")
        return heatmap_head

    def _build_dashboard(self, mode: str, class_ids: list[int], health_meter: NodeHealthMeter, heatmap_head,
                         antenna_weights: list | None) -> _DashboardPublisher:
        return _DashboardPublisher(self._emit_inference, self._emit_stream, self._emit_telemetry, mode=mode,
                                   class_ids=class_ids, health_meter=health_meter, heatmap_head=heatmap_head,
                                   antenna_weights=antenna_weights)

    def _serve_mesh(self, calibration_dir: str, model_path: str, mode: str, port: int, heatmap_head) -> None:
        """Stops within one receive timeout (0.5 s), or the fuse in progress, of `stop`."""
        self.log(f"Mesh setup detected. Loading per-node models from {model_path}...")
        serve_mode = _SERVE_MODES[mode].load(calibration_dir, model_path)
        if not serve_mode.nodes:
            self.log(f"ERROR: No valid mesh models found in {model_path}")
            return
        calibration_by_rx_node = serve_mode.calibration_by_rx_node
        rx_node_ids = tuple(calibration_by_rx_node)
        health_meter = NodeHealthMeter(calibration_by_node=calibration_by_rx_node)
        dashboard = self._build_dashboard(mode, serve_mode.class_ids, health_meter, heatmap_head, None)
        receiver = HealthObservedReceiver(UdpLinkBatchReceiver(port, timeout_s=0.5), health_meter)
        use_case = ServeMeshUseCase(serve_mode, clock=time.time, chunk_s=CHUNK_S, link_timeout_s=LINK_TIMEOUT_S,
                                    buffer_s=BUFFER_S)
        self.log("Stream started.")
        with contextlib.closing(use_case.iter_fused(receiver)) as fused_verdicts:
            for fused_verdict in fused_verdicts:
                if not self.is_running:
                    break
                if fused_verdict is not None and fused_verdict.blended_proba is not None:
                    _publish_fused_verdict(dashboard, fused_verdict, rx_node_ids)
        self.log("Stream ended.")

    def _serve_single_node(self, source, calibration_dir: str, model_path: str, mode: str, heatmap_head) -> None:
        """The head's contract sets the input, gain lock, frame averaging and baselines."""
        self.log("Single-node setup detected.")
        calibration_result, gain_lock = load_calibration(calibration_dir)
        head_mode = "presence" if mode == "count" else mode
        session = open_session(head_mode, model_path)
        # the single calibration has no node id of its own, so it covers every node the stream carries
        health_meter = NodeHealthMeter(default_calibration=calibration_result)
        dashboard = self._build_dashboard(mode, [int(class_id) for class_id in session.head.classes_],
                                          health_meter, heatmap_head, _read_antenna_weights(session.head))
        snooper = FrameSnooper(source, health_meter=health_meter)
        last_window_timestamp_s = 0.0
        self.log("Stream started.")
        with contextlib.closing(iter_window_verdicts(snooper, calibration_result, gain_lock, session,
                                                     head_mode)) as verdicts:
            for verdict in verdicts:
                if not self.is_running:
                    break
                # a recording or synthetic source yields faster than real time, so pace it to its timestamps
                if last_window_timestamp_s > 0:
                    window_gap_s = verdict.timestamp_s - last_window_timestamp_s
                    if window_gap_s > 0:
                        time.sleep(window_gap_s)
                last_window_timestamp_s = verdict.timestamp_s
                location = None
                if self.localizer and snooper.latest_frame is not None:
                    location = self.localizer.locate(snooper.latest_frame, timestamp=verdict.timestamp_s)
                dashboard.publish(verdict.timestamp_s, verdict.proba, verdict.window_view, snooper.node_power,
                                  location=location)
        self.log("Stream ended.")

    def run_camera_collection_request(self, request):
        """The camera labels on a worker thread while this thread records the mesh's CSI under
        `request.train_data`; `stop` ends both within a second."""
        self.is_running = True
        try:
            self._collect_camera(request)
        except Exception as error:
            self.log(f"ERROR Camera collection: {type(error).__name__}: {error}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False

    def _collect_camera(self, request):
        calibration_dir_by_node_id = find_node_calibration_dirs(request.calibration)
        if not calibration_dir_by_node_id and os.path.exists(os.path.join(request.calibration, "meta.json")):
            # older flat calibration dirs have no node*/ level; they count as node 0
            calibration_dir_by_node_id = {0: request.calibration}
        if not calibration_dir_by_node_id:
            self.log(f"ERROR: no calibration found at '{request.calibration}' — run Calib first")
            return
        self.log(f"nodes: {sorted(calibration_dir_by_node_id)}. loading YOLO-seg (first run downloads weights)...")
        try:
            camera = WebcamLabelCapture(stage=request.col_stage, camera_index=request.cam_index,
                                        weights_path=request.yolo_weights, weapon_classes=COCO_WEAPON_CLASSES,
                                        confidence_threshold=0.35, grid=_OCCUPANCY_GRID_SIZE,
                                        frame_rate_hz=_CAMERA_FRAME_RATE_HZ)
        except Exception as yolo_error:
            self.log(f"ERROR: YOLO init failed: {yolo_error}")
            return
        collect_camera(camera, stage=request.col_stage, port=request.udp_port, root=request.train_data,
                       calibration_dir_by_node_id=calibration_dir_by_node_id, subject_id="cam",
                       duration_s=request.duration, stop=self._stop_event, model_root=None, log=self.log,
                       on_capture_end=lambda: None)

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        self.log("System halt requested.")

