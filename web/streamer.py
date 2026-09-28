import asyncio
import time
import traceback
import numpy as np
import json
from collections import deque

from wavetrace.calibration import load_calibration, build_image_baseline
from wavetrace.recognition import find_weapon_entry, open_session, plan_input, train_presence, train_weapon
from wavetrace.frontend import iter_windows
from wavetrace.application.calibrate import calibrate_source
from wavetrace.application.collect import collect_source
from wavetrace.adapters.source import build_csi_source, parse_time_spans
from wavetrace.domain.contracts import (DEFAULT_HOP_FRAMES, DEFAULT_TARGET_SAMPLE_RATE_HZ,
                                        DEFAULT_WINDOW_FRAMES)

_OCC_GRID = 16
# labels change slowly next to the 1.28 s CSI window, so 5 fps is enough and costs
# a third of the CPU that 15 fps does
_CAMERA_FPS = 5.0


def _occupancy_from_variance(image: np.ndarray, grid_size: int = _OCC_GRID) -> np.ndarray:
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


class ArgsMock:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


from wavetrace.localize import Localizer


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
        self.localizer = None

    def log(self, message: str):
        time_of_day = time.strftime('%H:%M:%S')
        asyncio.run_coroutine_threadsafe(self.logs_queue.put(f"[{time_of_day}] {message}"), self.loop)

    def _emit_inference(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.inference_queue.put(json.dumps(payload)), self.loop)

    def _build_source(self, req):
        self.localizer = Localizer(req.antennas, range_enabled=True) if req.antennas >= 2 else None

        if not req.synthetic:
            port = getattr(req, "udp_port", 9876)
            self.log(f"[HW] UDP listener on :{port} — nodes push CSI here (PC_IP set in firmware)")
            self.log(f"[HW] Camera: {req.cam_url}")
            from wavetrace.adapters.source import UdpSource, UdpSourceOptions
            return UdpSource(UdpSourceOptions(port=port, timeout_seconds=60.0))

        self.log("[SIM] Synthetic source (no hardware connected).")
        args = ArgsMock(
            synthetic=True, recording=None, antennas=req.antennas,
            subcarriers=req.subcarriers, fs=req.fs, duration=req.duration,
            presence=req.col_spans if req.action == "collect" else "0:5,10:15,20:25,30:35,40:45,50:55",
            weapon="2:5,12:15,22:25,32:35,42:45,52:55", weapon_depth=0.5,
            seed=getattr(req, 'seed', 0),
        )
        return build_csi_source(args)

    def run_inference_request(self, req):
        source = self._build_source(req)
        self.log(f"Loading Calibration: {req.calibration}")
        self.log(f"Loading Model: {req.model}")
        self.run_inference(source, req.calibration, req.model, req.mode,
                             vote=req.vote, use_gain_lock=req.gainLock,
                             frame_average=req.frame_average, use_baseline=req.use_baseline,
                             port=getattr(req, "udp_port", 9876))

    def run_calibration_request(self, req):
        self.is_running = True
        source = self._build_source(req)
        if not self.is_running: return
        self.log(f"Starting Calibration -> {req.cal_out}")
        path, _ = calibrate_source(source, req.cal_out, baseline_packets=req.baseline_packets, use_gain_lock=req.gainLock)
        self.log(f"Calibration complete: {path}")
        self._emit_inference({"event": "pipeline_done"})
        self.is_running = False

    def run_collection_request(self, req):
        self.is_running = True
        source = self._build_source(req)
        if not self.is_running: return
        self.log(f"Collecting {req.col_stage} dataset (window={req.col_window}, hop={req.col_hop})")
        spans = parse_time_spans(req.col_spans)
        path, dataset = collect_source(source, req.calibration, "output/dataset_ui", spans,
                                       stage=req.col_stage, window=req.col_window, hop=req.col_hop,
                                       subtract_ic_baseline=getattr(req, "subtract_ic_baseline", False))
        self.log(f"Dataset saved ({dataset.y.size} samples) -> {path}")
        self._emit_inference({"event": "pipeline_done"})
        self.is_running = False

    def _emit_training(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.training_queue.put(json.dumps(payload)), self.loop)

    def run_training_request(self, req):
        self.is_running = True
        self.log(f"Training {req.col_stage}/{req.train_backend}...")

        dataset_path = getattr(req, "train_data", "output/dataset_ui")
        import os, glob
        candidate_dirs = sorted(glob.glob(os.path.join(dataset_path, "*")))
        dataset_dirs = [path for path in candidate_dirs
                        if os.path.isdir(path) and os.path.exists(os.path.join(path, "X_features.npy"))]
        if not dataset_dirs:
            dataset_dirs = [dataset_path]

        if not any(os.path.exists(path) for path in dataset_dirs):
            self.log(f"No dataset at {dataset_path}; run 'collect' first.")
            self.is_running = False
            return

        dataset = None
        try:
            from wavetrace.groundtruth import load_dataset
            from wavetrace.diagnostics import dataset_report
            dataset = load_dataset(dataset_dirs[0])
            summary = dataset_report(dataset)
            self._emit_training({"type": "train_init", **summary})
        except Exception as e:
            self.log(f"Dataset meta load failed: {e}")

        def report(epoch: int, metrics: dict) -> None:
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

        try:
            if req.col_stage == "presence":
                from wavetrace.adapters.recognition import is_sklearn_backend
                backend_is_sklearn = is_sklearn_backend(req.train_backend)
                backend = req.train_backend if backend_is_sklearn else "mlp"
                if not backend_is_sklearn:
                    self.log(f"Presence only supports mlp/svm backends; ignoring '{req.train_backend}', using mlp.")
                config = None
                if dataset is not None:
                    from wavetrace.config import ModelConfig
                    meta = dataset.meta
                    config = ModelConfig(stage="presence", k=int(meta["K"]), backend=backend,
                                         window=int(meta["window"]), hop=int(meta["hop"]),
                                         frame_average=int(meta.get("frame_average", 1)),
                                         subtract_baseline=bool(meta.get("subtract_baseline", False)))
                _, metrics = train_presence(dataset_dirs, out_dir=req.train_out, config=config)
                self._emit_training({"type": "done", "metrics": metrics})
            elif req.train_backend == "heatmap":
                metrics = self._train_heatmap(dataset_path, req, report)
                self._emit_training({"type": "done", "metrics": metrics})
            elif getattr(req, "per_link", False):
                trained_link_count = 0
                for node_dir in sorted(glob.glob(os.path.join(dataset_path, "node*"))):
                    node_id_digits = os.path.basename(node_dir)[4:]
                    if not node_id_digits.isdigit():
                        continue
                    for link_dir in sorted(glob.glob(os.path.join(node_dir, "link*"))):
                        link_tag = os.path.basename(link_dir)[4:]
                        link_dataset_dirs = [
                            path for path in sorted(glob.glob(os.path.join(link_dir, "*")))
                            if os.path.isdir(path) and os.path.exists(os.path.join(path, "X_features.npy"))]
                        if not link_dataset_dirs:
                            link_dataset_dirs = (
                                [link_dir] if os.path.exists(os.path.join(link_dir, "X_features.npy"))
                                else [])
                        if not link_dataset_dirs:
                            continue
                        link_model_dir = os.path.join(req.train_out, f"node{node_id_digits}",
                                                      f"link{link_tag}")
                        try:
                            from wavetrace.groundtruth import load_dataset
                            from wavetrace.config import ModelConfig
                            first_link_dataset = load_dataset(link_dataset_dirs[0])
                            subcarrier_count = int(first_link_dataset.meta["K"])
                            link_config = ModelConfig(stage="weapon", k=subcarrier_count, backend="ic27")
                            _, link_metrics = train_weapon(link_dataset_dirs, out_dir=link_model_dir,
                                                           config=link_config, feature_mode="ic27")
                            self._emit_training({"type": "done", "metrics": link_metrics})
                            trained_link_count += 1
                            self.log(f"[WPN] link {link_tag}->node{node_id_digits} -> {link_model_dir}")
                        except Exception as link_error:
                            self.log(f"[WPN] WARN link {link_tag}->node{node_id_digits}: {link_error}")
                self.log(f"[WPN] Per-link training done: {trained_link_count} heads.")
            else:
                from wavetrace.config import ModelConfig
                from wavetrace.groundtruth import load_dataset
                subcarrier_count = int(load_dataset(dataset_dirs[0]).meta["K"])
                config = ModelConfig(stage="weapon", k=subcarrier_count, backend=req.train_backend)
                from wavetrace.adapters.recognition import get_backend_class
                feature_mode = get_backend_class(req.train_backend).feature_mode
                # only the cnn backend calls report; ic27 and variance ignore it
                _, metrics = train_weapon(dataset_dirs, out_dir=req.train_out, config=config,
                                          feature_mode=feature_mode, report=report)
                self._emit_training({"type": "done", "metrics": metrics})
            self.log(f"Training complete -> {req.train_out}")
        except KeyboardInterrupt:
            self.log("Training stopped by UI.")
        except Exception as e:
            self.log(f"ERROR Training failed: {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self.is_running = False

    def _train_heatmap(self, dataset_path: str, req, report) -> dict:
        from wavetrace.groundtruth import load_dataset
        from wavetrace.recognition.heatmap import HeatmapHead
        from wavetrace.config import ModelConfig
        dataset = load_dataset(dataset_path)
        masks = [getattr(label, "mask", None) for label in dataset.labels]
        masks = [mask for mask in masks if mask is not None]
        if not masks:
            raise ValueError(
                "dataset has no Label.mask — collect with SegmentationLabeler / YoloSegLabeler")
        grid = int(getattr(dataset.labels[0], "mask_grid", None) or 16)
        Y = np.asarray(masks, dtype=np.float32)
        config = ModelConfig(stage="weapon", k=int(dataset.meta["K"]))
        head = HeatmapHead(config, grid=grid).fit(dataset.X_image[:len(masks)], Y, report=report)
        import os; os.makedirs(req.train_out, exist_ok=True)
        head.save(os.path.join(req.train_out, "heatmap.joblib"))
        return {"grid": grid, "n": int(Y.shape[0])}

    def _emit_telemetry(self, payload: dict) -> None:
        if self.telemetry_queue is not None:
            asyncio.run_coroutine_threadsafe(
                self.telemetry_queue.put(json.dumps(payload)), self.loop)

    def run_inference(self, source, calibration_dir: str, model_path: str, mode: str,
                        vote: bool = False, use_gain_lock: bool = True,
                        frame_average: int = 1, use_baseline: bool = False, port: int = 9876):
        self.is_running = True
        import os
        from wavetrace.diagnostics import NodeHealthMeter, cluster_sync
        from wavetrace.output.guard import AlertGuard, WeaponAlertEvent
        from wavetrace.recognition.link import LinkVoter, accuracy_weights
        
        is_mesh = os.path.isdir(model_path) and any(os.path.isdir(os.path.join(model_path, dir_name)) for dir_name in os.listdir(model_path) if dir_name.startswith("node"))

        # count reuses the presence head
        head_mode = "presence" if mode == "count" else mode

        if is_mesh:
            self.log(f"Mesh setup detected. Loading per-node models from {model_path}...")
            import glob, json
            nodes = {}
            for node_model_dir in sorted(glob.glob(os.path.join(model_path, "node*"))):
                base = os.path.basename(node_model_dir)
                if not base[len("node"):].isdigit(): continue
                node_id = int(base[len("node"):])
                node_calibration_dir = os.path.join(calibration_dir, base)
                node_model_path = os.path.join(node_model_dir, "model.joblib")
                if not (os.path.isdir(node_calibration_dir) and os.path.exists(node_model_path)): continue
                node_result, node_gain_lock = load_calibration(node_calibration_dir)
                node_session = open_session(head_mode, node_model_path)
                should_apply_node_lock, node_intercarrier, node_pick = plan_input(
                    head_mode, node_session.head)
                classes = [int(class_id) for class_id in node_session.head.classes_]
                # serve with the same IC baseline used in training, or σ²[p] does not match
                node_intercarrier_baseline = (
                    node_result.baseline_mag
                    if getattr(node_session.head.config, "subtract_ic_baseline", False) else None)
                nodes[node_id] = dict(
                    result=node_result,
                    lock=node_gain_lock if (should_apply_node_lock and use_gain_lock) else None,
                    intercarrier=node_intercarrier, pick=node_pick, session=node_session,
                    cfg=node_session.head.config, classes=classes, ic_baseline=node_intercarrier_baseline
                )
                try:
                    with open(os.path.join(node_model_dir, "metrics.json")) as metrics_file:
                        accuracy = float(json.load(metrics_file).get("logo", {}).get("session", {})
                                         .get("accuracy", 1.0))
                except: accuracy = 1.0
                nodes[node_id]["acc"] = accuracy
            
            if not nodes:
                self.log(f"ERROR: No valid mesh models found in {model_path}")
                self.is_running = False
                return

            if mode == "count":
                global_classes = sorted(set().union(*[set(entry["classes"]) for entry in nodes.values()]))
                for entry in nodes.values():
                    entry["col_map"] = [global_classes.index(class_id) for class_id in entry["classes"]]
                    entry["weight"] = max(entry["acc"] - (1.0/len(global_classes)), 0) / max(1.0 - (1.0/len(global_classes)), 1e-9)
            else:
                global_classes = list(next(iter(nodes.values()))["session"].head.classes_)
                weights = accuracy_weights({node_id: entry["acc"] for node_id, entry in nodes.items()})
                for node_id, entry in nodes.items(): entry["weight"] = weights.get(node_id, 1.0)

            config = next(iter(nodes.values()))["cfg"]
            positive_class_index = global_classes.index(1) if 1 in global_classes else -1
            antenna_weights = None
        else:
            self.log("Single-node setup detected.")
            result, gain_lock = load_calibration(calibration_dir)
            session = open_session(head_mode, model_path)
            should_apply_lock, intercarrier, pick = plan_input(head_mode, session.head)
            if not use_gain_lock: gain_lock = None
            config = session.head.config
            # mirror training's IC background subtraction at serve time
            intercarrier_baseline = (result.baseline_mag
                                     if getattr(config, "subtract_ic_baseline", False) else None)
            image_baseline = (build_image_baseline(result, locked=(should_apply_lock and gain_lock is not None))
                              if use_baseline else None)
            global_classes = session.head.classes_
            positive_class_index = list(global_classes).index(1) if 1 in global_classes else -1
            try:
                from wavetrace.recognition.explain import cnn_channel_weights
                channel_weights = cnn_channel_weights(session.head)
                antenna_weights = channel_weights.tolist() if channel_weights is not None else None
            except: antenna_weights = None
            nodes = {0: dict(result=result, lock=gain_lock, intercarrier=intercarrier, pick=pick, session=session, cfg=config)}

        model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        heatmap_path = os.path.join(model_dir, "heatmap.joblib")
        heatmap_head = None
        if os.path.exists(heatmap_path):
            try:
                from wavetrace.recognition.heatmap import HeatmapHead
                heatmap_head = HeatmapHead.load(heatmap_path)
                self.log(f"[HM] Heatmap head loaded ({heatmap_head.grid}×{heatmap_head.grid})")
            except Exception as heatmap_error:
                self.log(f"[HM] WARNING: heatmap load failed ({heatmap_error}); using fallback")

        weapon_entries = None
        if is_mesh and mode == "weapon":
            node_dirs = [os.path.join(model_path, dir_name) for dir_name in os.listdir(model_path)
                          if dir_name.startswith("node")
                          and os.path.isdir(os.path.join(model_path, dir_name))]
            has_links = any(
                any(dir_name.startswith("link") for dir_name in os.listdir(node_dir))
                for node_dir in node_dirs if os.path.isdir(node_dir)
            )
            if has_links:
                try:
                    from wavetrace.recognition import load_weapon_links
                    weapon_entries = load_weapon_links(calibration_dir, model_path)
                    for (tag, node_id), link_entry in weapon_entries.items():
                        if node_id in nodes:
                            nodes[node_id]["weight"] = max(nodes[node_id].get("weight", 0.0),
                                                           link_entry.get("weight", 1.0))
                    self.log(f"[WEAPON] {len(weapon_entries)} per-link entries loaded")
                except Exception as link_load_error:
                    self.log(f"[WEAPON] per-link load failed ({link_load_error}), using per-node")

        def _find_entry(key, _weapon_entries=weapon_entries, _nodes=nodes):
            if _weapon_entries is not None:
                return find_weapon_entry(_weapon_entries, key.tx_mac_suffix, key.rx_node_id)
            return _nodes.get(key.rx_node_id)

        health_meter = NodeHealthMeter()
        alert_guard = AlertGuard()
        voter_trace: deque[float] = deque(maxlen=60)
        is_alert_active = False
        drift_ratio = 0.0
        last_window_timestamp_s = 0.0
        last_telemetry_time_s = [0.0]

        import collections
        from wavetrace.adapters.source import parse_batch_by_link
        buffers = collections.defaultdict(lambda: collections.deque(maxlen=300))  # ~3 s at 100 Hz
        last_seen_time_s = {}
        link_ids = {}
        next_fuse_time_s = time.time() + 1.5

        self.log("Stream started.")
        try:
            if is_mesh:
                from wavetrace.recognition import dwell_proba, link_health
                # parse_batch_by_link needs the raw datagram, so read the socket directly
                # instead of going through snooper.frames()
                import socket
                udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp_socket.settimeout(0.5)
                udp_socket.bind(("", port))

                while self.is_running:
                    now = time.time()
                    try:
                        payload, _ = udp_socket.recvfrom(65535)
                        for key, frames in parse_batch_by_link(payload).items():
                            entry = _find_entry(key)
                            if entry is not None:
                                buffers[key].extend(frames)
                                last_seen_time_s[key] = now
                                link_ids.setdefault(key, len(link_ids))
                                for frame in frames: health_meter.observe(frame)
                    except socket.timeout: pass

                    if now < next_fuse_time_s: continue
                    next_fuse_time_s = now + 1.5

                    for link_buffer in buffers.values():
                        if link_buffer:
                            cutoff = link_buffer[-1].timestamp - 3.0
                            while link_buffer and link_buffer[0].timestamp < cutoff: link_buffer.popleft()

                    link_weights = {}
                    for link_key, link_id in link_ids.items():
                        entry = _find_entry(link_key)
                        link_weights[link_id] = entry["weight"] if entry else 1.0
                    voter = LinkVoter(link_weights if any(weight > 0 for weight in link_weights.values())
                                      else None)

                    display_image = None
                    display_features = None
                    display_intercarrier_features = None
                    node_power = {node_id: 0.0 for node_id in nodes}
                    link_stats = []  # per tx→rx: delivered Hz and missing-frame fraction

                    for key in sorted(buffers):
                        if now - last_seen_time_s.get(key, 0) > 3.0 or len(buffers[key]) < 2: continue
                        entry = _find_entry(key)
                        if entry is None: continue

                        grids = [np.abs(frame.grid).mean() for frame in buffers[key]]
                        node_power[key.rx_node_id] = float(np.mean(grids))
                        delivered_hz, missing_fraction = link_health(list(buffers[key]))
                        link_stats.append({"tx": key.tx_mac_suffix, "rx": key.rx_node_id,
                                           "hz": round(delivered_hz, 1), "miss": round(missing_fraction, 3)})

                        link_proba, image, features, intercarrier_features, _window_count = dwell_proba(
                            list(buffers[key]), entry["session"].head.contract.target_sample_rate_hz, entry)
                        if link_proba is None: continue
                        display_image, display_features, display_intercarrier_features = (
                            image, features, intercarrier_features)

                        if mode == "count":
                            global_proba = np.zeros(len(global_classes), dtype=np.float64)
                            for j, global_column in enumerate(entry["col_map"]):
                                global_proba[global_column] = link_proba[j]
                            sorted_proba = np.sort(link_proba)[::-1]
                            quality = (float(sorted_proba[0] - sorted_proba[1]) if link_proba.size > 1
                                       else float(sorted_proba[0]))
                            voter.add(link_ids[key], global_proba, quality=quality)
                        else:
                            quality = (abs(link_proba[positive_class_index] - 0.5) * 2.0
                                       if positive_class_index >= 0 else 0.0)
                            voter.add(link_ids[key], link_proba, quality=quality)

                    try:
                        voted_class, blended = voter.finalize()
                    except ValueError: continue

                    proba = np.asarray(blended, dtype=np.float64)
                    i = int(np.argmax(proba))
                    predicted_class = int(global_classes[i])
                    confidence = float(proba[i])

                    inference_payload = {"t": float(now), "class": predicted_class, "conf": confidence,
                                         "mode": mode}

                    if mode != "count":
                        alert_event = alert_guard.update(now, predicted_class)
                        if alert_event:
                            is_alert_active = isinstance(alert_event, WeaponAlertEvent)
                            asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps({**inference_payload, **alert_event.to_dict()})), self.loop)
                        if positive_class_index >= 0: voter_trace.append(float(proba[positive_class_index]))

                    asyncio.run_coroutine_threadsafe(
                        self.inference_queue.put(json.dumps(inference_payload)), self.loop)

                    if display_image is not None:
                        occupancy_grid = _predict_occupancy_grid(heatmap_head, display_image)
                        node_power_items = sorted(node_power.items())
                        stream_payload = {
                            "t": float(now), "image": display_image.tolist(),
                            "features": display_features.tolist(),
                            "ic": display_intercarrier_features.tolist(), "antennas": [power for _, power in node_power_items], "node_ids": [node_id for node_id, _ in node_power_items],
                            "heatmap_grid": occupancy_grid.tolist(), "grid_size": _OCC_GRID,
                        }
                        asyncio.run_coroutine_threadsafe(self.stream_queue.put(json.dumps(stream_payload)), self.loop)

                        if now - last_telemetry_time_s[0] >= 1.0:
                            last_telemetry_time_s[0] = now
                            contribution = {_class_name(mode, class_id): round(float(proba[j]), 3) for j, class_id in enumerate(global_classes)}
                            self._emit_telemetry({
                                "nodes": health_meter.snapshot(),
                                "sync": cluster_sync(health_meter),
                                "heatmap": occupancy_grid.tolist(),
                                "grid": _OCC_GRID,
                                "antenna_weights": antenna_weights,
                                "alert_active": is_alert_active,
                                "drift_ratio": drift_ratio,
                                "voter_trace": list(voter_trace),
                                "contribution": contribution,
                                "links": link_stats,
                            })

                udp_socket.close()

            else:
                snooper = FrameSnooper(source, health_meter=health_meter)
                for window_timestamp_s, features, image, intercarrier_features in iter_windows(
                    snooper.frames(), result.subcarriers, gain_lock if should_apply_lock else None,
                    window=config.window, hop=config.hop, intercarrier=True,
                    image_subcarriers=result.image_subcarriers,
                    frame_average=frame_average, image_baseline=image_baseline,
                    ic_baseline=intercarrier_baseline
                ):
                    if not self.is_running: break

                    if last_window_timestamp_s > 0:
                        window_gap_s = window_timestamp_s - last_window_timestamp_s
                        if window_gap_s > 0: time.sleep(window_gap_s)
                    last_window_timestamp_s = window_timestamp_s

                    proba = session.predict_window_proba(pick(features, image, intercarrier_features))
                    i = int(np.argmax(proba))
                    predicted_class = int(global_classes[i])
                    confidence = float(proba[i])
                    inference_payload = {"t": float(window_timestamp_s), "class": predicted_class,
                                         "conf": confidence, "mode": mode}

                    if mode != "count":
                        alert_event = alert_guard.update(window_timestamp_s, predicted_class)
                        if alert_event:
                            is_alert_active = isinstance(alert_event, WeaponAlertEvent)
                            asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps({**inference_payload, **alert_event.to_dict()})), self.loop)
                        if positive_class_index >= 0: voter_trace.append(float(proba[positive_class_index]))

                    spatial_payload = None
                    if self.localizer and snooper.latest_frame is not None:
                        location = self.localizer.locate(snooper.latest_frame, timestamp=window_timestamp_s)
                        spatial_payload = {
                            "x": float(location.x_m), "y": float(location.y_m),
                            "conf": float(location.confidence),
                            "heatmap": location.heatmap.flatten().tolist()
                        }
                        inference_payload["pos"] = [float(location.x_m), 0.0, float(location.y_m)]

                    asyncio.run_coroutine_threadsafe(
                        self.inference_queue.put(json.dumps(inference_payload)), self.loop)

                    node_power_items = sorted(snooper.node_power.items())
                    occupancy_grid = _predict_occupancy_grid(heatmap_head, image)
                    stream_payload = {
                        "t": float(window_timestamp_s), "image": image.tolist(),
                        "features": features.tolist(),
                        "ic": intercarrier_features.tolist(), "antennas": [power for _, power in node_power_items], "node_ids": [int(node_id) for node_id, _ in node_power_items],
                        "heatmap_grid": occupancy_grid.tolist(), "grid_size": _OCC_GRID,
                    }
                    if spatial_payload: stream_payload["spatial"] = spatial_payload

                    asyncio.run_coroutine_threadsafe(self.stream_queue.put(json.dumps(stream_payload)), self.loop)

                    if window_timestamp_s - last_telemetry_time_s[0] >= 1.0:
                        last_telemetry_time_s[0] = window_timestamp_s
                        contribution = {_class_name(mode, class_id): round(float(proba[j]), 3) for j, class_id in enumerate(global_classes)}
                        self._emit_telemetry({
                            "nodes": health_meter.snapshot(),
                            "sync": cluster_sync(health_meter),
                            "heatmap": occupancy_grid.tolist(),
                            "grid": _OCC_GRID,
                            "antenna_weights": antenna_weights,
                            "alert_active": is_alert_active,
                            "drift_ratio": drift_ratio,
                            "voter_trace": list(voter_trace),
                            "contribution": contribution,
                        })

            self.log("Stream ended.")
        except Exception as e:
            self.log(f"ERROR Runner: {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False

    def run_camera_collection_request(self, req):
        """The webcam labeler runs on a worker thread while this thread ingests mesh CSI."""
        self.is_running = True
        import os, glob, socket, threading, collections

        try:
            from wavetrace.groundtruth.camera_labeler import (YoloLabelerOptions, YoloSegLabeler,
                                                              presence_label_fn, weapon_label_fn)
            from wavetrace.groundtruth.webcam import (WebcamCapture, WebcamOptions, record_labels_online,
                                                       COCO_WEAPON_CLASSES)
            from wavetrace.groundtruth.dataset_builder import build_stacked_dataset, save_dataset
            from wavetrace.adapters.source import (parse_batch_by_link, bind_udp_socket, save_recording,
                                                   RecordingSource)
            from wavetrace.domain.dsp.resample import resample_uniform
            from wavetrace.calibration import load_calibration
        except ImportError as import_error:
            self.log(f"ERROR: missing dependency: {import_error}")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return

        camera_index = int(getattr(req, "cam_index", 0))
        duration = float(getattr(req, "duration", 30.0))
        is_per_link = bool(getattr(req, "per_link", False))
        root = getattr(req, "train_data", "data/2g4_ht40")

        calibrations = {}
        for path in sorted(glob.glob(f"{req.calibration}/node*")):
            base = os.path.basename(path)
            if base[4:].isdigit():
                calibrations[int(base[4:])] = load_calibration(path)
        if not calibrations:
            # older flat calibration dirs have no node*/ level; they count as node 0
            flat_meta_path = os.path.join(req.calibration, "meta.json")
            if os.path.exists(flat_meta_path):
                calibrations[0] = load_calibration(req.calibration)
        if not calibrations:
            self.log(f"ERROR: no calibration found at '{req.calibration}' — run Calib first")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return
        calibrated_node_ids = sorted(calibrations)
        self.log(f"[CAM] Nodes: {calibrated_node_ids}. Loading YOLO-seg model...")

        label_function = weapon_label_fn if req.col_stage == "weapon" else presence_label_fn
        yolo_weights_path = getattr(req, "yolo_weights", None)
        try:
            labeler = YoloSegLabeler(
                options=YoloLabelerOptions(weights_path=yolo_weights_path,
                                           weapon_classes=COCO_WEAPON_CLASSES),
                conf=0.35, label_fn=label_function)
        except Exception as yolo_error:
            self.log(f"ERROR: YOLO init failed: {yolo_error}")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return

        self.log(f"[CAM] Capturing {duration:g}s  stage={req.col_stage}  cam={camera_index}  "
                 f"port={req.udp_port}")

        frames_by_node = collections.defaultdict(list)
        frames_by_link = collections.defaultdict(list)
        camera_result: dict = {}
        camera_stop_event = threading.Event()

        def _run_camera_labeler():
            try:
                with WebcamCapture(WebcamOptions(index=camera_index)) as webcam:
                    labeled_counter = {"n": 0}
                    def _on_label(label):
                        labeled_counter["n"] += 1
                        if labeled_counter["n"] % 30 == 0:
                            self.log(f"[CAM] {labeled_counter['n']} frames labeled (class={label.class_id})")
                    camera_result["labels"] = record_labels_online(
                        webcam.read, labeler, duration,
                        fps=_CAMERA_FPS, stop=camera_stop_event, on_label=_on_label,
                    )
            except Exception as camera_error:
                camera_result["error"] = str(camera_error)

        camera_thread = threading.Thread(target=_run_camera_labeler, daemon=True)
        camera_thread.start()

        port = int(getattr(req, "udp_port", 9876))
        try:
            udp_socket = bind_udp_socket(port, timeout_s=1.0)
            capture_end_time_s = time.monotonic() + duration
            while time.monotonic() < capture_end_time_s and self.is_running:
                try:
                    payload, _ = udp_socket.recvfrom(65535)
                except socket.timeout:
                    continue
                for (tx, rx), frames in parse_batch_by_link(payload).items():
                    if rx in calibrated_node_ids or (not calibrated_node_ids and rx == 0):
                        frames_by_node[rx].extend(frames)
                        frames_by_link[(tx, rx)].extend(frames)
        finally:
            udp_socket.close()
            camera_stop_event.set()  # stop the camera thread even when the duration has not elapsed

        camera_thread.join(timeout=max(5.0, duration * 0.1))
        if camera_thread.is_alive():
            self.log("[CAM] Camera worker still running after stop — terminating")
        if "error" in camera_result:
            self.log(f"ERROR webcam: {camera_result['error']}")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return

        labels = camera_result.get("labels", [])
        if not labels:
            self.log("ERROR: no webcam frames — check camera permission or cam_index")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return

        positive_count = sum(label.class_id == 1 for label in labels)
        self.log(f"[CAM] {positive_count}/{len(labels)} frames positive ({req.col_stage})")

        session_id = "cam_s0"
        resampled_by_node = {}
        for node_id, frames in frames_by_node.items():
            resampled_frames = resample_uniform(frames, DEFAULT_TARGET_SAMPLE_RATE_HZ)
            for frame in resampled_frames:
                frame.node_id = node_id
            resampled_by_node[node_id] = resampled_frames

        built_node_ids = []
        for node_id in calibrated_node_ids:
            frames = resampled_by_node.get(node_id, [])
            if len(frames) < DEFAULT_WINDOW_FRAMES:
                self.log(f"[CAM] SKIP node {node_id}: only {len(frames)} frames")
                continue
            recording_dir = f"{root}/cam_rec/{session_id}/node{node_id}"
            dataset_dir = f"{root}/cam_ds/{req.col_stage}/node{node_id}/{session_id}"
            save_recording(frames, recording_dir)
            collect_source(RecordingSource(recording_dir), f"{req.calibration}/node{node_id}",
                            dataset_dir, [], stage=req.col_stage, labeler=labels,
                            session_id=session_id, subject_id="cam",
                            subtract_ic_baseline=(req.col_stage == "weapon"))
            built_node_ids.append(node_id)
            self.log(f"[CAM] node {node_id} -> {dataset_dir}")

        merged = [frame for node_id in calibrated_node_ids for frame in resampled_by_node.get(node_id, [])]
        if merged:
            heatmap_dataset_dir = f"{root}/cam_ds/heatmap/{session_id}"
            heatmap_dataset = build_stacked_dataset(merged, calibrations, labels,
                                                    window=DEFAULT_WINDOW_FRAMES, hop=DEFAULT_HOP_FRAMES,
                                                    session_id=session_id, subject_id="cam")
            save_dataset(heatmap_dataset, heatmap_dataset_dir)
            mask_count = sum(1 for label in heatmap_dataset.labels
                             if getattr(label, "mask", None) is not None)
            self.log(f"[CAM] heatmap stacked -> {heatmap_dataset_dir} "
                     f"({heatmap_dataset.X_image.shape[0]} windows, {mask_count} masks)")

        if req.col_stage == "weapon" and is_per_link:
            for (tx, rx), frames in frames_by_link.items():
                if rx not in calibrated_node_ids:
                    continue
                resampled_frames = resample_uniform(frames, DEFAULT_TARGET_SAMPLE_RATE_HZ)
                if len(resampled_frames) < DEFAULT_WINDOW_FRAMES:
                    continue
                tag = tx.replace(":", "") if tx else "xx"
                link_dataset_dir = f"{root}/cam_ds/weapon/node{rx}/link{tag}/{session_id}"
                link_recording_dir = f"{root}/cam_rec/{session_id}/link{tag}_node{rx}"
                save_recording(resampled_frames, link_recording_dir)
                collect_source(RecordingSource(link_recording_dir), f"{req.calibration}/node{rx}",
                                link_dataset_dir, [], stage="weapon", labeler=labels,
                                session_id=session_id, subject_id="cam",
                                subtract_ic_baseline=True)
                self.log(f"[CAM] per-link weapon {tx}->{rx} -> {link_dataset_dir}")

        self.log(f"[CAM] Done. per-node nodes: {built_node_ids}")
        self._emit_inference({"event": "pipeline_done"})
        self.is_running = False

    def stop(self):
        self.is_running = False
        self.log("System halt requested.")

