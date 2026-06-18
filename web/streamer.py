import asyncio
import time
import traceback
import numpy as np
import json
from collections import deque

from wavetrace.Calibration import load_calibration, image_baseline as get_image_baseline
from wavetrace.recognition import mode_session, SegmentVoter, train_presence, train_weapon
from wavetrace.Frontend import iter_windows
from wavetrace.Cli import _serving_plan, _source_from_args, calibrate_source, collect_source
from wavetrace import RecognitionResult

_OCC_GRID = 16


def _occupancy_fallback(image: np.ndarray, G: int = _OCC_GRID) -> np.ndarray:
    """Per-subcarrier variance of (K, window) image tiled/downsampled to G×G flat array [0,1].
    Tiles when K < G² (avoids zero-padding which makes most bars black)."""
    var = image.var(axis=1).astype(np.float32)  # (K,)
    g2 = G * G
    if var.size < g2:
        reps = (g2 + var.size - 1) // var.size
        var = np.tile(var, reps)[:g2]
    elif var.size > g2:
        step = var.size // g2
        var = var[:step * g2].reshape(g2, step).mean(axis=1)
    lo, hi = var.min(), var.max()
    if hi > lo:
        var = (var - lo) / (hi - lo)
    return var.astype(np.float32)


class ArgsMock:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


from wavetrace.Localize import Localizer


class FrameSnooper:
    def __init__(self, source, health_meter=None):
        self._source = source
        self.latest_grid = None
        self.latest_frame = None
        self._meter = health_meter

    def frames(self):
        for fr in self._source.frames():
            self.latest_grid = np.asarray(fr.grid)
            self.latest_frame = fr
            if self._meter is not None:
                self._meter.observe(fr)
            yield fr


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

    def log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        asyncio.run_coroutine_threadsafe(self.logs_queue.put(f"[{ts}] {msg}"), self.loop)

    def _get_source(self, req):
        self.localizer = Localizer(req.antennas, range_enabled=True) if req.antennas >= 2 else None

        if not req.synthetic:
            self.log(f"[HW] UDP listener on :5566 — nodes: {req.nodes}")
            self.log(f"[HW] Camera: {req.cam_url}")
            from wavetrace.Source import UdpSource
            return UdpSource(port=5566, timeout_s=60.0)

        # Synthetic path — retained for CLI/test use only.
        self.log("[SIM] Synthetic source (no hardware connected).")
        args = ArgsMock(
            synthetic=True, recording=None, antennas=req.antennas,
            subcarriers=req.subcarriers, fs=req.fs, duration=req.duration,
            presence=req.col_spans if req.action == "collect" else "0:5,10:15,20:25,30:35,40:45,50:55",
            weapon="2:5,12:15,22:25,32:35,42:45,52:55", weapon_depth=0.5,
            seed=getattr(req, 'seed', 0),
        )
        return _source_from_args(args)

    def start_inference_managed(self, req):
        source = self._get_source(req)
        self.log(f"Loading Calibration: {req.calibration}")
        self.log(f"Loading Model: {req.model}")
        self.start_inference(source, req.calibration, req.model, req.mode,
                             vote=req.vote, use_gain_lock=req.gain_lock,
                             frame_average=req.frame_average, use_baseline=req.use_baseline)

    def start_calibration_managed(self, req):
        self.is_running = True
        source = self._get_source(req)
        if not self.is_running: return
        self.log(f"Starting Calibration -> {req.cal_out}")
        path, _ = calibrate_source(source, req.cal_out, baseline_packets=req.baseline_packets, use_gain_lock=req.gain_lock)
        self.log(f"Calibration complete: {path}")
        self.is_running = False

    def start_collection_managed(self, req):
        self.is_running = True
        source = self._get_source(req)
        if not self.is_running: return
        self.log(f"Collecting {req.col_stage} dataset (window={req.col_window}, hop={req.col_hop})")
        from wavetrace.Cli import _parse_spans
        spans = _parse_spans(req.col_spans)
        path, ds = collect_source(source, req.calibration, "output/dataset_ui", spans,
                                  stage=req.col_stage, window=req.col_window, hop=req.col_hop)
        self.log(f"Dataset saved ({ds.y.size} samples) -> {path}")
        self.is_running = False

    def _emit_train(self, obj: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.training_queue.put(json.dumps(obj)), self.loop)

    def start_training_managed(self, req):
        self.is_running = True
        self.log(f"Training {req.col_stage}/{req.train_backend}...")

        dataset_path = "output/dataset_ui"
        import os
        if not os.path.exists(dataset_path):
            self.log(f"No dataset at {dataset_path}; run 'collect' first.")
            self.is_running = False
            return

        try:
            from wavetrace.groundtruth import load_dataset
            from wavetrace.diagnostics import dataset_report
            ds = load_dataset(dataset_path)
            rep = dataset_report(ds)
            self._emit_train({"type": "train_init", **rep})
        except Exception as e:
            self.log(f"Dataset meta load failed: {e}")

        def report(epoch: int, m: dict) -> None:
            if not self.is_running:
                raise KeyboardInterrupt("training stopped by UI")
            self._emit_train({
                "type": "epoch",
                "epoch": epoch,
                "loss": float(m.get("loss", 0.0)),
                "val_loss": float(m.get("val_loss", m.get("loss", 0.0))),
                "accuracy": float(m.get("acc", 0.0)),
                "val_accuracy": float(m.get("val_acc", 0.0)),
            })
            self.log(f"Epoch {epoch}: loss={m.get('loss', 0):.4f}")

        try:
            if req.col_stage == "presence":
                _, m = train_presence([dataset_path], out_dir=req.train_out)
                self._emit_train({"type": "done", "metrics": m})
            elif req.train_backend == "heatmap":
                m = self._train_heatmap(dataset_path, req, report)
                self._emit_train({"type": "done", "metrics": m})
            else:
                from wavetrace.Config import ModelConfig
                from wavetrace.groundtruth import load_dataset
                k = int(load_dataset(dataset_path).meta["K"])
                cfg = ModelConfig(stage="weapon", k=k, backend=req.train_backend)
                fm = "cnn" if req.train_backend == "cnn" else "ic27"
                _, m = train_weapon([dataset_path], out_dir=req.train_out, config=cfg,
                                    feature_mode=fm)
                self._emit_train({"type": "done", "metrics": m})
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
        """Train the camera-supervised G×G heatmap head from datasets with Label.mask."""
        from wavetrace.groundtruth import load_dataset
        from wavetrace.recognition.Heatmap import HeatmapHead
        from wavetrace.Config import ModelConfig
        ds = load_dataset(dataset_path)
        masks = [getattr(lb, "mask", None) for lb in ds.labels]
        masks = [m for m in masks if m is not None]
        if not masks:
            raise ValueError(
                "dataset has no Label.mask — collect with SegmentationLabeler / YoloSegLabeler")
        grid = int(getattr(ds.labels[0], "mask_grid", None) or 16)
        Y = np.asarray(masks, dtype=np.float32)
        cfg = ModelConfig(stage="weapon", k=int(ds.meta["K"]))
        head = HeatmapHead(cfg, grid=grid).fit(ds.X_image[:len(masks)], Y, report=report)
        import os; os.makedirs(req.train_out, exist_ok=True)
        head.save(os.path.join(req.train_out, "heatmap.joblib"))
        return {"grid": grid, "n": int(Y.shape[0])}

    def _emit_telemetry(self, payload: dict) -> None:
        if self.telemetry_queue is not None:
            asyncio.run_coroutine_threadsafe(
                self.telemetry_queue.put(json.dumps(payload)), self.loop)

    def start_inference(self, source, calib_dir: str, model_path: str, mode: str,
                        vote: bool = False, use_gain_lock: bool = True,
                        frame_average: int = 1, use_baseline: bool = False):
        self.is_running = True
        result, gain_lock = load_calibration(calib_dir)
        session = mode_session(mode, model_path)
        apply_lock, intercarrier, pick = _serving_plan(mode, session.head)
        if not use_gain_lock: gain_lock = None

        image_baseline = get_image_baseline(result, locked=gain_lock is not None) if use_baseline else None
        cfg = session.head.config
        voter = SegmentVoter() if vote else None
        last_t = 0.0

        from wavetrace.diagnostics import NodeHealthMeter, cluster_sync
        from wavetrace.output.Guard import AlertGuard, DriftMonitor

        health_meter = NodeHealthMeter()
        _last_tel_t = [0.0]
        snooper = FrameSnooper(source, health_meter=health_meter)
        alert_guard = AlertGuard()
        _bm = getattr(result, 'mags_mean', None)
        drift_mon = DriftMonitor(_bm) if _bm is not None else None

        # Static per-antenna CNN channel weights (L2 norms of first conv filters) — computed once.
        try:
            from wavetrace.recognition.Explain import cnn_channel_weights
            _aw = cnn_channel_weights(session.head)
            _ant_weights = _aw.tolist() if _aw is not None else None
        except Exception:
            _ant_weights = None

        # Per-window positive-class prob trace for SegmentVoteTrace (last 60 windows, circular).
        _voter_trace: deque[float] = deque(maxlen=60)
        _alert_active = False
        _drift_ratio = 0.0

        # Positive-class index in the head's class list.
        def _pos_idx(classes) -> int:
            cl = list(classes)
            return cl.index(1) if 1 in cl else len(cl) - 1

        self.log("Stream started.")
        try:
            for t, features, image, ic in iter_windows(
                snooper.frames(), result.subcarriers, gain_lock if apply_lock else None,
                window=cfg.window, hop=cfg.hop, intercarrier=True,
                image_subcarriers=result.image_subcarriers,
                frame_average=frame_average, image_baseline=image_baseline
            ):
                if not self.is_running: break

                if last_t > 0:
                    dt = t - last_t
                    if dt > 0: time.sleep(dt)
                last_t = t

                # single forward pass — reuse proba for verdict, voter, and voter trace
                probs = session.predict_proba_window(pick(features, image, ic))
                _classes = session.head.classes_
                i = int(np.argmax(probs))
                cls = int(_classes[i])
                conf = float(probs[i])
                r = {"t": float(t), "class": cls, "conf": conf, "mode": mode}

                # AlertGuard: N consecutive positives → weapon_alert; M negatives → clear
                alert_ev = alert_guard.update(t, cls)
                if alert_ev:
                    _alert_active = alert_ev["event"] == "weapon_alert"
                    asyncio.run_coroutine_threadsafe(
                        self.inference_queue.put(json.dumps({**r, **alert_ev})), self.loop)

                # DriftMonitor: slow EMA vs calibration baseline (fires at most once per 10 min)
                if drift_mon is not None and snooper.latest_grid is not None:
                    raw_mags = np.abs(snooper.latest_grid).mean(axis=0)
                    drift_ev = drift_mon.update(t, raw_mags)
                    if drift_ev:
                        _drift_ratio = float(drift_ev["drift"])
                        self.log(f"[WARN] Drift advisory drift={_drift_ratio:.2f} — recalibrate soon")

                # Voter trace: track positive-class prob history for UI visualization.
                pidx = _pos_idx(_classes)
                _voter_trace.append(float(probs[pidx]))

                if voter is not None:
                    voter.add(probs)

                # Spatial Localization
                spatial_data = None
                if self.localizer and snooper.latest_frame is not None:
                    loc = self.localizer.locate(snooper.latest_frame, timestamp=t)
                    spatial_data = {
                        "x": float(loc.x_m), "y": float(loc.y_m),
                        "conf": float(loc.confidence),
                        "heatmap": loc.heatmap.flatten().tolist()
                    }
                    r["pos"] = [float(loc.x_m), 0.0, float(loc.y_m)]

                asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps(r)), self.loop)

                ant_power = np.abs(snooper.latest_grid).mean(axis=1).tolist() if snooper.latest_grid is not None else None
                occ_grid = _occupancy_fallback(image)
                stream_payload = {
                    "t": float(t), "image": image.tolist(), "features": features.tolist(),
                    "ic": ic.tolist(), "antennas": ant_power,
                    "heatmap_grid": occ_grid.tolist(), "grid_size": _OCC_GRID,
                }
                if spatial_data:
                    stream_payload["spatial"] = spatial_data

                asyncio.run_coroutine_threadsafe(self.stream_queue.put(json.dumps(stream_payload)), self.loop)

                # Emit full telemetry snapshot ~1 Hz
                if t - _last_tel_t[0] >= 1.0:
                    _last_tel_t[0] = t
                    self._emit_telemetry({
                        "nodes": health_meter.snapshot(),
                        "sync": cluster_sync(health_meter),
                        "heatmap": occ_grid.tolist(),
                        "grid": _OCC_GRID,
                        "antenna_weights": _ant_weights,
                        "alert_active": _alert_active,
                        "drift_ratio": _drift_ratio,
                        "voter_trace": list(_voter_trace),
                    })

            if voter is not None and len(voter):
                vcls, vmean = voter.finalize()
                r = {"t": float(t), "class": int(vcls), "conf": float(vmean[vcls]), "mode": mode, "final_vote": True}
                asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps(r)), self.loop)
            self.log("Stream ended.")
        except Exception as e:
            self.log(f"ERROR Runner: {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False
        self.log("System halt requested.")
