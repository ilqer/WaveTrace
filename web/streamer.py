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


def _heatmap_grid(head, image: np.ndarray) -> np.ndarray:
    """Use trained HeatmapHead if loaded, else spectral fallback."""
    if head is None:
        return _occupancy_fallback(image)
    try:
        x = image[np.newaxis]   # (1, K, W)
        return head.predict_heatmap(x)[0].flatten().astype(np.float32)
    except Exception:
        return _occupancy_fallback(image)


def _class_label(mode: str, c: int) -> str:
    """Human-readable class name for the per-class decision readout."""
    c = int(c)
    if mode == "presence":
        return {0: "empty", 1: "present"}.get(c, str(c))
    if mode == "weapon":
        return {0: "no weapon", 1: "weapon"}.get(c, str(c))
    return str(c)  # count / other: numeric class id


class ArgsMock:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


from wavetrace.Localize import Localizer


class FrameSnooper:
    def __init__(self, source, health_meter=None):
        self._source = source
        self.latest_grid = None
        self.latest_frame = None
        # node_id -> latest mean |CSI| over subcarriers. Each UDP/mesh frame is single-antenna
        # (CsiFrame(1, S)) tagged with a node_id, so power is per RX board, not per antenna.
        self.node_power: dict[int, float] = {}
        self._meter = health_meter

    def frames(self):
        for fr in self._source.frames():
            self.latest_grid = np.asarray(fr.grid)
            self.latest_frame = fr
            self.node_power[getattr(fr, "node_id", 0)] = float(np.abs(self.latest_grid).mean())
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

    def _emit_inference(self, obj: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.inference_queue.put(json.dumps(obj)), self.loop)

    def _get_source(self, req):
        self.localizer = Localizer(req.antennas, range_enabled=True) if req.antennas >= 2 else None

        if not req.synthetic:
            port = getattr(req, "udp_port", 9876)
            self.log(f"[HW] UDP listener on :{port} — nodes push CSI here (PC_IP set in firmware)")
            self.log(f"[HW] Camera: {req.cam_url}")
            from wavetrace.Source import UdpSource
            return UdpSource(port=port, timeout_s=60.0)

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
                             frame_average=req.frame_average, use_baseline=req.use_baseline,
                             port=getattr(req, "udp_port", 9876))

    def start_calibration_managed(self, req):
        self.is_running = True
        source = self._get_source(req)
        if not self.is_running: return
        self.log(f"Starting Calibration -> {req.cal_out}")
        path, _ = calibrate_source(source, req.cal_out, baseline_packets=req.baseline_packets, use_gain_lock=req.gain_lock)
        self.log(f"Calibration complete: {path}")
        self._emit_inference({"event": "pipeline_done"})
        self.is_running = False

    def start_collection_managed(self, req):
        self.is_running = True
        source = self._get_source(req)
        if not self.is_running: return
        self.log(f"Collecting {req.col_stage} dataset (window={req.col_window}, hop={req.col_hop})")
        from wavetrace.Cli import _parse_spans
        spans = _parse_spans(req.col_spans)
        path, ds = collect_source(source, req.calibration, "output/dataset_ui", spans,
                                  stage=req.col_stage, window=req.col_window, hop=req.col_hop,
                                  subtract_ic_baseline=getattr(req, "subtract_ic_baseline", False))
        self.log(f"Dataset saved ({ds.y.size} samples) -> {path}")
        self._emit_inference({"event": "pipeline_done"})
        self.is_running = False

    def _emit_train(self, obj: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.training_queue.put(json.dumps(obj)), self.loop)

    def start_training_managed(self, req):
        self.is_running = True
        self.log(f"Training {req.col_stage}/{req.train_backend}...")

        dataset_path = getattr(req, "train_data", "output/dataset_ui")
        import os, glob as _glob
        # Support cumulative pool: if dataset_path contains saved dataset subdirs, use all of them
        _sub = sorted(_glob.glob(os.path.join(dataset_path, "*")))
        ds_dirs = [d for d in _sub if os.path.isdir(d) and os.path.exists(os.path.join(d, "X_features.npy"))]
        if not ds_dirs:
            ds_dirs = [dataset_path]  # treat as a single dataset dir

        if not any(os.path.exists(d) for d in ds_dirs):
            self.log(f"No dataset at {dataset_path}; run 'collect' first.")
            self.is_running = False
            return

        try:
            from wavetrace.groundtruth import load_dataset
            from wavetrace.diagnostics import dataset_report
            ds = load_dataset(ds_dirs[0])
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
                _, m = train_presence(ds_dirs, out_dir=req.train_out)
                self._emit_train({"type": "done", "metrics": m})
            elif req.train_backend == "heatmap":
                m = self._train_heatmap(dataset_path, req, report)
                self._emit_train({"type": "done", "metrics": m})
            elif getattr(req, "per_link", False):
                # Per-link weapon: one head per node*/link*/ dataset subdir
                n_ok = 0
                for _nd in sorted(_glob.glob(os.path.join(dataset_path, "node*"))):
                    _nid_s = os.path.basename(_nd)[4:]
                    if not _nid_s.isdigit():
                        continue
                    for _ld in sorted(_glob.glob(os.path.join(_nd, "link*"))):
                        _tag = os.path.basename(_ld)[4:]
                        _subs = [d for d in sorted(_glob.glob(os.path.join(_ld, "*")))
                                 if os.path.isdir(d) and os.path.exists(os.path.join(d, "X_features.npy"))]
                        if not _subs:
                            _subs = [_ld] if os.path.exists(os.path.join(_ld, "X_features.npy")) else []
                        if not _subs:
                            continue
                        _link_out = os.path.join(req.train_out, f"node{_nid_s}", f"link{_tag}")
                        try:
                            from wavetrace.groundtruth import load_dataset
                            from wavetrace.Config import ModelConfig
                            _ds0 = load_dataset(_subs[0])
                            _k = int(_ds0.meta["K"])
                            _cfg = ModelConfig(stage="weapon", k=_k, backend="ic27")
                            _, _m = train_weapon(_subs, out_dir=_link_out, config=_cfg,
                                                 feature_mode="ic27")
                            self._emit_train({"type": "done", "metrics": _m})
                            n_ok += 1
                            self.log(f"[WPN] link {_tag}->node{_nid_s} -> {_link_out}")
                        except Exception as _le:
                            self.log(f"[WPN] WARN link {_tag}->node{_nid_s}: {_le}")
                self.log(f"[WPN] Per-link training done: {n_ok} heads.")
            else:
                from wavetrace.Config import ModelConfig
                from wavetrace.groundtruth import load_dataset
                k = int(load_dataset(ds_dirs[0]).meta["K"])
                cfg = ModelConfig(stage="weapon", k=k, backend=req.train_backend)
                fm = "cnn" if req.train_backend == "cnn" else "ic27"
                _, m = train_weapon(ds_dirs, out_dir=req.train_out, config=cfg,
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
                        frame_average: int = 1, use_baseline: bool = False, port: int = 9876):
        self.is_running = True
        import os
        from wavetrace.diagnostics import NodeHealthMeter, cluster_sync
        from wavetrace.output.Guard import AlertGuard, DriftMonitor
        from wavetrace.recognition.Link import LinkVoter, accuracy_weights
        
        is_mesh = os.path.isdir(model_path) and any(os.path.isdir(os.path.join(model_path, d)) for d in os.listdir(model_path) if d.startswith("node"))
        
        # Determine internal mode for session loading (count uses presence head)
        load_mode = "presence" if mode == "count" else mode
        
        if is_mesh:
            self.log(f"Mesh setup detected! Loading per-node models from {model_path}...")
            # Inline loading logic for mesh nodes
            import glob, json
            nodes = {}
            for mdir in sorted(glob.glob(os.path.join(model_path, "node*"))):
                base = os.path.basename(mdir)
                if not base[len("node"):].isdigit(): continue
                nid = int(base[len("node"):])
                cdir = os.path.join(calib_dir, base)
                mpath = os.path.join(mdir, "model.joblib")
                if not (os.path.isdir(cdir) and os.path.exists(mpath)): continue
                res, glock = load_calibration(cdir)
                sess = mode_session(load_mode, mpath)
                alock, ic, pck = _serving_plan(load_mode, sess.head)
                classes = [int(c) for c in sess.head.classes_]
                # Item 10/CAUSE 2B: if this weapon head was trained with IC background subtraction,
                # serve with the SAME per-node baseline or σ²[p] silently mismatches training.
                ic_base = (res.baseline_mag
                           if getattr(sess.head.config, "subtract_ic_baseline", False) else None)
                nodes[nid] = dict(
                    result=res, lock=glock if (alock and use_gain_lock) else None,
                    intercarrier=ic, pick=pck, session=sess, cfg=sess.head.config,
                    classes=classes, ic_baseline=ic_base
                )
                try:
                    with open(os.path.join(mdir, "metrics.json")) as f:
                        acc = float(json.load(f).get("logo", {}).get("session", {}).get("accuracy", 1.0))
                except: acc = 1.0
                nodes[nid]["acc"] = acc
            
            if not nodes:
                self.log(f"ERROR: No valid mesh models found in {model_path}")
                self.is_running = False
                return

            if mode == "count":
                global_classes = sorted(set().union(*[set(m["classes"]) for m in nodes.values()]))
                for m in nodes.values():
                    m["col_map"] = [global_classes.index(c) for c in m["classes"]]
                    m["weight"] = max(m["acc"] - (1.0/len(global_classes)), 0) / max(1.0 - (1.0/len(global_classes)), 1e-9)
            else:
                global_classes = list(next(iter(nodes.values()))["session"].head.classes_)
                weights = accuracy_weights({nid: m["acc"] for nid, m in nodes.items()})
                for nid, m in nodes.items(): m["weight"] = weights.get(nid, 1.0)
                
            cfg = next(iter(nodes.values()))["cfg"]
            _pos_idx = global_classes.index(1) if 1 in global_classes else -1
            _ant_weights = None
        else:
            # Single node fallback
            self.log("Single-node setup detected.")
            result, gain_lock = load_calibration(calib_dir)
            session = mode_session(load_mode, model_path)
            apply_lock, intercarrier, pick = _serving_plan(load_mode, session.head)
            if not use_gain_lock: gain_lock = None
            cfg = session.head.config
            # Item 10/CAUSE 2B: mirror training's IC background subtraction at serve time.
            _ic_base = result.baseline_mag if getattr(cfg, "subtract_ic_baseline", False) else None
            _img_base = get_image_baseline(result, locked=(apply_lock and gain_lock is not None)) if use_baseline else None
            global_classes = session.head.classes_
            _pos_idx = list(global_classes).index(1) if 1 in global_classes else -1
            try:
                from wavetrace.recognition.Explain import cnn_channel_weights
                _aw = cnn_channel_weights(session.head)
                _ant_weights = _aw.tolist() if _aw is not None else None
            except: _ant_weights = None
            nodes = {0: dict(result=result, lock=gain_lock, intercarrier=intercarrier, pick=pick, session=session, cfg=cfg)}

        # ---- Gap 2: trained heatmap head (replaces _occupancy_fallback when present) ----
        _model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        _hm_path = os.path.join(_model_dir, "heatmap.joblib")
        heatmap_head = None
        if os.path.exists(_hm_path):
            try:
                from wavetrace.recognition.Heatmap import HeatmapHead
                heatmap_head = HeatmapHead.load(_hm_path)
                self.log(f"[HM] Heatmap head loaded ({heatmap_head.grid}×{heatmap_head.grid})")
            except Exception as _hm_e:
                self.log(f"[HM] WARNING: heatmap load failed ({_hm_e}); using fallback")

        # ---- Gap 3: per-link weapon entries (auto-detected from node*/link*/ dirs) ------
        weapon_entries = None
        if is_mesh and mode == "weapon":
            _node_dirs = [os.path.join(model_path, b) for b in os.listdir(model_path)
                          if b.startswith("node") and os.path.isdir(os.path.join(model_path, b))]
            _has_links = any(
                any(d.startswith("link") for d in os.listdir(nd))
                for nd in _node_dirs if os.path.isdir(nd)
            )
            if _has_links:
                try:
                    from run_weapon import load_weapon_links
                    weapon_entries = load_weapon_links(calib_dir, model_path)
                    for (tag, nid), e in weapon_entries.items():
                        if nid in nodes:
                            nodes[nid]["weight"] = max(nodes[nid].get("weight", 0.0),
                                                       e.get("weight", 1.0))
                    self.log(f"[WEAPON] {len(weapon_entries)} per-link entries loaded")
                except Exception as _we:
                    self.log(f"[WEAPON] per-link load failed ({_we}), using per-node")

        def _lookup_entry(key, _we=weapon_entries, _n=nodes):
            if _we is not None:
                tx = key[0].replace(":", "") if key[0] else None
                return _we.get((tx, key[1])) or _we.get((None, key[1]))
            return _n.get(key[1])

        health_meter = NodeHealthMeter()
        alert_guard = AlertGuard()
        _voter_trace: deque[float] = deque(maxlen=60)
        _alert_active = False
        _drift_ratio = 0.0
        last_t = 0.0
        _last_tel_t = [0.0]

        # Mesh specific state
        import collections
        from wavetrace.Source import parse_batch_links
        buffers = collections.defaultdict(lambda: collections.deque(maxlen=300))  # ~3s at 100Hz (#17)
        last_seen = {}
        link_ids = {}
        next_fuse = time.time() + 1.5

        self.log("Stream started.")
        try:
            if is_mesh:
                # Use raw UDP ingestion for parse_batch_links instead of snooper.frames()
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(0.5)
                sock.bind(("", port))
                
                while self.is_running:
                    now = time.time()
                    try:
                        payload, _ = sock.recvfrom(65535)
                        for key, frames in parse_batch_links(payload).items():
                            m = _lookup_entry(key)
                            if m is not None:
                                buffers[key].extend(frames)
                                last_seen[key] = now
                                link_ids.setdefault(key, len(link_ids))
                                for fr in frames: health_meter.observe(fr)
                    except socket.timeout: pass

                    if now < next_fuse: continue
                    next_fuse = now + 1.5

                    # Trim buffers to 3 seconds
                    for buf in buffers.values():
                        if buf:
                            cutoff = buf[-1].timestamp - 3.0
                            while buf and buf[0].timestamp < cutoff: buf.popleft()

                    _sw = {}
                    for _k, _lid in link_ids.items():
                        _e = _lookup_entry(_k)
                        _sw[_lid] = _e["weight"] if _e else 1.0
                    voter = LinkVoter(_sw if any(w > 0 for w in _sw.values()) else None)
                    
                    rep_image = None
                    rep_features = None
                    rep_ic = None
                    node_power = {nid: 0.0 for nid in nodes}
                    
                    from wavetrace.Source import resample_uniform
                    for key in sorted(buffers):
                        if now - last_seen.get(key, 0) > 3.0 or len(buffers[key]) < 2: continue
                        m = _lookup_entry(key)
                        if m is None: continue
                        
                        # Accumulate node power for UI
                        grids = [np.abs(f.grid).mean() for f in buffers[key]]
                        node_power[key[1]] = float(np.mean(grids))

                        res = resample_uniform(list(buffers[key]), 100.0)
                        if len(res) < m["cfg"].window: continue
                        
                        win_probs = []
                        for t, features, image, ic in iter_windows(
                            res, m["result"].subcarriers, m["lock"],
                            window=m["cfg"].window, hop=m["cfg"].hop, intercarrier=m["intercarrier"],
                            image_subcarriers=m["result"].image_subcarriers,
                            ic_baseline=m.get("ic_baseline"),
                        ):
                            win_probs.append(m["session"].predict_proba_window(m["pick"](features, image, ic)))
                            rep_image, rep_features, rep_ic = image, features, ic

                        if not win_probs: continue
                        # Item 12/CAUSE 5C: temporal (soft) vote across the whole dwell, not just the
                        # last window — matches run_weapon._dwell_proba (Zhou 51%->93%).
                        last_probs = np.mean(win_probs, axis=0)
                        
                        if mode == "count":
                            g = np.zeros(len(global_classes), dtype=np.float64)
                            for j, col in enumerate(m["col_map"]): g[col] = last_probs[j]
                            top = np.sort(last_probs)[::-1]
                            qual = float(top[0] - top[1]) if last_probs.size > 1 else float(top[0])
                            voter.add(link_ids[key], g, quality=qual)
                        else:
                            qual = abs(last_probs[_pos_idx] - 0.5) * 2.0 if _pos_idx >= 0 else 0.0
                            voter.add(link_ids[key], last_probs, quality=qual)

                    try:
                        vcls, blended = voter.finalize()
                    except ValueError: continue

                    probs = np.asarray(blended, dtype=np.float64)
                    i = int(np.argmax(probs))
                    cls = int(global_classes[i])
                    conf = float(probs[i])
                    
                    r = {"t": float(now), "class": cls, "conf": conf, "mode": mode}
                    
                    if mode != "count":
                        alert_ev = alert_guard.update(now, cls)
                        if alert_ev:
                            _alert_active = alert_ev["event"] == "weapon_alert"
                            asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps({**r, **alert_ev})), self.loop)
                        if _pos_idx >= 0: _voter_trace.append(float(probs[_pos_idx]))

                    asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps(r)), self.loop)

                    # Stream payload (use the last valid link's image for visualization)
                    if rep_image is not None:
                        occ_grid = _heatmap_grid(heatmap_head, rep_image)
                        np_items = sorted(node_power.items())
                        stream_payload = {
                            "t": float(now), "image": rep_image.tolist(), "features": rep_features.tolist(),
                            "ic": rep_ic.tolist(), "antennas": [p for _, p in np_items], "node_ids": [n for n, _ in np_items],
                            "heatmap_grid": occ_grid.tolist(), "grid_size": _OCC_GRID,
                        }
                        asyncio.run_coroutine_threadsafe(self.stream_queue.put(json.dumps(stream_payload)), self.loop)
                    
                        if now - _last_tel_t[0] >= 1.0:
                            _last_tel_t[0] = now
                            contrib = {_class_label(mode, c): round(float(probs[j]), 3) for j, c in enumerate(global_classes)}
                            self._emit_telemetry({
                                "nodes": health_meter.snapshot(),
                                "sync": cluster_sync(health_meter),
                                "heatmap": occ_grid.tolist(),
                                "grid": _OCC_GRID,
                                "antenna_weights": _ant_weights,
                                "alert_active": _alert_active,
                                "drift_ratio": _drift_ratio,
                                "voter_trace": list(_voter_trace),
                                "contribution": contrib,
                            })

                sock.close()

            else:
                snooper = FrameSnooper(source, health_meter=health_meter)
                for t, features, image, ic in iter_windows(
                    snooper.frames(), result.subcarriers, gain_lock if apply_lock else None,
                    window=cfg.window, hop=cfg.hop, intercarrier=True,
                    image_subcarriers=result.image_subcarriers,
                    frame_average=frame_average, image_baseline=_img_base, ic_baseline=_ic_base
                ):
                    if not self.is_running: break

                    if last_t > 0:
                        dt = t - last_t
                        if dt > 0: time.sleep(dt)
                    last_t = t

                    probs = session.predict_proba_window(pick(features, image, ic))
                    i = int(np.argmax(probs))
                    cls = int(global_classes[i])
                    conf = float(probs[i])
                    r = {"t": float(t), "class": cls, "conf": conf, "mode": mode}

                    if mode != "count":
                        alert_ev = alert_guard.update(t, cls)
                        if alert_ev:
                            _alert_active = alert_ev["event"] == "weapon_alert"
                            asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps({**r, **alert_ev})), self.loop)
                        if _pos_idx >= 0: _voter_trace.append(float(probs[_pos_idx]))

                    spatial_data = None
                    if self.localizer and snooper.latest_frame is not None:
                        loc = self.localizer.locate(snooper.latest_frame, timestamp=t)
                        spatial_data = {
                            "x": float(loc.x_m), "y": float(loc.y_m), "conf": float(loc.confidence),
                            "heatmap": loc.heatmap.flatten().tolist()
                        }
                        r["pos"] = [float(loc.x_m), 0.0, float(loc.y_m)]

                    asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps(r)), self.loop)

                    node_items = sorted(snooper.node_power.items())
                    occ_grid = _heatmap_grid(heatmap_head, image)
                    stream_payload = {
                        "t": float(t), "image": image.tolist(), "features": features.tolist(),
                        "ic": ic.tolist(), "antennas": [p for _, p in node_items], "node_ids": [int(nid) for nid, _ in node_items],
                        "heatmap_grid": occ_grid.tolist(), "grid_size": _OCC_GRID,
                    }
                    if spatial_data: stream_payload["spatial"] = spatial_data

                    asyncio.run_coroutine_threadsafe(self.stream_queue.put(json.dumps(stream_payload)), self.loop)

                    if t - _last_tel_t[0] >= 1.0:
                        _last_tel_t[0] = t
                        contrib = {_class_label(mode, c): round(float(probs[j]), 3) for j, c in enumerate(global_classes)}
                        self._emit_telemetry({
                            "nodes": health_meter.snapshot(),
                            "sync": cluster_sync(health_meter),
                            "heatmap": occ_grid.tolist(),
                            "grid": _OCC_GRID,
                            "antenna_weights": _ant_weights,
                            "alert_active": _alert_active,
                            "drift_ratio": _drift_ratio,
                            "voter_trace": list(_voter_trace),
                            "contribution": contrib,
                        })

            self.log("Stream ended.")
        except Exception as e:
            self.log(f"ERROR Runner: {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False

    def start_camera_collect_managed(self, req):
        """Camera-supervised collection: concurrent webcam YOLO + mesh CSI -> datasets.
        Builds per-node presence/weapon datasets, stacked heatmap dataset, and optionally
        per-link weapon datasets (when col_stage=weapon and per_link=True)."""
        self.is_running = True
        import os, glob as _g, socket as _sock, threading, time as _t, collections as _col

        try:
            from wavetrace.groundtruth.CameraLabeler import (YoloSegLabeler,
                                                              presence_label_fn, weapon_label_fn)
            from wavetrace.groundtruth.Webcam import (WebcamCapture, record_labels_online,
                                                       COCO_WEAPON_CLASSES)
            from wavetrace.groundtruth.DatasetBuilder import build_dataset_stacked, save_dataset
            from wavetrace.Source import (parse_batch_links, resample_uniform, bind_udp,
                                          save_recording, RecordingSource)
            from wavetrace.Calibration import load_calibration
            from wavetrace.Cli import collect_source as _collect_source
        except ImportError as _ie:
            self.log(f"ERROR: missing dependency: {_ie}")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return

        WINDOW, TARGET_FS = 128, 100.0
        cam_index = int(getattr(req, "cam_index", 0))
        duration = float(getattr(req, "duration", 30.0))
        per_link = bool(getattr(req, "per_link", False))
        root = getattr(req, "train_data", "data/2g4_ht40")

        # ── Load calibrations ──────────────────────────────────────────────
        # Try per-node layout first (node0/, node1/, …), fall back to flat dir.
        calibs = {}
        for d in sorted(_g.glob(f"{req.calibration}/node*")):
            base = os.path.basename(d)
            if base[4:].isdigit():
                calibs[int(base[4:])] = load_calibration(d)
        if not calibs:
            # Flat calibration dir (single-node or unified calib) — treat as node 0
            flat_meta = os.path.join(req.calibration, "meta.json")
            if os.path.exists(flat_meta):
                calibs[0] = load_calibration(req.calibration)
        if not calibs:
            self.log(f"ERROR: no calibration found at '{req.calibration}' — run Calib first")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return
        cal_nodes = sorted(calibs)
        self.log(f"[CAM] Nodes: {cal_nodes}. Loading YOLO-seg model...")

        label_fn = weapon_label_fn if req.col_stage == "weapon" else presence_label_fn
        yolo_weights = getattr(req, "yolo_weights", "yolov8n-seg.pt") or "yolov8n-seg.pt"
        try:
            labeler = YoloSegLabeler(yolo_weights, weapon_classes=COCO_WEAPON_CLASSES,
                                     conf=0.35, label_fn=label_fn)
        except Exception as _ye:
            self.log(f"ERROR: YOLO init failed: {_ye}")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return

        self.log(f"[CAM] Capturing {duration:g}s  stage={req.col_stage}  cam={cam_index}  port={req.udp_port}")
        # Camera runs at 5 fps for YOLO labeling — labels change slowly (person enters/leaves room),
        # CSI windows are 1.28s wide, so one label per 200ms is plenty. Reduces CPU load 3x vs 15fps.
        CAM_FPS = 5.0

        per_node = _col.defaultdict(list)
        per_link_csi = _col.defaultdict(list)
        box: dict = {}
        cam_stop = threading.Event()  # set this to stop the camera worker early

        def _cam_worker():
            try:
                with WebcamCapture(index=cam_index) as cap:
                    _cnt = {"n": 0}
                    def _on_label(lb):
                        _cnt["n"] += 1
                        if _cnt["n"] % 30 == 0:
                            self.log(f"[CAM] {_cnt['n']} frames labeled (class={lb.class_id})")
                    box["labels"] = record_labels_online(
                        cap.read, labeler, duration,
                        fps=CAM_FPS, stop=cam_stop, on_label=_on_label,
                    )
            except Exception as _ce:
                box["error"] = str(_ce)

        th = threading.Thread(target=_cam_worker, daemon=True)
        th.start()

        port = int(getattr(req, "udp_port", 9876))
        try:
            s = bind_udp(port, timeout=1.0)
            t_end = _t.monotonic() + duration
            while _t.monotonic() < t_end and self.is_running:
                try:
                    payload, _ = s.recvfrom(65535)
                except _sock.timeout:
                    continue
                for (tx, rx), frames in parse_batch_links(payload).items():
                    if rx in cal_nodes or (not cal_nodes and rx == 0):
                        per_node[rx].extend(frames)
                        per_link_csi[(tx, rx)].extend(frames)
        finally:
            s.close()
            cam_stop.set()  # signal the camera thread to stop even if duration not elapsed

        th.join(timeout=max(5.0, duration * 0.1))
        if th.is_alive():
            self.log("[CAM] Camera worker still running after stop — terminating")
        if "error" in box:
            self.log(f"ERROR webcam: {box['error']}")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return

        labels = box.get("labels", [])
        if not labels:
            self.log("ERROR: no webcam frames — check camera permission or cam_index")
            self._emit_inference({"event": "pipeline_done"})
            self.is_running = False
            return

        n_pos = sum(lb.class_id == 1 for lb in labels)
        self.log(f"[CAM] {n_pos}/{len(labels)} frames positive ({req.col_stage})")

        sess = "cam_s0"
        res = {}
        for nid, frs in per_node.items():
            rf = resample_uniform(frs, TARGET_FS)
            for f in rf:
                f.node_id = nid
            res[nid] = rf

        # 1) Per-node presence/weapon datasets
        pres_built = []
        for nid in cal_nodes:
            frs = res.get(nid, [])
            if len(frs) < WINDOW:
                self.log(f"[CAM] SKIP node {nid}: only {len(frs)} frames")
                continue
            rec = f"{root}/cam_rec/{sess}/node{nid}"
            ds = f"{root}/cam_ds/{req.col_stage}/node{nid}/{sess}"
            save_recording(frs, rec)
            _collect_source(RecordingSource(rec), f"{req.calibration}/node{nid}", ds, [],
                            stage=req.col_stage, labeler=labels,
                            session_id=sess, subject_id="cam",
                            subtract_ic_baseline=(req.col_stage == "weapon"))
            pres_built.append(nid)
            self.log(f"[CAM] node {nid} -> {ds}")

        # 2) Stacked heatmap dataset (all nodes as channels + occupancy mask)
        merged = [f for nid in cal_nodes for f in res.get(nid, [])]
        if merged:
            hm_dir = f"{root}/cam_ds/heatmap/{sess}"
            hm_ds = build_dataset_stacked(merged, calibs, labels, window=WINDOW, hop=32,
                                          session_id=sess, subject_id="cam")
            save_dataset(hm_ds, hm_dir)
            n_mask = sum(1 for lb in hm_ds.labels if getattr(lb, "mask", None) is not None)
            self.log(f"[CAM] heatmap stacked -> {hm_dir} "
                     f"({hm_ds.X_image.shape[0]} windows, {n_mask} masks)")

        # 3) Per-link weapon datasets (requires per_link=True and stage=weapon)
        if req.col_stage == "weapon" and per_link:
            for (tx, rx), frs in per_link_csi.items():
                if rx not in cal_nodes:
                    continue
                rf = resample_uniform(frs, TARGET_FS)
                if len(rf) < WINDOW:
                    continue
                tag = tx.replace(":", "") if tx else "xx"
                ld = f"{root}/cam_ds/weapon/node{rx}/link{tag}/{sess}"
                lr = f"{root}/cam_rec/{sess}/link{tag}_node{rx}"
                save_recording(rf, lr)
                _collect_source(RecordingSource(lr), f"{req.calibration}/node{rx}", ld, [],
                                stage="weapon", labeler=labels,
                                session_id=sess, subject_id="cam",
                                subtract_ic_baseline=True)
                self.log(f"[CAM] per-link weapon {tx}->{rx} -> {ld}")

        self.log(f"[CAM] Done. per-node nodes: {pres_built}")
        self._emit_inference({"event": "pipeline_done"})
        self.is_running = False

    def stop(self):
        self.is_running = False
        self.log("System halt requested.")

