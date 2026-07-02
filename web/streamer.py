import asyncio
import time
import traceback
import numpy as np
import json
from collections import deque

from wavetrace.Calibration import loadCalibration, imageBaseline as get_image_baseline
from wavetrace.recognition import modeSession, SegmentVoter, trainPresence, trainWeapon
from wavetrace.Frontend import iterWindows
from wavetrace.Cli import _servingPlan, _sourceFromArgs, calibrateSource, collectSource
from wavetrace import RecognitionResult

_OCC_GRID = 16


def _occupancyFallback(image: np.ndarray, G: int = _OCC_GRID) -> np.ndarray:
    """Returns per-subcarrier variance of image scaled to GxG [0,1]. Tiles if K < G² to avoid zero-padding black bars."""
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


def _heatmapGrid(head, image: np.ndarray) -> np.ndarray:
    """Uses trained HeatmapHead or fallback."""
    if head is None:
        return _occupancyFallback(image)
    try:
        x = image[np.newaxis]   # (1, K, W)
        return head.predictHeatmap(x)[0].flatten().astype(np.float32)
    except Exception:
        return _occupancyFallback(image)


def _classLabel(mode: str, c: int) -> str:
    """Returns string class label for mode."""
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
        # node_id -> latest mean |CSI|; each UDP/mesh frame is single-antenna, so power is per RX board.
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

    def _emitInference(self, obj: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.inference_queue.put(json.dumps(obj)), self.loop)

    def _getSource(self, req):
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
        return _sourceFromArgs(args)

    def startInferenceManaged(self, req):
        source = self._getSource(req)
        self.log(f"Loading Calibration: {req.calibration}")
        self.log(f"Loading Model: {req.model}")
        self.startInference(source, req.calibration, req.model, req.mode,
                             vote=req.vote, use_gain_lock=req.gainLock,
                             frame_average=req.frame_average, use_baseline=req.use_baseline,
                             port=getattr(req, "udp_port", 9876))

    def startCalibrationManaged(self, req):
        self.is_running = True
        source = self._getSource(req)
        if not self.is_running: return
        self.log(f"Starting Calibration -> {req.cal_out}")
        path, _ = calibrateSource(source, req.cal_out, baseline_packets=req.baseline_packets, use_gain_lock=req.gainLock)
        self.log(f"Calibration complete: {path}")
        self._emitInference({"event": "pipeline_done"})
        self.is_running = False

    def startCollectionManaged(self, req):
        self.is_running = True
        source = self._getSource(req)
        if not self.is_running: return
        self.log(f"Collecting {req.col_stage} dataset (window={req.col_window}, hop={req.col_hop})")
        from wavetrace.Cli import _parseSpans
        spans = _parseSpans(req.col_spans)
        path, ds = collectSource(source, req.calibration, "output/dataset_ui", spans,
                                  stage=req.col_stage, window=req.col_window, hop=req.col_hop,
                                  subtract_ic_baseline=getattr(req, "subtract_ic_baseline", False))
        self.log(f"Dataset saved ({ds.y.size} samples) -> {path}")
        self._emitInference({"event": "pipeline_done"})
        self.is_running = False

    def _emitTrain(self, obj: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self.training_queue.put(json.dumps(obj)), self.loop)

    def startTrainingManaged(self, req):
        self.is_running = True
        self.log(f"Training {req.col_stage}/{req.train_backend}...")

        datasetPath = getattr(req, "train_data", "output/dataset_ui")
        import os, glob as _glob
        # Support cumulative pool: if datasetPath contains saved dataset subdirs, use all of them
        _sub = sorted(_glob.glob(os.path.join(datasetPath, "*")))
        dsDirs = [d for d in _sub if os.path.isdir(d) and os.path.exists(os.path.join(d, "X_features.npy"))]
        if not dsDirs:
            dsDirs = [datasetPath]

        if not any(os.path.exists(d) for d in dsDirs):
            self.log(f"No dataset at {datasetPath}; run 'collect' first.")
            self.is_running = False
            return

        ds = None
        try:
            from wavetrace.groundtruth import loadDataset
            from wavetrace.diagnostics import datasetReport
            ds = loadDataset(dsDirs[0])
            rep = datasetReport(ds)
            self._emitTrain({"type": "train_init", **rep})
        except Exception as e:
            self.log(f"Dataset meta load failed: {e}")

        def report(epoch: int, m: dict) -> None:
            if not self.is_running:
                raise KeyboardInterrupt("training stopped by UI")
            self._emitTrain({
                "type": "epoch",
                "epoch": epoch,
                "loss": float(m.get("loss", 0.0)),
                "loss_std": float(m.get("loss_std", 0.0)),  # within-epoch batch spread -> curve band
                "accuracy": float(m.get("acc", m.get("accuracy", 0.0))),
            })
            self.log(f"Epoch {epoch}: loss={m.get('loss', 0):.4f}")

        try:
            if req.col_stage == "presence":
                # PresenceHead is sklearn-only (P6 lock, wavetrace/recognition/Model.py) — cnn/variance
                # are weapon-only backends and don't apply here.
                backend = req.train_backend if req.train_backend in ("mlp", "svm") else "mlp"
                if req.train_backend not in ("mlp", "svm"):
                    self.log(f"Presence only supports mlp/svm backends; ignoring '{req.train_backend}', using mlp.")
                cfg = None
                if ds is not None:
                    from wavetrace.Config import ModelConfig
                    meta = ds.meta
                    cfg = ModelConfig(stage="presence", k=int(meta["K"]), backend=backend,
                                      window=int(meta["window"]), hop=int(meta["hop"]),
                                      frame_average=int(meta.get("frame_average", 1)),
                                      subtract_baseline=bool(meta.get("subtract_baseline", False)))
                _, m = trainPresence(dsDirs, out_dir=req.train_out, config=cfg)
                self._emitTrain({"type": "done", "metrics": m})
            elif req.train_backend == "heatmap":
                m = self._trainHeatmap(datasetPath, req, report)
                self._emitTrain({"type": "done", "metrics": m})
            elif getattr(req, "per_link", False):
                # Per-link weapon: one head per node*/link*/ dataset subdir
                nOk = 0
                for _nd in sorted(_glob.glob(os.path.join(datasetPath, "node*"))):
                    _nidS = os.path.basename(_nd)[4:]
                    if not _nidS.isdigit():
                        continue
                    for _ld in sorted(_glob.glob(os.path.join(_nd, "link*"))):
                        _tag = os.path.basename(_ld)[4:]
                        _subs = [d for d in sorted(_glob.glob(os.path.join(_ld, "*")))
                                 if os.path.isdir(d) and os.path.exists(os.path.join(d, "X_features.npy"))]
                        if not _subs:
                            _subs = [_ld] if os.path.exists(os.path.join(_ld, "X_features.npy")) else []
                        if not _subs:
                            continue
                        _linkOut = os.path.join(req.train_out, f"node{_nidS}", f"link{_tag}")
                        try:
                            from wavetrace.groundtruth import loadDataset
                            from wavetrace.Config import ModelConfig
                            _ds0 = loadDataset(_subs[0])
                            _k = int(_ds0.meta["K"])
                            _cfg = ModelConfig(stage="weapon", k=_k, backend="ic27")
                            _, _m = trainWeapon(_subs, out_dir=_linkOut, config=_cfg,
                                                 feature_mode="ic27")
                            self._emitTrain({"type": "done", "metrics": _m})
                            nOk += 1
                            self.log(f"[WPN] link {_tag}->node{_nidS} -> {_linkOut}")
                        except Exception as _le:
                            self.log(f"[WPN] WARN link {_tag}->node{_nidS}: {_le}")
                self.log(f"[WPN] Per-link training done: {nOk} heads.")
            else:
                from wavetrace.Config import ModelConfig
                from wavetrace.groundtruth import loadDataset
                k = int(loadDataset(dsDirs[0]).meta["K"])
                cfg = ModelConfig(stage="weapon", k=k, backend=req.train_backend)
                fm = "cnn" if req.train_backend == "cnn" else "ic27"
                # report streams per-epoch curves to the dashboard (cnn only; ignored by ic27/variance)
                _, m = trainWeapon(dsDirs, out_dir=req.train_out, config=cfg,
                                    feature_mode=fm, report=report)
                self._emitTrain({"type": "done", "metrics": m})
            self.log(f"Training complete -> {req.train_out}")
        except KeyboardInterrupt:
            self.log("Training stopped by UI.")
        except Exception as e:
            self.log(f"ERROR Training failed: {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self.is_running = False

    def _trainHeatmap(self, dataset_path: str, req, report) -> dict:
        """Train the camera-supervised G×G heatmap head from datasets with Label.mask."""
        from wavetrace.groundtruth import loadDataset
        from wavetrace.recognition.Heatmap import HeatmapHead
        from wavetrace.Config import ModelConfig
        ds = loadDataset(dataset_path)
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

    def _emitTelemetry(self, payload: dict) -> None:
        if self.telemetry_queue is not None:
            asyncio.run_coroutine_threadsafe(
                self.telemetry_queue.put(json.dumps(payload)), self.loop)

    def startInference(self, source, calib_dir: str, model_path: str, mode: str,
                        vote: bool = False, use_gain_lock: bool = True,
                        frame_average: int = 1, use_baseline: bool = False, port: int = 9876):
        self.is_running = True
        import os
        from wavetrace.diagnostics import NodeHealthMeter, clusterSync
        from wavetrace.output.Guard import AlertGuard, DriftMonitor
        from wavetrace.recognition.Link import LinkVoter, accuracyWeights
        
        isMesh = os.path.isdir(model_path) and any(os.path.isdir(os.path.join(model_path, d)) for d in os.listdir(model_path) if d.startswith("node"))

        # Determine internal mode for session loading (count uses presence head)
        loadMode = "presence" if mode == "count" else mode

        if isMesh:
            self.log(f"Mesh setup detected. Loading per-node models from {model_path}...")
            import glob, json
            nodes = {}
            for mdir in sorted(glob.glob(os.path.join(model_path, "node*"))):
                base = os.path.basename(mdir)
                if not base[len("node"):].isdigit(): continue
                nid = int(base[len("node"):])
                cdir = os.path.join(calib_dir, base)
                mpath = os.path.join(mdir, "model.joblib")
                if not (os.path.isdir(cdir) and os.path.exists(mpath)): continue
                res, glock = loadCalibration(cdir)
                sess = modeSession(loadMode, mpath)
                alock, ic, pck = _servingPlan(loadMode, sess.head)
                classes = [int(c) for c in sess.head.classes_]
                # Item 10/CAUSE 2B: must serve with the same IC baseline used in training or σ²[p] mismatches.
                icBase = (res.baseline_mag
                           if getattr(sess.head.config, "subtract_ic_baseline", False) else None)
                nodes[nid] = dict(
                    result=res, lock=glock if (alock and use_gain_lock) else None,
                    intercarrier=ic, pick=pck, session=sess, cfg=sess.head.config,
                    classes=classes, ic_baseline=icBase
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
                globalClasses = sorted(set().union(*[set(m["classes"]) for m in nodes.values()]))
                for m in nodes.values():
                    m["col_map"] = [globalClasses.index(c) for c in m["classes"]]
                    m["weight"] = max(m["acc"] - (1.0/len(globalClasses)), 0) / max(1.0 - (1.0/len(globalClasses)), 1e-9)
            else:
                globalClasses = list(next(iter(nodes.values()))["session"].head.classes_)
                weights = accuracyWeights({nid: m["acc"] for nid, m in nodes.items()})
                for nid, m in nodes.items(): m["weight"] = weights.get(nid, 1.0)

            cfg = next(iter(nodes.values()))["cfg"]
            _posIdx = globalClasses.index(1) if 1 in globalClasses else -1
            _antWeights = None
        else:
            self.log("Single-node setup detected.")
            result, gainLock = loadCalibration(calib_dir)
            session = modeSession(loadMode, model_path)
            apply_lock, intercarrier, pick = _servingPlan(loadMode, session.head)
            if not use_gain_lock: gainLock = None
            cfg = session.head.config
            # Item 10/CAUSE 2B: mirror training's IC background subtraction at serve time.
            _icBase = result.baseline_mag if getattr(cfg, "subtract_ic_baseline", False) else None
            _imgBase = get_image_baseline(result, locked=(apply_lock and gainLock is not None)) if use_baseline else None
            globalClasses = session.head.classes_
            _posIdx = list(globalClasses).index(1) if 1 in globalClasses else -1
            try:
                from wavetrace.recognition.Explain import cnnChannelWeights
                _aw = cnnChannelWeights(session.head)
                _antWeights = _aw.tolist() if _aw is not None else None
            except: _antWeights = None
            nodes = {0: dict(result=result, lock=gainLock, intercarrier=intercarrier, pick=pick, session=session, cfg=cfg)}

        # ---- Gap 2: trained heatmap head (replaces _occupancyFallback when present) ----
        _modelDir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        _hmPath = os.path.join(_modelDir, "heatmap.joblib")
        heatmapHead = None
        if os.path.exists(_hmPath):
            try:
                from wavetrace.recognition.Heatmap import HeatmapHead
                heatmapHead = HeatmapHead.load(_hmPath)
                self.log(f"[HM] Heatmap head loaded ({heatmapHead.grid}×{heatmapHead.grid})")
            except Exception as _hmE:
                self.log(f"[HM] WARNING: heatmap load failed ({_hmE}); using fallback")

        # ---- Gap 3: per-link weapon entries (auto-detected from node*/link*/ dirs) ------
        weaponEntries = None
        if isMesh and mode == "weapon":
            _nodeDirs = [os.path.join(model_path, b) for b in os.listdir(model_path)
                          if b.startswith("node") and os.path.isdir(os.path.join(model_path, b))]
            _hasLinks = any(
                any(d.startswith("link") for d in os.listdir(nd))
                for nd in _nodeDirs if os.path.isdir(nd)
            )
            if _hasLinks:
                try:
                    from scripts.run_weapon import loadWeaponLinks
                    weaponEntries = loadWeaponLinks(calib_dir, model_path)
                    for (tag, nid), e in weaponEntries.items():
                        if nid in nodes:
                            nodes[nid]["weight"] = max(nodes[nid].get("weight", 0.0),
                                                       e.get("weight", 1.0))
                    self.log(f"[WEAPON] {len(weaponEntries)} per-link entries loaded")
                except Exception as _we:
                    self.log(f"[WEAPON] per-link load failed ({_we}), using per-node")

        def _lookupEntry(key, _we=weaponEntries, _n=nodes):
            if _we is not None:
                tx = key[0].replace(":", "") if key[0] else None
                return _we.get((tx, key[1])) or _we.get((None, key[1]))
            return _n.get(key[1])

        healthMeter = NodeHealthMeter()
        alertGuard = AlertGuard()
        _voterTrace: deque[float] = deque(maxlen=60)
        _alertActive = False
        _driftRatio = 0.0
        lastT = 0.0
        _lastTelT = [0.0]

        import collections
        from wavetrace.Source import parseBatchLinks
        buffers = collections.defaultdict(lambda: collections.deque(maxlen=300))  # ~3s at 100Hz (#17)
        lastSeen = {}
        linkIds = {}
        nextFuse = time.time() + 1.5

        self.log("Stream started.")
        try:
            if isMesh:
                # per-link serving math lives once in run_weapon; presence/count/weapon all share it.
                from scripts.run_weapon import dwellProbaDetailed, _linkHealth
                # Use raw UDP ingestion for parseBatchLinks instead of snooper.frames()
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(0.5)
                sock.bind(("", port))

                while self.is_running:
                    now = time.time()
                    try:
                        payload, _ = sock.recvfrom(65535)
                        for key, frames in parseBatchLinks(payload).items():
                            m = _lookupEntry(key)
                            if m is not None:
                                buffers[key].extend(frames)
                                lastSeen[key] = now
                                linkIds.setdefault(key, len(linkIds))
                                for fr in frames: healthMeter.observe(fr)
                    except socket.timeout: pass

                    if now < nextFuse: continue
                    nextFuse = now + 1.5

                    for buf in buffers.values():
                        if buf:
                            cutoff = buf[-1].timestamp - 3.0
                            while buf and buf[0].timestamp < cutoff: buf.popleft()

                    _sw = {}
                    for _k, _lid in linkIds.items():
                        _e = _lookupEntry(_k)
                        _sw[_lid] = _e["weight"] if _e else 1.0
                    voter = LinkVoter(_sw if any(w > 0 for w in _sw.values()) else None)

                    repImage = None
                    repFeatures = None
                    repIc = None
                    nodePower = {nid: 0.0 for nid in nodes}
                    linkStats = []  # per (tx->rx) delivered Hz + missing-frame fraction (C9b) for the UI

                    for key in sorted(buffers):
                        if now - lastSeen.get(key, 0) > 3.0 or len(buffers[key]) < 2: continue
                        m = _lookupEntry(key)
                        if m is None: continue

                        grids = [np.abs(f.grid).mean() for f in buffers[key]]
                        nodePower[key[1]] = float(np.mean(grids))
                        _hz, _miss = _linkHealth(list(buffers[key]))
                        linkStats.append({"tx": key[0], "rx": key[1],
                                           "hz": round(_hz, 1), "miss": round(_miss, 3)})

                        # temporal soft vote over the buffer + last window's image/features/ic for the spectrogram
                        lastProbs, image, features, ic, _nw = dwellProbaDetailed(
                            list(buffers[key]), 100.0, m)
                        if lastProbs is None: continue
                        repImage, repFeatures, repIc = image, features, ic

                        if mode == "count":
                            g = np.zeros(len(globalClasses), dtype=np.float64)
                            for j, col in enumerate(m["col_map"]): g[col] = lastProbs[j]
                            top = np.sort(lastProbs)[::-1]
                            qual = float(top[0] - top[1]) if lastProbs.size > 1 else float(top[0])
                            voter.add(linkIds[key], g, quality=qual)
                        else:
                            qual = abs(lastProbs[_posIdx] - 0.5) * 2.0 if _posIdx >= 0 else 0.0
                            voter.add(linkIds[key], lastProbs, quality=qual)

                    try:
                        vcls, blended = voter.finalize()
                    except ValueError: continue

                    probs = np.asarray(blended, dtype=np.float64)
                    i = int(np.argmax(probs))
                    cls = int(globalClasses[i])
                    conf = float(probs[i])

                    r = {"t": float(now), "class": cls, "conf": conf, "mode": mode}

                    if mode != "count":
                        alertEv = alertGuard.update(now, cls)
                        if alertEv:
                            _alertActive = alertEv["event"] == "weapon_alert"
                            asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps({**r, **alertEv})), self.loop)
                        if _posIdx >= 0: _voterTrace.append(float(probs[_posIdx]))

                    asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps(r)), self.loop)

                    # Stream payload (use the last valid link's image for visualization)
                    if repImage is not None:
                        occGrid = _heatmapGrid(heatmapHead, repImage)
                        npItems = sorted(nodePower.items())
                        streamPayload = {
                            "t": float(now), "image": repImage.tolist(), "features": repFeatures.tolist(),
                            "ic": repIc.tolist(), "antennas": [p for _, p in npItems], "node_ids": [n for n, _ in npItems],
                            "heatmap_grid": occGrid.tolist(), "grid_size": _OCC_GRID,
                        }
                        asyncio.run_coroutine_threadsafe(self.stream_queue.put(json.dumps(streamPayload)), self.loop)

                        if now - _lastTelT[0] >= 1.0:
                            _lastTelT[0] = now
                            contrib = {_classLabel(mode, c): round(float(probs[j]), 3) for j, c in enumerate(globalClasses)}
                            self._emitTelemetry({
                                "nodes": healthMeter.snapshot(),
                                "sync": clusterSync(healthMeter),
                                "heatmap": occGrid.tolist(),
                                "grid": _OCC_GRID,
                                "antenna_weights": _antWeights,
                                "alert_active": _alertActive,
                                "drift_ratio": _driftRatio,
                                "voter_trace": list(_voterTrace),
                                "contribution": contrib,
                                "links": linkStats,
                            })

                sock.close()

            else:
                snooper = FrameSnooper(source, health_meter=healthMeter)
                for t, features, image, ic in iterWindows(
                    snooper.frames(), result.subcarriers, gainLock if apply_lock else None,
                    window=cfg.window, hop=cfg.hop, intercarrier=True,
                    image_subcarriers=result.image_subcarriers,
                    frame_average=frame_average, imageBaseline=_imgBase, ic_baseline=_icBase
                ):
                    if not self.is_running: break

                    if lastT > 0:
                        dt = t - lastT
                        if dt > 0: time.sleep(dt)
                    lastT = t

                    probs = session.predictProbaWindow(pick(features, image, ic))
                    i = int(np.argmax(probs))
                    cls = int(globalClasses[i])
                    conf = float(probs[i])
                    r = {"t": float(t), "class": cls, "conf": conf, "mode": mode}

                    if mode != "count":
                        alertEv = alertGuard.update(t, cls)
                        if alertEv:
                            _alertActive = alertEv["event"] == "weapon_alert"
                            asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps({**r, **alertEv})), self.loop)
                        if _posIdx >= 0: _voterTrace.append(float(probs[_posIdx]))

                    spatialData = None
                    if self.localizer and snooper.latest_frame is not None:
                        loc = self.localizer.locate(snooper.latest_frame, timestamp=t)
                        spatialData = {
                            "x": float(loc.x_m), "y": float(loc.y_m), "conf": float(loc.confidence),
                            "heatmap": loc.heatmap.flatten().tolist()
                        }
                        r["pos"] = [float(loc.x_m), 0.0, float(loc.y_m)]

                    asyncio.run_coroutine_threadsafe(self.inference_queue.put(json.dumps(r)), self.loop)

                    nodeItems = sorted(snooper.node_power.items())
                    occGrid = _heatmapGrid(heatmapHead, image)
                    streamPayload = {
                        "t": float(t), "image": image.tolist(), "features": features.tolist(),
                        "ic": ic.tolist(), "antennas": [p for _, p in nodeItems], "node_ids": [int(nid) for nid, _ in nodeItems],
                        "heatmap_grid": occGrid.tolist(), "grid_size": _OCC_GRID,
                    }
                    if spatialData: streamPayload["spatial"] = spatialData

                    asyncio.run_coroutine_threadsafe(self.stream_queue.put(json.dumps(streamPayload)), self.loop)

                    if t - _lastTelT[0] >= 1.0:
                        _lastTelT[0] = t
                        contrib = {_classLabel(mode, c): round(float(probs[j]), 3) for j, c in enumerate(globalClasses)}
                        self._emitTelemetry({
                            "nodes": healthMeter.snapshot(),
                            "sync": clusterSync(healthMeter),
                            "heatmap": occGrid.tolist(),
                            "grid": _OCC_GRID,
                            "antenna_weights": _antWeights,
                            "alert_active": _alertActive,
                            "drift_ratio": _driftRatio,
                            "voter_trace": list(_voterTrace),
                            "contribution": contrib,
                        })

            self.log("Stream ended.")
        except Exception as e:
            self.log(f"ERROR Runner: {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines():
                self.log(f"  {line}")
        finally:
            self._emitInference({"event": "pipeline_done"})
            self.is_running = False

    def startCameraCollectManaged(self, req):
        """Camera collection: concurrent webcam YOLO and mesh CSI. Builds per-node, stacked heatmap, and optionally per-link weapon datasets."""
        self.is_running = True
        import os, glob as _g, socket as _sock, threading, time as _t, collections as _col

        try:
            from wavetrace.groundtruth.CameraLabeler import (YoloSegLabeler,
                                                              presenceLabelFn, weaponLabelFn)
            from wavetrace.groundtruth.Webcam import (WebcamCapture, recordLabelsOnline,
                                                       COCO_WEAPON_CLASSES)
            from wavetrace.groundtruth.DatasetBuilder import buildDatasetStacked, saveDataset
            from wavetrace.Source import (parseBatchLinks, resampleUniform, bindUdp,
                                          saveRecording, RecordingSource)
            from wavetrace.Calibration import loadCalibration
            from wavetrace.Cli import collectSource as _collect_source
        except ImportError as _ie:
            self.log(f"ERROR: missing dependency: {_ie}")
            self._emitInference({"event": "pipeline_done"})
            self.is_running = False
            return

        WINDOW, TARGET_FS = 128, 100.0
        camIndex = int(getattr(req, "cam_index", 0))
        duration = float(getattr(req, "duration", 30.0))
        perLink = bool(getattr(req, "per_link", False))
        root = getattr(req, "train_data", "data/2g4_ht40")

        # ── Load calibrations ──────────────────────────────────────────────
        # Try per-node layout first (node0/, node1/, …), fall back to flat dir.
        calibs = {}
        for d in sorted(_g.glob(f"{req.calibration}/node*")):
            base = os.path.basename(d)
            if base[4:].isdigit():
                calibs[int(base[4:])] = loadCalibration(d)
        if not calibs:
            # Flat calibration dir (single-node or unified calib) — treat as node 0
            flatMeta = os.path.join(req.calibration, "meta.json")
            if os.path.exists(flatMeta):
                calibs[0] = loadCalibration(req.calibration)
        if not calibs:
            self.log(f"ERROR: no calibration found at '{req.calibration}' — run Calib first")
            self._emitInference({"event": "pipeline_done"})
            self.is_running = False
            return
        calNodes = sorted(calibs)
        self.log(f"[CAM] Nodes: {calNodes}. Loading YOLO-seg model...")

        labelFn = weaponLabelFn if req.col_stage == "weapon" else presenceLabelFn
        yoloWeights = getattr(req, "yolo_weights", "yolov8n-seg.pt") or "yolov8n-seg.pt"
        try:
            labeler = YoloSegLabeler(yoloWeights, weapon_classes=COCO_WEAPON_CLASSES,
                                     conf=0.35, label_fn=labelFn)
        except Exception as _ye:
            self.log(f"ERROR: YOLO init failed: {_ye}")
            self._emitInference({"event": "pipeline_done"})
            self.is_running = False
            return

        self.log(f"[CAM] Capturing {duration:g}s  stage={req.col_stage}  cam={camIndex}  port={req.udp_port}")
        # 5fps is plenty since labels change slowly relative to the 1.28s CSI window; cuts CPU load 3x vs 15fps.
        CAM_FPS = 5.0

        perNode = _col.defaultdict(list)
        perLinkCsi = _col.defaultdict(list)
        box: dict = {}
        camStop = threading.Event()  # set this to stop the camera worker early

        def _camWorker():
            try:
                with WebcamCapture(index=camIndex) as cap:
                    _cnt = {"n": 0}
                    def _onLabel(lb):
                        _cnt["n"] += 1
                        if _cnt["n"] % 30 == 0:
                            self.log(f"[CAM] {_cnt['n']} frames labeled (class={lb.class_id})")
                    box["labels"] = recordLabelsOnline(
                        cap.read, labeler, duration,
                        fps=CAM_FPS, stop=camStop, onLabel=_onLabel,
                    )
            except Exception as _ce:
                box["error"] = str(_ce)

        th = threading.Thread(target=_camWorker, daemon=True)
        th.start()

        port = int(getattr(req, "udp_port", 9876))
        try:
            s = bindUdp(port, timeout=1.0)
            tEnd = _t.monotonic() + duration
            while _t.monotonic() < tEnd and self.is_running:
                try:
                    payload, _ = s.recvfrom(65535)
                except _sock.timeout:
                    continue
                for (tx, rx), frames in parseBatchLinks(payload).items():
                    if rx in calNodes or (not calNodes and rx == 0):
                        perNode[rx].extend(frames)
                        perLinkCsi[(tx, rx)].extend(frames)
        finally:
            s.close()
            camStop.set()  # signal the camera thread to stop even if duration not elapsed

        th.join(timeout=max(5.0, duration * 0.1))
        if th.is_alive():
            self.log("[CAM] Camera worker still running after stop — terminating")
        if "error" in box:
            self.log(f"ERROR webcam: {box['error']}")
            self._emitInference({"event": "pipeline_done"})
            self.is_running = False
            return

        labels = box.get("labels", [])
        if not labels:
            self.log("ERROR: no webcam frames — check camera permission or cam_index")
            self._emitInference({"event": "pipeline_done"})
            self.is_running = False
            return

        nPos = sum(lb.class_id == 1 for lb in labels)
        self.log(f"[CAM] {nPos}/{len(labels)} frames positive ({req.col_stage})")

        sess = "cam_s0"
        res = {}
        for nid, frs in perNode.items():
            rf = resampleUniform(frs, TARGET_FS)
            for f in rf:
                f.node_id = nid
            res[nid] = rf

        # 1) Per-node presence/weapon datasets
        presBuilt = []
        for nid in calNodes:
            frs = res.get(nid, [])
            if len(frs) < WINDOW:
                self.log(f"[CAM] SKIP node {nid}: only {len(frs)} frames")
                continue
            rec = f"{root}/cam_rec/{sess}/node{nid}"
            ds = f"{root}/cam_ds/{req.col_stage}/node{nid}/{sess}"
            saveRecording(frs, rec)
            _collect_source(RecordingSource(rec), f"{req.calibration}/node{nid}", ds, [],
                            stage=req.col_stage, labeler=labels,
                            session_id=sess, subject_id="cam",
                            subtract_ic_baseline=(req.col_stage == "weapon"))
            presBuilt.append(nid)
            self.log(f"[CAM] node {nid} -> {ds}")

        # 2) Stacked heatmap dataset (all nodes as channels + occupancy mask)
        merged = [f for nid in calNodes for f in res.get(nid, [])]
        if merged:
            hmDir = f"{root}/cam_ds/heatmap/{sess}"
            hmDs = buildDatasetStacked(merged, calibs, labels, window=WINDOW, hop=32,
                                          session_id=sess, subject_id="cam")
            saveDataset(hmDs, hmDir)
            nMask = sum(1 for lb in hmDs.labels if getattr(lb, "mask", None) is not None)
            self.log(f"[CAM] heatmap stacked -> {hmDir} "
                     f"({hmDs.X_image.shape[0]} windows, {nMask} masks)")

        # 3) Per-link weapon datasets (requires per_link=True and stage=weapon)
        if req.col_stage == "weapon" and perLink:
            for (tx, rx), frs in perLinkCsi.items():
                if rx not in calNodes:
                    continue
                rf = resampleUniform(frs, TARGET_FS)
                if len(rf) < WINDOW:
                    continue
                tag = tx.replace(":", "") if tx else "xx"
                ld = f"{root}/cam_ds/weapon/node{rx}/link{tag}/{sess}"
                lr = f"{root}/cam_rec/{sess}/link{tag}_node{rx}"
                saveRecording(rf, lr)
                _collect_source(RecordingSource(lr), f"{req.calibration}/node{rx}", ld, [],
                                stage="weapon", labeler=labels,
                                session_id=sess, subject_id="cam",
                                subtract_ic_baseline=True)
                self.log(f"[CAM] per-link weapon {tx}->{rx} -> {ld}")

        self.log(f"[CAM] Done. per-node nodes: {presBuilt}")
        self._emitInference({"event": "pipeline_done"})
        self.is_running = False

    def stop(self):
        self.is_running = False
        self.log("System halt requested.")

