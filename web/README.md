# `web/` — dashboard backend

FastAPI backend that streams live CSI data and model predictions to the React frontend over WebSockets.

## Run

```bash
# start the backend (port 8000)
python web/app.py

# start the frontend (separate terminal)
cd web/ui && npm install && npm run dev
# open http://localhost:5173
```

## API endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/action/start` | Start calibration, collection, training, or live inference |
| `POST` | `/api/action/stop` | Stop the running pipeline |
| `GET` | `/api/pipeline/state` | Check whether a pipeline is running |
| `GET` | `/api/model/weights` | CNN channel weights for the current model |
| `GET` | `/api/fusion/weights` | Per-band trust weights from a saved BandFusion model |
| `GET` | `/api/weapon/litmus` | σ²[p] go/no-go check (per node or per directed link) |
| `GET` | `/api/calib/info` | Read a saved calibration dir; returns subcarrier count and bandwidth label |
| `GET` | `/api/paths/scan` | Scan the project tree for calibration dirs, model files, dataset dirs (every dir holding a `features.npy`, plus its parent) |
| `GET` | `/api/paths/browse` | Open a native macOS Finder dialog and return the chosen path |
| `GET` | `/api/serial/ports` | List connected serial ports |
| `GET` | `/api/device/state` | Current state of all device monitors |
| `POST` | `/api/serial/monitor/start` | Start a serial monitor on a port |
| `POST` | `/api/serial/monitor/stop` | Stop a serial monitor |
| `POST` | `/api/flash` | Build and flash firmware to an ESP32 (runs in background) |
| `POST` | `/api/pi/run` | SSH into the Pi and run a command |
| `POST` | `/api/script/run` | Run a local script by name |
| `POST` | `/api/device/stop` | Stop a running device process |
| `POST` | `/api/device/input` | Send stdin input to a running device process |
| `POST` | `/api/model/upload` | Receive a base64-encoded model.joblib and write it to `output/` |
| `GET` | `/api/camera/check` | One-frame camera probe via ffmpeg |
| `GET` | `/api/camera/stream` | MJPEG webcam stream (optional YOLO annotation) |
| `POST` | `/api/camera/stop` | Stop the camera stream |

WebSocket channels: `/ws/inference`, `/ws/stream`, `/ws/logs`, `/ws/training`, `/ws/telemetry`, `/ws/device`.

Calib (`action: "calib"`) and Collect (`action: "collect"`) always end with `pipeline_done` on the inference
socket and the pipeline no longer running, also after a stop or an error; an error is logged as
`ERROR Calibration failed: ...` / `ERROR Collection failed: ...` with its traceback.

`run` serves a mesh model root (`node*/` dirs) through the same `ServeMeshUseCase` and `ServeMode` as the `run_*.py` scripts: one verdict per 1.5 s fuse, each live link scored as its mode scores it (presence and count on the link's last window, weapon on the mean over its buffer). Any other model runs as one node, with frame averaging, baseline subtraction and the gain lock taken from the model's own contract, so the request's `gainLock`, `frame_average`, `use_baseline` and `vote` fields have no effect on inference.

During inference `/ws/telemetry` sends one health entry per node. `gain_drift` is the node's live mean |H| over its calibrated quiet-room mean |H| (1.0 = gain unchanged) and `subcarriers` are its NBVI picks; a mesh run uses each `node{id}/` calibration, a single-calibration run applies that one calibration to every node.

Warning: `/api/script/run` passes the script name and args to the shell via `device_hub.run_script`. Do not expose this server to untrusted users.

## Files

| File | What it does |
|---|---|
| `app.py` | Builds the FastAPI app: the lifespan (one `DashboardState`, the Foxglove bridge, one broadcast task per channel, stopping the runner on shutdown), CORS, the five routers in route order, and the static UI mount last |
| `dashboard_state.py` | `DashboardState` (the six channels, `DeviceHub`, the current runner and its task), stored on `app.state` and handed to routes by `Depends(get_dashboard_state)` (the `Dashboard` annotation); `BroadcastChannel` — one queue sent to every WebSocket client of one channel (inference and stream also go to Foxglove first) |
| `routers/pipeline.py` | `StartRequest`, `/api/action/start`, `/api/action/stop`, `/api/pipeline/state` and the five pipeline sockets (`/ws/inference`, `/ws/stream`, `/ws/logs`, `/ws/training`, `/ws/telemetry`) |
| `routers/model.py` | `/api/model/weights`, `/api/fusion/weights`, `/api/model/upload` (paths confined to `output/`), `/api/weapon/litmus`, `/api/calib/info` |
| `routers/paths.py` | `/api/paths/scan`, `/api/paths/browse` |
| `routers/device.py` | The serial, flash, Pi, script and device-process routes, and `/ws/device` |
| `routers/camera.py` | `/api/camera/check`, `/api/camera/stream`, `/api/camera/stop` (ffmpeg frames, optional YOLO annotation) |
| `streamer.py` | `WaveTraceRunner` — drives the WaveTrace pipeline in a worker thread and pushes results to the queues; inference consumes `ServeMeshUseCase.iter_fused` (mesh) or `iter_window_verdicts` (one node); camera collection translates the request into `pipeline.collect_camera` (the same use-case and messages as `scripts/collect_camera.py`, 5 Hz labels, subject `cam`, no training, root `train_data`; an older flat calibration dir counts as node 0), and `stop` ends it within a second; training translates the request onto `train_every_node` and the collect scripts' training policies (see Train below) |
| `device_ctl.py` | `DeviceHub` — serial monitor, firmware flash, Pi SSH, local script runner |
| `foxglove.py` | Optional Foxglove Studio integration for 3D visualization |
| `ws_publisher.py` | `WsPublisher` — `Publisher` that serializes each result and puts it on an asyncio queue from a background thread |
| `server_options.py` | `WebServerOptions` — uvicorn host, port and reload settings; validates the port range |

## Train

`action: "train"` trains what the matching collect script trains, on that script's pool under the
capture-profile root `train_data`, into the script's default model root:

| `col_stage` (+ `camera_collect`) | Pool under `train_data` | Heads |
|---|---|---|
| `presence` | `ds_*/node<id>/link_*` | `<root>/model/node<id>` |
| `count` | `count_ds_*/c<k>/node<id>/link_*` (classes 0..highest `c<k>`) | `<root>/model_count/node<id>` |
| `weapon` | the cumulative pool `weapon_ds/node<id>/*` | `<root>/model_weapon/node<id>`, or `node<id>/link<tag>` with `per_link` |
| `presence` + `camera_collect` | `cam_ds/presence/node<id>/*/*` | `<root>/model_camera/node<id>` |

`train_backend` picks the backend (presence and count fall back to `mlp` for anything but `mlp` / `svm`);
`heatmap` instead trains `heatmap.joblib` on the one camera session dir `train_data`
(`<root>/cam_ds/heatmap/<session>`) next to the stage's camera heads (`model_camera`, or `model_weapon` for
weapon); a dataset without camera masks is skipped. `train_out` is ignored. The training socket gets one
`train_init` describing the whole pool, `epoch` messages from the cnn and heatmap fits (`stop` ends them at
the next epoch, and any other training before its next head), and one `done` with the last trained head's metrics; the inference socket always gets
`pipeline_done` at the end, also after an error or when nothing trained. `count` with `camera_collect` is
an error (camera sessions are labelled for presence only) and trains nothing.

## Frontend

See [`ui/README.md`](ui/README.md).
