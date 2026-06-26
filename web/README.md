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
| `GET` | `/api/paths/scan` | Scan the project tree for calibration dirs, model files, dataset dirs |
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

Warning: `/api/script/run` passes the script name and args to the shell via `device_hub.run_script`. Do not expose this server to untrusted users.

## Files

| File | What it does |
|---|---|
| `app.py` | FastAPI app, all HTTP routes and WebSocket endpoints |
| `streamer.py` | `WaveTraceRunner` — drives the WaveTrace pipeline in a worker thread and pushes results to the queues |
| `device_ctl.py` | `DeviceHub` — serial monitor, firmware flash, Pi SSH, local script runner |
| `foxglove.py` | Optional Foxglove Studio integration for 3D visualization |

## Frontend

See [`ui/README.md`](ui/README.md).
