# `web/` — web dashboard backend

Python backend that streams live CSI data and model predictions to the React frontend over WebSockets.

## Run

```bash
# start the backend
python web/streamer.py

# start the frontend (separate terminal)
cd web/ui && npm install && npm run dev
# open http://localhost:5173
```

## Files

| File | What it does |
|---|---|
| `streamer.py` | Pulls live data from the WaveTrace pipeline and pushes it to WebSocket clients |
| `app.py` | Flask app and HTTP routes |
| `WsPublisher.py` | WebSocket publisher — a `Publisher` implementation that writes to connected browser clients |
| `device_ctl.py` | Device control endpoints: flash firmware, trigger recalibration, upload a model |
| `foxglove.py` | Optional Foxglove Studio integration for 3D visualization |

## Frontend

See [`ui/README.md`](ui/README.md).
