# `web/ui/` — React dashboard

The web frontend for WaveTrace. Shows live CSI spectrograms, per-node health, and model predictions. Connects to the Python backend over WebSockets.

## Run

```bash
cd web/ui
npm install       # once
npm run dev       # starts the Vite dev server
```

Open `http://localhost:5173` in a browser. The Python backend (`python web/app.py`, port 8000) must be running for live data.

## Structure

| Path | What it is |
|---|---|
| `src/components/` | UI building blocks (node health table, spectrogram, controls, device panel) |
| `src/hooks/useWaveTrace.ts` | WebSocket connection to the Python backend; feeds data to components |
| `src/hooks/useDevice.ts` | Device control state (flash, recalibrate buttons) |
| `vite.config.ts` | Vite build config |
