# WaveTrace Web Dashboard (`web/`)

This folder contains the visualization tools for the project. Because looking at terminal output isn't always helpful for debugging WiFi signals, we use a web dashboard to see the CSI heatmaps and predictions in real-time.

## Backend (Python)
* **`streamer.py`**: This script takes the live data and predictions from the WaveTrace system and streams them over WebSockets so the frontend can read them.
* **`app.py` & `WsPublisher.py`**: Handlers for the server and WebSocket connections.
* **`foxglove.py`**: Integration with Foxglove Studio, which is a robotics visualization tool we can use to look at the 3D data.

## Frontend (`ui/` folder)
* **`ui/`**: This is a React web application built with Vite and TypeScript. 
* To run it, you navigate into this folder and run `npm run dev`. 
