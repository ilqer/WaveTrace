# WaveTrace React Dashboard (`web/ui/`)

This is the frontend code for the web dashboard. It is a React application built with Vite and TypeScript. 

It connects to the Python backend via WebSockets to display live telemetry, CSI spectrograms, and the final predictions of the machine learning models.

## How to Run
Make sure you have Node.js installed.
1. Run `npm install` to get the dependencies.
2. Run `npm run dev` to start the local development server.

## Folder Structure
* **`src/components/`**: The visual building blocks of the UI. For example, `NodeHealth.tsx` displays a table of the ESP32 boards, and `Spectrogram.tsx` draws the live WiFi frequency images.
* **`src/hooks/`**: Custom React hooks (like `useWaveTrace.ts`) that handle the WebSocket connection to the Python backend.
