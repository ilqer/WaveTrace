# WaveTrace Hardware Ingest (`src/hardware/`)

This folder is responsible for taking the raw, messy bytes that come over the network from the ESP32 boards and turning them into usable data.

* **`FrameParser.hpp`**: When a UDP packet arrives, it is just a string of bytes. This class decodes those bytes, extracts the actual WiFi CSI data, and puts it into a `CsiFrame` object.
* **`NodeAggregator.hpp`**: Because we have a mesh of multiple ESP32 nodes, data arrives out of order and at different times. This class organizes the incoming frames, syncing them up based on their timestamps so the machine learning models receive a clean, unified picture of the room at any given millisecond.
