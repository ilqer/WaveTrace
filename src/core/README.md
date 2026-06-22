# WaveTrace Core Types (`src/core/`)

This folder contains the fundamental data structures used throughout the C++ backend. 

* **`CsiFrame.hpp`**: The most important class. It represents a single "snapshot" of WiFi Channel State Information (CSI) received at a specific time. It holds the complex numbers (amplitude and phase) for every antenna and subcarrier.
* **`Types.hpp`**: Defines standard data types and aliases to keep the C++ code clean.
* **`Errors.hpp`**: Custom error handling classes so we know exactly what went wrong if the signal processing fails.
