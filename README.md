# WaveTrace

WaveTrace is a WiFi sensing project that detects human presence and objects (like weapons) in a room without using cameras. It works by analyzing Channel State Information (CSI)—which is essentially the data describing how WiFi waves bounce off walls, objects, and people. 

By setting up a mesh of ESP32 microcontrollers, we can read these WiFi reflections. The heavy signal processing is done in C++ for speed, while Python handles the machine learning to predict what is happening in the room based on the data.

## Tech Stack
* **Hardware**: ESP32 microcontrollers (2.4 GHz) and optionally Raspberry Pi (5 GHz).
* **Signal Processing Core**: C++ (Fast Fourier Transforms, phase unwrapping, noise filtering).
* **Backend & ML**: Python, `scikit-learn` for training models, `pybind11` to link Python and C++.
* **Networking**: Raw UDP sockets to stream CSI data from the ESP32s to the PC.
* **Frontend Dashboard**: React, Vite, and TypeScript (in the `web/ui` folder) to visualize the data.

## What You Need First (Prerequisites)

### 1. Hardware Setup
* You need multiple ESP32 boards flashed with the correct CSI firmware.
* Power them on and place them around the room you want to monitor.
* Ensure your PC is connected to the same WiFi network as the ESP32 boards.

### 2. Software Setup
* **Python**: You need Python installed along with a virtual environment (`.venv`).
* **C++ Compiler**: You need CMake and a working C++ compiler (like GCC or Clang) to build the core `wavetrace` library.
* **Node.js**: Required if you want to run the React web dashboard.

To install the Python dependencies and compile the C++ backend, you typically run:
```bash
pip install -e .
```

## How to Execute the Project

The system works in phases: checking the hardware, calibrating the empty room, training the machine learning model, and finally running the live detection.

### Step 1: Verify the Hardware
Before collecting any data, make sure the ESP32 nodes are actually sending data to your computer over UDP.
* **Check Node Health**: Run `python health_monitor.py`. This listens on port 9877 and displays a live table showing which boards are online, their uptime, and available memory.
* **Verify Mesh Links**: Run `python mesh_verify.py`. This listens on port 9876 and confirms that the transmitters and receivers are successfully exchanging CSI packets.

### Step 2: Calibrate the Room
The system needs to understand what the room looks like when it's completely empty. This baseline allows the code to filter out static objects like furniture.
* Make sure the room is completely empty and quiet.
* Run `python collect_baseline.py`.
* This will save calibration data for each node into the `data/cal/` folder.

### Step 3: Train the Models
Now you need to collect data to train the machine learning models. For basic presence detection:
* Run `python collect_presence.py`.
* The script will guide you: first, it will record while the room is empty. Then, it will tell you to walk around the room.
* The script will automatically train a model for each node and save it to `data/model/`.
*(Note: There are also scripts like `collect_weapon.py` and `collect_count.py` if you want to train the system for weapon detection or people counting).*

### Step 4: Run Live Detection
Once the models are trained, you can run the live detection system.
* Run `python run_live_mesh.py`.
* This script streams live data, passes it through the C++ signal processor, and uses the Python ML models to predict if someone is in the room. It combines the votes from all active ESP32 links to give a final verdict.

### Step 5: Web Dashboard (Optional)
To visualize the CSI data and see what the ML models are doing in real-time:
* Start the backend data streamer: `python web/streamer.py`.
* Start the React frontend: Navigate to `web/ui` and run `npm run dev`.

## Project Structure

* **`src/`**: The C++ core. This does the heavy lifting: parsing raw UDP packets, applying Hampel filters, extracting features, and building spectrograms. The `Bindings.cpp` file connects this C++ code to Python.
* **`wavetrace/`**: The Python ML and logic package. It handles calibration (`Calibration.py`), frontend logic, and machine learning models (inside the `recognition/` folder).
* **Root Scripts (`collect_*.py`, `run_*.py`)**: The scripts you actually run in the terminal to drive the system.
* **`web/`**: Code for the React dashboard and WebSocket publisher.
* **`tests/`**: Unit tests for both the C++ and Python code.
