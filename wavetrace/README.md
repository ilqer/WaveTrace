# WaveTrace Python Library (`wavetrace/`)

This folder is the main Python brain of the project. While the `src/` folder handles the heavy math in C++, this folder takes those processed signals and runs the machine learning models to figure out what is happening in the room.

## Main Files
* **`Source.py`**: Connects to the incoming data streams (like reading live UDP data from the mesh or reading saved recordings).
* **`Frontend.py`**: The pipeline manager. It takes the raw data source, sends it to the C++ core for processing, and prepares it for the machine learning models.
* **`Calibration.py`**: Handles the logic for figuring out what the empty room looks like.
* **`Cli.py`**: The command-line interface logic (what runs when you type terminal commands).

## Subfolders Overview

* **`recognition/`**: This is where the machine learning happens. It contains scripts to train, evaluate, and run live predictions using models built with `scikit-learn`. 
* **`groundtruth/`**: Tools used during data collection to sync up camera feeds with WiFi data so we know exactly when a person was present during training.
* **`diagnostics/`**: Contains telemetry code to monitor the health of the system.
* **`output/`**: Handles pushing the final predictions out to the user or web dashboard.
