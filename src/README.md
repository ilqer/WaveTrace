# WaveTrace C++ Core (`src/`)

This folder contains the high-performance C++ backend for the WaveTrace project. Because processing WiFi Channel State Information (CSI) requires heavy math (like Fast Fourier Transforms and phase unwrapping) to be done very quickly, we do it here in C++ instead of Python.

## How It Connects to Python
We use a tool called `pybind11`. The file `Bindings.cpp` is the bridge. It takes the C++ classes and functions we write here and makes them available to the Python scripts in the `wavetrace/` folder.

## Subfolders Overview

* **`core/`**: Contains the basic data structures. For example, `CsiFrame` defines what a single frame of WiFi data looks like.
* **`hardware/`**: Handles the raw incoming data from the UDP sockets.
* **`signal/`**: The digital signal processing (DSP) math happens here (filtering, noise removal, feature extraction).
* **`util/`**: Helper tools, like math functions and fast memory buffers.

## Compilation
This folder is compiled into a library using CMake. When you run `pip install -e .` in the root of the project, it automatically compiles this C++ code so Python can use it.
