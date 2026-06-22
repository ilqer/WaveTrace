# WaveTrace Diagnostics (`wavetrace/diagnostics/`)

This folder is used to monitor the health and performance of the WaveTrace system while it is running.

* **`Telemetry.py`**: Gathers metrics about the system (like how many packets we are dropping, how fast the C++ code is running, and memory usage) and packages it so we can view it on the web dashboard or print it to the terminal.
