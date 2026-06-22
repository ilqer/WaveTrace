# WaveTrace Ground Truth (`wavetrace/groundtruth/`)

"Ground Truth" means knowing exactly what actually happened in the room, so we can train the AI to recognize it accurately.

* **`CameraLabeler.py`**: When we collect training data, we usually record a video at the same time. This script helps us look at the video to say exactly when a person walked into the frame, so we can tag the WiFi data with "Person Present".
* **`DatasetBuilder.py`**: Takes the labels (from the camera or from manual tagging) and pairs them with the raw WiFi CSI data to build a clean dataset that our machine learning algorithms can use for training.
* **`Align.py`**: Synchronizes the timestamps between the camera and the ESP32 boards, since their clocks are never perfectly matched.
