# WaveTrace Recognition (Machine Learning) (`wavetrace/recognition/`)

This is the brain of the Python code. It takes the cleaned WiFi signals and learns to recognize patterns.

* **`Train.py`**: The script that feeds the dataset into `scikit-learn` to train a new machine learning model.
* **`Evaluate.py`**: Tests the trained model to see how accurate it is, generating statistics like false positive rates.
* **`Infer.py`**: Runs a trained model on live data to make real-time predictions.
* **`Weapon.py` & `Occupancy.py`**: Specific logic for different types of detection. Weapon detection requires different math than just checking if a room is occupied.
* **`Vote.py` & `Fusion.py`**: Because we have multiple ESP32 nodes, each one might make a different guess. These scripts combine the guesses (voting) to make a highly confident final decision.
* **`Heatmap.py`**: Generates a spatial representation of the room to try and locate *where* the person is.
