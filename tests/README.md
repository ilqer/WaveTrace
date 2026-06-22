# WaveTrace Tests (`tests/`)

This folder contains all the automated tests to make sure the code doesn't break when we make changes. 

## How It Works
We use a Python testing framework called `pytest`. You can run all the tests by simply typing `pytest` in your terminal while in the root of the project.

## What is Tested?
* **C++ Core (`TestCore.py`, `TestFrameParser.py`)**: Tests to ensure the C++ math and packet parsing are accurate and memory-safe.
* **Signal Processing (`TestFeatures.py`, `TestPreprocess.py`)**: Checks that the noise filtering and feature extraction are producing the correct numbers.
* **Machine Learning (`TestRecognition.py`, `TestWeapon.py`)**: Verifies that the models can train correctly and make accurate predictions on dummy data.
* **Networking (`TestUdpSource.py`, `TestMeshLinks.py`)**: Ensures that the code correctly handles dropped packets or messy network streams.
