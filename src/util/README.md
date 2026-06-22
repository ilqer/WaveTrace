# WaveTrace Utilities (`src/util/`)

This folder contains generic helper tools used by the rest of the C++ code.

* **`Fft.hpp`**: A fast implementation of the Fast Fourier Transform. This is heavy math used to convert signals from the time domain (how it changes over time) into the frequency domain (what pitches/frequencies are present).
* **`RingBuffer.hpp`**: A highly efficient memory structure that lets us store a rolling window of recent data without constantly copying arrays in memory.
