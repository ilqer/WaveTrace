# `src/util/`

Low-level helpers used by the rest of the C++ code.

| File | What it does |
|---|---|
| `Fft.hpp` | Self-contained radix-2 FFT; used for Doppler/PSD features and the spectrogram. O(n log n). |
| `RingBuffer.hpp` | Fixed-capacity circular buffer for streaming windows; no allocations after construction |
