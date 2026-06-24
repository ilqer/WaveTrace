# `src/` — C++ signal processing core

This folder is compiled into the `wavetrace` Python package via pybind11. It handles the parts of the pipeline that must run fast: parsing raw UDP datagrams, cleaning the signal, and extracting features.

## Compile

```bash
pip install -e .    # runs CMake; rebuilds automatically when you re-run this
```

## Subfolders

| Folder | What it contains |
|---|---|
| `core/` | `CsiFrame` — the type that carries one frame's complex subcarrier amplitudes, timestamp, and node ID |
| `hardware/` | UDP datagram parser + frame assembler (reads the wire-format binary, outputs `CsiFrame`) |
| `signal/` | All DSP: conjugate-multiply (clock drift cancel), Hampel filter (outlier removal), phase unwrap, gain lock, NBVI subcarrier selection, FFT, feature extraction, spectrogram builder |
| `util/` | Radix-2 FFT, ring buffers, math helpers |
| `Bindings.cpp` | pybind11 bridge — exposes every public C++ class and function to Python |

## Key algorithms (in the order the pipeline runs them)

1. **Conjugate multiply** — cancels clock drift (CFO/SFO) between TX and RX by multiplying adjacent antenna or subcarrier pairs. Must run before any feature extraction.
2. **Hampel filter** — removes impulse noise (big amplitude spikes from unrelated Wi-Fi traffic). Window of ±3 samples, threshold 1.4826 × MAD.
3. **Phase unwrap** — removes 2π jumps from the raw phase series so phase-based features are continuous.
4. **Gain lock** (`GainLock`) — fixes the AGC amplification factor to the value measured during calibration. Applied per frame, O(n).
5. **NBVI subcarrier selection** — offline ranking of subcarriers by how much they change when a person enters. Keeps the top K; reduces noise and downstream compute.
6. **Feature extraction** — nine statistical features per subcarrier per window: mean, std, max, min, IQR, skew, lag-1 autocorrelation, MAD, waveform length.
7. **Inter-subcarrier stats** — per-packet mean µ[p] and variance σ²[p] across subcarriers; σ²[p] is the main weapon discriminator.
8. **Spectrogram builder** — stacks amplitude frames into a (subcarriers × time) image for the CNN path.

Do not rename the pybind11 binding names in `Bindings.cpp` or the corresponding stubs in `wavetrace/_wavetrace.pyi` — Python code imports them by name.
