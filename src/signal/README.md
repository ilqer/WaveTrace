# `src/signal/`

DSP pipeline: cleans the raw CSI and extracts features the models can use.

| File | What it does |
|---|---|
| `Preprocess.hpp` | Conjugate-multiply (cancels CFO/SFO clock drift), Hampel filter (removes outlier spikes), phase unwrap, EMA detrend |
| `GainLock.hpp` | Fixes the AGC amplification to the calibration value so amplitude is comparable across sessions |
| `Features.hpp` | Nine per-subcarrier statistics (mean, std, max, min, IQR, skew, lag-1 autocorr, MAD, waveform length); inter-subcarrier µ[p] and σ²[p] (weapon discriminator); `reconstruct_complex_csi` |
| `Spectrogram.hpp` | Builds a (subcarriers × time) amplitude image for the CNN path |
| `PresenceSegment.hpp` | Splits a continuous frame stream into motion segments (active vs. quiet) for the voter |
| `SubcarrierSelect.hpp` | NBVI: ranks subcarriers by presence sensitivity and keeps the top K |
