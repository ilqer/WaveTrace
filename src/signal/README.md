# WaveTrace Signal Processing (`src/signal/`)

This is the math engine of the project. Raw WiFi CSI is incredibly noisy—it drifts, it spikes, and it jumps around. This folder cleans it up and extracts the actual human movement.

* **`Preprocess.hpp`**: Cleans the raw signal. It removes hardware artifacts like Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO) which cause the signal phase to spin randomly.
* **`GainLock.hpp`**: WiFi routers constantly adjust their "volume" (AGC - Automatic Gain Control). This class reverses that, locking the signal gain so our models see consistent data.
* **`Features.hpp`**: Extracts statistical features (like variance and mean) that describe how much the signal is changing. This is what the machine learning model actually looks at.
* **`Spectrogram.hpp`**: Converts the signal into a spectrogram (a 2D image of frequencies over time), which is extremely useful for detecting specific movements or weapons.
* **`PresenceSegment.hpp`**: Helps chop a continuous stream of data into distinct "events" or segments of activity.
* **`SubcarrierSelect.hpp`**: Not all WiFi frequencies (subcarriers) are useful. Some are just noise. This picks the best ones to listen to.
