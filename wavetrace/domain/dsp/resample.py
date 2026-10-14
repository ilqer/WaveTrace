import numpy as np

from wavetrace import CsiFrame


def resample_uniform(frames, sample_rate_hz):
    """Linear interpolation onto a uniform 1/sample_rate_hz grid; output is time-sorted. O(n·A·S).

    Mesh rate jitters (30-300 Hz); pinning it prevents dropped windows and smeared spectrograms.
    Pass a single-link stream."""
    if sample_rate_hz <= 0:
        raise ValueError("resample_uniform: sample_rate_hz must be positive")
    if len(frames) < 2:
        return list(frames)
    num_antennas, num_subcarriers = frames[0].grid.shape
    timestamps_s = np.array([frame.timestamp for frame in frames], dtype=np.float64)
    order = np.argsort(timestamps_s, kind="stable")
    timestamps_s = timestamps_s[order]
    flat_grids = np.stack([frames[i].grid.reshape(-1) for i in order]).astype(np.complex64)  # (n, A·S)
    num_resampled_frames = max(2, int((timestamps_s[-1] - timestamps_s[0]) * sample_rate_hz) + 1)
    uniform_timestamps_s = (timestamps_s[0]
                            + np.arange(num_resampled_frames, dtype=np.float64) / sample_rate_hz)
    real_parts = np.empty((num_resampled_frames, num_antennas * num_subcarriers), dtype=np.float32)
    imaginary_parts = np.empty_like(real_parts)
    # per-cell 1-D interp; A·S is small (≈64 for HT20) so this is cheap
    for cell in range(num_antennas * num_subcarriers):
        real_parts[:, cell] = np.interp(uniform_timestamps_s, timestamps_s, flat_grids[:, cell].real)
        imaginary_parts[:, cell] = np.interp(uniform_timestamps_s, timestamps_s, flat_grids[:, cell].imag)
    resampled_grids = (real_parts + 1j * imaginary_parts).astype(np.complex64).reshape(
        num_resampled_frames, num_antennas, num_subcarriers
    )
    node_id = getattr(frames[0], "node_id", 0)
    result = []
    for i in range(num_resampled_frames):
        frame = CsiFrame(num_antennas, num_subcarriers)
        frame.grid[:, :] = resampled_grids[i]
        frame.timestamp = float(uniform_timestamps_s[i])
        frame.node_id = node_id
        result.append(frame)
    return result
