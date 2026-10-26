"""SpectrogramBuilder: a sliding selected-subcarrier x time CSI image."""

import numpy as np
import pytest

from wavetrace import SpectrogramBuilder, WaveTraceError


def test_spectrogram_shape_and_cadence():
    K, T, H = 12, 128, 32
    builder = SpectrogramBuilder(K, T, H)
    emits = [i for i in range(300) if builder.push(np.full(K, float(i), dtype=np.float32))]
    assert builder.image.shape == (K, T)
    assert emits[0] == T - 1               # First image emitted once window fills.
    assert all((emit - emits[0]) % H == 0 for emit in emits)  # Then emits every hop.


def test_spectrogram_column_order_and_content():
    # Encode frame index and subcarrier to verify (K x T) layout.
    K, T, H = 4, 8, 2
    builder = SpectrogramBuilder(K, T, H)
    emitted_frame_index = None
    for i in range(T):
        if builder.push(np.array([i * 10 + subcarrier for subcarrier in range(K)], dtype=np.float32)):
            emitted_frame_index = i
    assert emitted_frame_index == T - 1
    image = builder.image
    # Column j is frame j (oldest..newest); row s is subcarrier s.
    expected = np.array([[j * 10 + subcarrier for j in range(T)] for subcarrier in range(K)],
                        dtype=np.float32)
    assert np.array_equal(image, expected)


def test_spectrogram_deterministic():
    K, T, H = 6, 16, 4
    rng = np.random.default_rng(0)
    frames = rng.standard_normal((40, K)).astype(np.float32)

    def run():
        builder = SpectrogramBuilder(K, T, H)
        images = []
        for frame in frames:
            if builder.push(np.ascontiguousarray(frame)):
                images.append(builder.image.copy())
        return images

    first_run, second_run = run(), run()
    assert len(first_run) == len(second_run) and all(
        np.array_equal(x, y) for x, y in zip(first_run, second_run))


def test_spectrogram_push_wrong_length_raises():
    builder = SpectrogramBuilder(8, 16, 4)
    with pytest.raises(WaveTraceError):
        builder.push(np.ones(5, dtype=np.float32))
