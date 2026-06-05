"""Phase 4 (step 4c) — signal/Spectrogram: the sliding selected-subcarrier × time CSI image.

Validates emit cadence, output shape, chronological column ordering, and determinism — the DoD
"spectrogram has the expected shape and is reproducible on the fixture" check.
"""

import numpy as np
import pytest

from wavetrace import SpectrogramBuilder, WaveTraceError


def test_spectrogram_shape_and_cadence():
    K, T, H = 12, 128, 32
    sb = SpectrogramBuilder(K, T, H)
    emits = [i for i in range(300) if sb.push(np.full(K, float(i), dtype=np.float32))]
    assert sb.image.shape == (K, T)        # selected-subcarrier × time
    assert emits[0] == T - 1               # first image once the window first fills
    assert all((e - emits[0]) % H == 0 for e in emits)  # then every hop


def test_spectrogram_column_order_and_content():
    # Encode frame index + subcarrier into each value so we can check the (K x T) layout exactly.
    K, T, H = 4, 8, 2
    sb = SpectrogramBuilder(K, T, H)
    emitFrame = None
    for i in range(T):
        if sb.push(np.array([i * 10 + s for s in range(K)], dtype=np.float32)):
            emitFrame = i
    assert emitFrame == T - 1
    img = sb.image
    # Column j is frame j (oldest..newest); row s is subcarrier s.
    expected = np.array([[j * 10 + s for j in range(T)] for s in range(K)], dtype=np.float32)
    assert np.array_equal(img, expected)


def test_spectrogram_deterministic():
    K, T, H = 6, 16, 4
    rng = np.random.default_rng(0)
    frames = rng.standard_normal((40, K)).astype(np.float32)

    def run():
        sb = SpectrogramBuilder(K, T, H)
        imgs = []
        for f in frames:
            if sb.push(np.ascontiguousarray(f)):
                imgs.append(sb.image.copy())
        return imgs

    a, b = run(), run()
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_spectrogram_push_wrong_length_raises():
    sb = SpectrogramBuilder(8, 16, 4)
    with pytest.raises(WaveTraceError):
        sb.push(np.ones(5, dtype=np.float32))
