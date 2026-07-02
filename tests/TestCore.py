"""Core types + synthetic-CSI DSP checks."""

import numpy as np
import pytest

from wavetrace import CsiFrame, FrameError, Label, RecognitionResult
from fixtures.SyntheticCsi import generateStream


# --- core types -------------------------------------------------------------------------

def test_csiframe_shape_and_dtype():
    frame = CsiFrame(num_antennas=3, num_subcarriers=64)
    assert frame.num_antennas == 3
    assert frame.num_subcarriers == 64
    assert frame.size == 3 * 64
    assert frame.grid.shape == (3, 64)
    assert frame.grid.dtype == np.complex64
    assert frame.node_id == -1  # single-node default


def test_csiframe_grid_is_zero_copy():
    frame = CsiFrame(num_antennas=2, num_subcarriers=8)
    frame.grid[1, 5] = 3.0 - 2.0j           # write through the view
    assert frame.grid[1, 5] == 3.0 - 2.0j   # a fresh view sees the same buffer
    # Confirm shared memory.
    view = frame.grid
    view[0, 0] = 7.0 + 1.0j
    assert frame.grid[0, 0] == 7.0 + 1.0j


def test_csiframe_reshape_and_metadata():
    frame = CsiFrame(num_antennas=2, num_subcarriers=4)
    frame.timestamp = 12.5
    frame.node_id = 7
    frame.reshape(1, 16)
    assert frame.grid.shape == (1, 16)
    assert frame.timestamp == 12.5
    assert frame.node_id == 7


def test_csiframe_invalid_index_raises_frameerror():
    frame = CsiFrame(num_antennas=1, num_subcarriers=4)
    with pytest.raises(FrameError):
        CsiFrame(num_antennas=0, num_subcarriers=4)
    assert frame.grid.shape == (1, 4)


def test_recognition_result_fields():
    r = RecognitionResult()
    assert r.class_id == -1
    assert r.confidence == 0.0
    assert r.bbox is None
    assert r.keypoints == []
    r.class_id = 2
    r.confidence = 0.91
    r.bbox = (0.1, 0.2, 0.3, 0.4)
    r.keypoints = [1.0, 2.0, 0.9]
    assert r.class_id == 2
    assert list(r.bbox) == pytest.approx([0.1, 0.2, 0.3, 0.4])  # float32 round-trip
    assert r.keypoints == pytest.approx([1.0, 2.0, 0.9])


def test_label_fields():
    label = Label()
    assert label.class_id == -1
    assert label.name == ""
    assert label.bbox is None
    label.name = "standing"
    label.class_id = 0
    label.timestamp = 3.0
    label.bbox = (0.0, 0.0, 1.0, 1.0)
    assert label.name == "standing"
    assert list(label.bbox) == pytest.approx([0.0, 0.0, 1.0, 1.0])


# DSP perturbation recovery.

def _recoverPerturbationHz(frames, sampleRateHz, scLo, scHi, fLo, fHi):
    """Reference recovery of motion frequency via cross-subcarrier differential phase -> Hann -> FFT.
    CFO cancels in conjugate product."""
    s = np.array([f.grid[0, scHi] * np.conj(f.grid[0, scLo]) for f in frames])
    sig = np.unwrap(np.angle(s))  # §2.3: undo 2pi jumps from the static phase offset
    sig = sig - sig.mean()
    n = len(sig)
    x = sig * np.hanning(n)
    # Zero-pad well beyond N to resolve band finely.
    nfft = 1 << (max(8 * n, 64) - 1).bit_length()
    spec = np.fft.rfft(x, nfft)
    power = spec.real**2 + spec.imag**2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sampleRateHz)
    band = (freqs >= fLo) & (freqs <= fHi)
    return freqs[band][np.argmax(power[band])]


def test_perturbation_recovered_by_reference_fft():
    fs = 100.0
    fTrue = 0.3  # injected motion frequency (e.g. 18 cycles/min)
    frames, gt = generateStream(
        numAntennas=2, numSubcarriers=64, sampleRateHz=fs, numFrames=512,
        perturbationHz=fTrue, perturbationDepth=0.5, cfoHz=4.0, noiseStd=0.01, seed=7,
    )
    recovered = _recoverPerturbationHz(frames, fs, scLo=0, scHi=63, fLo=0.1, fHi=2.0)
    # Recover injected frequency despite CFO and noise.
    assert recovered == pytest.approx(gt["perturbation_hz"], abs=0.05)
