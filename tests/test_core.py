import numpy as np
import pytest

from wavetrace import CsiFrame, FrameError, Label, RecognitionResult
from wavetrace.synthetic import generate_stream


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
    frame.grid[1, 5] = 3.0 - 2.0j
    assert frame.grid[1, 5] == 3.0 - 2.0j   # a fresh view sees the same buffer
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
    result = RecognitionResult()
    assert result.class_id == -1
    assert result.confidence == 0.0
    assert result.bbox is None
    assert result.keypoints == []
    result.class_id = 2
    result.confidence = 0.91
    result.bbox = (0.1, 0.2, 0.3, 0.4)
    result.keypoints = [1.0, 2.0, 0.9]
    assert result.class_id == 2
    assert list(result.bbox) == pytest.approx([0.1, 0.2, 0.3, 0.4])  # float32 round-trip
    assert result.keypoints == pytest.approx([1.0, 2.0, 0.9])


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


def _recover_perturbation_hz(frames, sample_rate_hz, low_subcarrier, high_subcarrier,
                             band_low_hz, band_high_hz):
    """Reference recovery of motion frequency via cross-subcarrier differential phase -> Hann -> FFT.
    CFO cancels in conjugate product."""
    cross_subcarrier_product = np.array(
        [frame.grid[0, high_subcarrier] * np.conj(frame.grid[0, low_subcarrier]) for frame in frames])
    # undo the 2pi jumps from the static phase offset
    differential_phase = np.unwrap(np.angle(cross_subcarrier_product))
    differential_phase = differential_phase - differential_phase.mean()
    sample_count = len(differential_phase)
    windowed_phase = differential_phase * np.hanning(sample_count)
    # Zero-pad well beyond the signal length to resolve the band finely.
    fft_length = 1 << (max(8 * sample_count, 64) - 1).bit_length()
    spectrum = np.fft.rfft(windowed_phase, fft_length)
    power = spectrum.real**2 + spectrum.imag**2
    frequencies_hz = np.fft.rfftfreq(fft_length, d=1.0 / sample_rate_hz)
    band = (frequencies_hz >= band_low_hz) & (frequencies_hz <= band_high_hz)
    return frequencies_hz[band][np.argmax(power[band])]


def test_perturbation_recovered_by_reference_fft():
    sample_rate_hz = 100.0
    true_frequency_hz = 0.3  # injected motion frequency (e.g. 18 cycles/min)
    frames, ground_truth = generate_stream(
        num_antennas=2, num_subcarriers=64, sample_rate_hz=sample_rate_hz, num_frames=512,
        perturbation_hz=true_frequency_hz, perturbation_depth=0.5, cfo_hz=4.0, noise_std=0.01, seed=7,
    )
    recovered_frequency_hz = _recover_perturbation_hz(
        frames, sample_rate_hz, low_subcarrier=0, high_subcarrier=63, band_low_hz=0.1, band_high_hz=2.0)
    # CFO and noise must not shift the recovered frequency.
    assert recovered_frequency_hz == pytest.approx(ground_truth["perturbation_hz"], abs=0.05)
