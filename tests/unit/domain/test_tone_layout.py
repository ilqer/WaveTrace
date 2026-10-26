"""Tone-index tables per CSI capture layout.

The non-carrying columns of each ESP32-S3 layout must equal the always-zero columns measured on
real recordings (data/2g4_ht20, data/2g4_ht40), and an FFT-order frame with one path at a known
delay must come back at that delay once its carrying tones are selected and sorted.
"""

import numpy as np
import pytest

from wavetrace.domain.tone_layout import (
    ESP32S3_HT20,
    ESP32S3_HT40,
    NEXMON_VHT80,
    ToneLayout,
    infer_tone_layout,
    select_field,
    tone_layout_by_name,
)
from wavetrace.recognition.cir import cir_from_csi

MEASURED_ZERO_COLUMNS = {
    "esp32s3_ht20": [0, *range(27, 38), 64, *range(93, 100)],
    "esp32s3_ht40": [*range(0, 6), 32, *range(59, 66), *range(123, 134), 191],
}


def _non_carrying_columns(layout):
    columns = []
    for tone_field in layout.fields:
        columns += (tone_field.first_column + np.flatnonzero(~tone_field.is_carrying)).tolist()
    return sorted(columns)


def _fft_order_frame(layout, delay_s, num_antennas=1):
    """(A, num_columns) CSI of one path at `delay_s`: H = exp(-j2π·tone·Δf·τ) on carrying columns, 0 on nulls."""
    frame = np.zeros((num_antennas, layout.num_columns), dtype=np.complex64)
    for tone_field in layout.fields:
        columns = tone_field.first_column + np.flatnonzero(tone_field.is_carrying)
        tones = tone_field.tone_indices[tone_field.is_carrying]
        frame[:, columns] = np.exp(-2j * np.pi * tones * layout.subcarrier_spacing_hz * delay_s)
    return frame


@pytest.mark.parametrize("layout", [ESP32S3_HT20, ESP32S3_HT40])
def test_non_carrying_columns_equal_measured_zero_columns(layout):
    assert _non_carrying_columns(layout) == MEASURED_ZERO_COLUMNS[layout.name]


@pytest.mark.parametrize("layout, num_carrying", [(ESP32S3_HT20, (52, 56)), (ESP32S3_HT40, (52, 114))])
def test_carrying_tone_counts(layout, num_carrying):
    assert tuple(int(tone_field.is_carrying.sum()) for tone_field in layout.fields) == num_carrying


def test_layouts_cover_every_column_once():
    for layout in (ESP32S3_HT20, ESP32S3_HT40, NEXMON_VHT80):
        assert layout.subcarrier_spacing_hz == 312_500.0
        assert sum(tone_field.tone_indices.size for tone_field in layout.fields) == layout.num_columns
        assert layout.fields[0].first_column == 0


def test_ht40_l_ltf_is_centred_and_ht_ltf_is_fft_order():
    l_ltf, ht_ltf = ESP32S3_HT40.fields
    assert l_ltf.tone_indices[0] == -32 and l_ltf.tone_indices[32] == 0 and l_ltf.tone_indices[63] == 31
    assert ht_ltf.tone_indices[0] == 0 and ht_ltf.tone_indices[63] == 63 and ht_ltf.tone_indices[64] == -64


def test_nexmon_vht80_marks_every_non_dc_tone_carrying():
    (vht_ltf,) = NEXMON_VHT80.fields
    assert NEXMON_VHT80.num_columns == 256
    assert np.flatnonzero(~vht_ltf.is_carrying).tolist() == [0]


def test_select_field_returns_carrying_tones_ascending():
    csi = np.arange(128, dtype=np.complex64)  # value = column, so the output names the columns picked
    carrying_csi, tone_indices = select_field(csi, ESP32S3_HT20)
    assert tone_indices.tolist() == [*range(-28, 0), *range(1, 29)]
    # HT-LTF column 64 + c holds tone c for c < 32, c - 64 otherwise
    assert carrying_csi.real.astype(int).tolist() == [64 + (tone % 64) for tone in tone_indices]


def test_select_field_keeps_leading_axes():
    csi = np.ones((3, 2, 192), dtype=np.complex64)
    carrying_csi, tone_indices = select_field(csi, ESP32S3_HT40, field="l_ltf")
    assert carrying_csi.shape == (3, 2, 52) and tone_indices.size == 52


def test_select_field_rejects_wrong_width_and_unknown_field():
    with pytest.raises(ValueError, match="columns"):
        select_field(np.ones(192, dtype=np.complex64), ESP32S3_HT20)
    with pytest.raises(ValueError, match="field"):
        select_field(np.ones(128, dtype=np.complex64), ESP32S3_HT20, field="vht_ltf")


def test_lookup_by_name_and_legacy_width_inference():
    assert tone_layout_by_name("esp32s3_ht40") is ESP32S3_HT40
    assert infer_tone_layout(128) is ESP32S3_HT20
    assert infer_tone_layout(192) is ESP32S3_HT40
    with pytest.raises(ValueError):
        infer_tone_layout(256)
    with pytest.raises(ValueError):
        tone_layout_by_name("esp32c6_ht20")


@pytest.mark.parametrize("layout", [ESP32S3_HT20, ESP32S3_HT40])
def test_fft_order_frame_gives_planted_delay_through_cir(layout):
    delay_s = 30e-9
    frame = _fft_order_frame(layout, delay_s)[0]
    carrying_csi, tone_indices = select_field(frame, layout)
    cir = cir_from_csi(carrying_csi, subcarrier_indices=tone_indices,
                       subcarrier_spacing_hz=layout.subcarrier_spacing_hz, lam=0.02, max_iterations=200)
    bin_width_s = cir.tap_delays_s[1] - cir.tap_delays_s[0]
    assert abs(cir.dominant_delay_s - delay_s) <= bin_width_s


def test_default_field_is_the_delay_field_and_select_field_uses_it():
    assert ESP32S3_HT20.default_field == "ht_ltf" and ESP32S3_HT40.default_field == "ht_ltf"
    assert NEXMON_VHT80.default_field == "vht_ltf"
    _, tone_indices = select_field(np.ones(256, dtype=np.complex64), NEXMON_VHT80)
    assert tone_indices.size == 255


def test_default_field_must_name_a_field():
    with pytest.raises(ValueError, match="default_field"):
        ToneLayout("bad", 312_500.0, 128, ESP32S3_HT20.fields, default_field="vht_ltf")
