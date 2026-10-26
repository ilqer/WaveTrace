"""Which signed OFDM tone each CSI frame column holds, per capture layout.

A frame column is not a tone index: ESP32-S3 frames store each LTF field in FFT order (tones
0..N/2-1 then -N/2..-1), the HT40 L-LTF is centred instead, and the second field repeats the tones
of the first. Delay-domain code (CIR, range, delay steering) needs the signed tone, or the FFT wrap
adds a 2π·(N·Δf)·τ phase jump mid-frame.
"""

from dataclasses import dataclass, field

import numpy as np

SUBCARRIER_SPACING_HZ = 312_500.0


@dataclass(frozen=True, slots=True)
class ToneField:
    """One LTF field: columns first_column .. first_column + len(tone_indices) - 1 of the frame.
    `carrying_columns` / `carrying_tone_indices` are the carrying tones sorted ascending by signed
    tone, precomputed so selecting a field is one fancy index."""

    name: str
    first_column: int
    tone_indices: np.ndarray    # int16 (N,), signed tone of each column
    is_carrying: np.ndarray     # bool (N,), False on DC, guard and null tones
    carrying_columns: np.ndarray = field(init=False, repr=False)
    carrying_tone_indices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        field_columns = np.flatnonzero(self.is_carrying)
        field_columns = field_columns[np.argsort(self.tone_indices[field_columns], kind="stable")]
        carrying_columns = (self.first_column + field_columns).astype(np.intp)
        carrying_tone_indices = self.tone_indices[field_columns].astype(np.int16)
        for array in (self.tone_indices, self.is_carrying, carrying_columns, carrying_tone_indices):
            array.setflags(write=False)
        object.__setattr__(self, "carrying_columns", carrying_columns)
        object.__setattr__(self, "carrying_tone_indices", carrying_tone_indices)


@dataclass(frozen=True, slots=True)
class ToneLayout:
    name: str
    subcarrier_spacing_hz: float
    num_columns: int
    fields: tuple[ToneField, ...]
    default_field: str          # the field delay-domain code reads when none is named

    def __post_init__(self):
        if self.default_field not in {tone_field.name for tone_field in self.fields}:
            raise ValueError(f"tone layout {self.name}: default_field {self.default_field!r} names no field; "
                             f"fields: {[tone_field.name for tone_field in self.fields]}")

    def field_named(self, field_name: str) -> ToneField:
        for tone_field in self.fields:
            if tone_field.name == field_name:
                return tone_field
        raise ValueError(f"tone layout {self.name} has no field {field_name!r}; "
                         f"fields: {[tone_field.name for tone_field in self.fields]}")


def _fft_order_tones(num_tones: int) -> np.ndarray:
    return np.fft.fftfreq(num_tones, d=1.0 / num_tones).astype(np.int16)


def _centred_tones(num_tones: int) -> np.ndarray:
    return (np.arange(num_tones) - num_tones // 2).astype(np.int16)


def _tone_field(name: str, first_column: int, tone_indices: np.ndarray,
                lowest_carrying_tone: int, highest_carrying_tone: int) -> ToneField:
    magnitude = np.abs(tone_indices)
    is_carrying = (magnitude >= lowest_carrying_tone) & (magnitude <= highest_carrying_tone)
    return ToneField(name, first_column, tone_indices, is_carrying)


# ESP-IDF ESP32-S3 CSI docs: LLTF then HT-LTF; HT20 both `0~31, -32~-1`; HT40 LLTF `0~63` (centred
# on the primary channel, primary-relative tones -32..31), HT-LTF `0~63, -64~-1` on the 128-point
# grid. Carrying ranges match the always-zero columns measured on data/2g4_ht20 and data/2g4_ht40.
ESP32S3_HT20 = ToneLayout("esp32s3_ht20", SUBCARRIER_SPACING_HZ, 128, (
    _tone_field("l_ltf", 0, _fft_order_tones(64), 1, 26),
    _tone_field("ht_ltf", 64, _fft_order_tones(64), 1, 28),
), "ht_ltf")
ESP32S3_HT40 = ToneLayout("esp32s3_ht40", SUBCARRIER_SPACING_HZ, 192, (
    _tone_field("l_ltf", 0, _centred_tones(64), 1, 26),
    _tone_field("ht_ltf", 64, _fft_order_tones(128), 2, 58),
), "ht_ltf")
# FFT order is assumed and the carrying set is unmeasured (no Pi capture yet): every non-DC tone carries.
NEXMON_VHT80 = ToneLayout("nexmon_vht80", SUBCARRIER_SPACING_HZ, 256, (
    _tone_field("vht_ltf", 0, _fft_order_tones(256), 1, 128),
), "vht_ltf")

TONE_LAYOUTS = {layout.name: layout for layout in (ESP32S3_HT20, ESP32S3_HT40, NEXMON_VHT80)}
_LEGACY_LAYOUT_BY_NUM_COLUMNS = {128: ESP32S3_HT20, 192: ESP32S3_HT40}


def tone_layout_by_name(name: str) -> ToneLayout:
    try:
        return TONE_LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tone layout {name!r}; known: {sorted(TONE_LAYOUTS)}") from None


def infer_tone_layout(num_columns: int) -> ToneLayout:
    """Width-only guess for recordings that predate the explicit layout name; valid only because
    every such recording is ESP32-S3 (width 128 is also a Nexmon 40 MHz frame)."""
    try:
        return _LEGACY_LAYOUT_BY_NUM_COLUMNS[int(num_columns)]
    except KeyError:
        raise ValueError(f"cannot infer a tone layout from {num_columns} columns; "
                         f"name it explicitly ({sorted(TONE_LAYOUTS)})") from None


def select_field(csi: np.ndarray, layout: ToneLayout,
                 field: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(..., num_columns) CSI -> ((..., K) carrying CSI, (K,) int16 signed tones), sorted ascending
    by tone; `field` defaults to `layout.default_field`. One fancy index over the last axis, O(K) per row;
    the tone array is shared, read-only."""
    csi = np.asarray(csi)
    if csi.shape[-1] != layout.num_columns:
        raise ValueError(f"select_field: CSI has {csi.shape[-1]} columns, "
                         f"layout {layout.name} has {layout.num_columns}")
    tone_field = layout.field_named(layout.default_field if field is None else field)
    return csi[..., tone_field.carrying_columns], tone_field.carrying_tone_indices
