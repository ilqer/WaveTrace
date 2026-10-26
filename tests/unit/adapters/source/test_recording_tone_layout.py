"""The recording meta.json names its tone layout; legacy recordings without the key get it from width."""

import json
import logging

import numpy as np

from wavetrace import CsiFrame
from wavetrace.adapters.source.recording import load_recording, load_tone_layout, save_recording
from wavetrace.domain.tone_layout import ESP32S3_HT20, ESP32S3_HT40, NEXMON_VHT80


def _frames(num_subcarriers, num_frames=3):
    frames = []
    for i in range(num_frames):
        frame = CsiFrame(1, num_subcarriers)
        frame.timestamp = float(i)
        frame.node_id = 1
        frames.append(frame)
    return frames


def test_writer_saves_tone_layout_and_reader_returns_it(tmp_path):
    save_recording(_frames(256), tmp_path, tone_layout=NEXMON_VHT80)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["tone_layout"] == "nexmon_vht80"
    assert meta["num_subcarriers"] == 256  # existing keys keep their names
    assert load_tone_layout(tmp_path) is NEXMON_VHT80


def test_writer_without_layout_omits_the_key(tmp_path):
    save_recording(_frames(128), tmp_path)
    assert "tone_layout" not in json.loads((tmp_path / "meta.json").read_text())


def test_legacy_recording_infers_layout_from_width_with_one_warning(tmp_path, caplog):
    save_recording(_frames(192), tmp_path / "ht40")
    save_recording(_frames(128), tmp_path / "ht20")
    with caplog.at_level(logging.WARNING):
        assert load_tone_layout(tmp_path / "ht40") is ESP32S3_HT40
    assert len([record for record in caplog.records if "tone_layout" in record.getMessage()]) == 1
    assert load_tone_layout(tmp_path / "ht20") is ESP32S3_HT20


def test_saved_frames_still_round_trip(tmp_path):
    frames = _frames(128)
    frames[1].grid[:, :] = np.full((1, 128), 2 + 1j, dtype=np.complex64)
    save_recording(frames, tmp_path, tone_layout=ESP32S3_HT20)
    assert np.allclose(list(load_recording(tmp_path))[1].grid, 2 + 1j)
