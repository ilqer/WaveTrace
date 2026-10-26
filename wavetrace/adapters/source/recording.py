"""Saved CSI recordings.

A recording is grid.npy (F,A,S) complex64 + t.npy (F,) + node_id.npy (F,) + meta.json, and costs
O(F·A·S) to write or read. meta.json carries "tone_layout" (a domain/tone_layout.py name) when the
writer knew the radio; recordings without it are all ESP32-S3 and get the layout from their width.
"""

import json
import logging
from pathlib import Path

import numpy as np

from wavetrace import CsiFrame
from wavetrace.application.ports import CsiSource
from wavetrace.domain.tone_layout import ToneLayout, infer_tone_layout, tone_layout_by_name


class RecordingSource(CsiSource):
    def __init__(self, recording_dir):
        self._recording_dir = Path(recording_dir)

    def frames(self):
        return load_recording(self._recording_dir)


def save_recording(frames, out_dir, *, tone_layout: ToneLayout | None = None) -> Path:
    frames = list(frames)
    if not frames:
        raise ValueError("save_recording: no frames")
    num_antennas, num_subcarriers = frames[0].num_antennas, frames[0].num_subcarriers
    grid = np.stack([np.asarray(frame.grid) for frame in frames]).astype(np.complex64)  # (F, A, S)
    timestamps_s = np.asarray([float(frame.timestamp) for frame in frames], dtype=np.float64)
    node_ids = np.asarray([int(frame.node_id) for frame in frames], dtype=np.int32)
    recording_dir = Path(out_dir)
    recording_dir.mkdir(parents=True, exist_ok=True)
    np.save(recording_dir / "grid.npy", grid)
    np.save(recording_dir / "t.npy", timestamps_s)
    np.save(recording_dir / "node_id.npy", node_ids)
    meta = {"num_frames": len(frames), "num_antennas": int(num_antennas),
            "num_subcarriers": int(num_subcarriers)}
    if tone_layout is not None:
        meta["tone_layout"] = tone_layout.name
    with open(recording_dir / "meta.json", "w") as meta_file:
        json.dump(meta, meta_file, indent=2)
    return recording_dir


def load_tone_layout(recording_dir) -> ToneLayout:
    """The recording's ToneLayout from meta.json, or inferred from its width (with a warning) when
    the recording predates the key. Raises ValueError on an unknown name or width."""
    recording_dir = Path(recording_dir)
    with open(recording_dir / "meta.json") as meta_file:
        meta = json.load(meta_file)
    if "tone_layout" in meta:
        return tone_layout_by_name(meta["tone_layout"])
    tone_layout = infer_tone_layout(meta["num_subcarriers"])
    logging.getLogger(__name__).warning("%s has no tone_layout in meta.json; inferred %s from its %d columns",
                                        recording_dir, tone_layout.name, meta["num_subcarriers"])
    return tone_layout


def load_recording(recording_dir):
    recording_dir = Path(recording_dir)
    grid = np.load(recording_dir / "grid.npy")          # (F, A, S) complex64
    timestamps_s = np.load(recording_dir / "t.npy")
    node_ids = np.load(recording_dir / "node_id.npy")
    num_frames, num_antennas, num_subcarriers = grid.shape
    for i in range(num_frames):
        frame = CsiFrame(num_antennas, num_subcarriers)
        frame.timestamp = float(timestamps_s[i])
        frame.node_id = int(node_ids[i])
        frame.grid[:, :] = grid[i]             # zero-copy write into the native buffer
        yield frame
