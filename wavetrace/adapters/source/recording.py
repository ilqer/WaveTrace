"""Saved CSI recordings.

A recording is grid.npy (F,A,S) complex64 + t.npy (F,) + node_id.npy (F,) + meta.json, and costs
O(F·A·S) to write or read.
"""

import json
from pathlib import Path

import numpy as np

from wavetrace import CsiFrame
from wavetrace.application.ports import CsiSource


class RecordingSource(CsiSource):
    """Replay frames saved by `save_recording`. Reconstructs each CsiFrame on demand."""

    def __init__(self, recording_dir):
        self._dir = Path(recording_dir)

    def frames(self):
        return load_recording(self._dir)


def save_recording(frames, out_dir) -> Path:
    """Serialize a CsiFrame list to out_dir (grid/t/node_id .npy + meta.json). O(F·A·S)."""
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
    with open(recording_dir / "meta.json", "w") as meta_file:
        json.dump({"num_frames": len(frames), "num_antennas": int(num_antennas),
                   "num_subcarriers": int(num_subcarriers)}, meta_file, indent=2)
    return recording_dir


def load_recording(recording_dir):
    """Yield reconstructed CsiFrames from a saved recording. O(F·A·S)."""
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
