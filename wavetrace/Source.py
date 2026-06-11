"""Phase 8 — CSI sources for the CLI.

A `CsiSource` yields CsiFrames; the rest of the pipeline (front-end → recognition → output) is
source-agnostic. Two sources work today:
  * SyntheticSource — wraps an in-memory frame list (the fixtures generate it for dev/CI).
  * RecordingSource — replays frames saved by `save_recording` (the `capture` CLI mode).
Live capture (SerialReader, reading CSI off the dedicated ESP link) is a documented SEAM — it lands
with Phase-0 firmware. Until then "real or recorded CSI" (the DoD) means recorded/synthetic.

Recording format under out_dir (mirrors save_dataset): grid.npy (F,A,S) complex64 + t.npy (F,) +
node_id.npy (F,) + meta.json. O(F·A·S) to (de)serialize.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path

import numpy as np

from wavetrace import CsiFrame


class CsiSource(ABC):
    """A stream of CsiFrames feeding the front-end."""

    @abstractmethod
    def frames(self):
        """Yield CsiFrame objects in capture order."""


class SyntheticSource(CsiSource):
    """Replay an in-memory frame list (e.g. from fixtures.SyntheticCsi/SyntheticRecording)."""

    def __init__(self, frames):
        self._frames = list(frames)

    def frames(self):
        return iter(self._frames)


class RecordingSource(CsiSource):
    """Replay frames saved by `save_recording`. Reconstructs each CsiFrame on demand."""

    def __init__(self, rec_dir):
        self._dir = Path(rec_dir)

    def frames(self):
        return load_recording(self._dir)


def save_recording(frames, out_dir) -> Path:
    """Serialize a CsiFrame list to out_dir (grid/t/node_id .npy + meta.json). O(F·A·S)."""
    frames = list(frames)
    if not frames:
        raise ValueError("save_recording: no frames")
    A, S = frames[0].num_antennas, frames[0].num_subcarriers
    grid = np.stack([np.asarray(fr.grid) for fr in frames]).astype(np.complex64)  # (F, A, S)
    t = np.asarray([float(fr.timestamp) for fr in frames], dtype=np.float64)
    node = np.asarray([int(fr.node_id) for fr in frames], dtype=np.int32)
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "grid.npy", grid)
    np.save(p / "t.npy", t)
    np.save(p / "node_id.npy", node)
    with open(p / "meta.json", "w") as f:
        json.dump({"num_frames": len(frames), "num_antennas": int(A), "num_subcarriers": int(S)}, f,
                  indent=2)
    return p


def load_recording(rec_dir):
    """Yield reconstructed CsiFrames from a saved recording. O(F·A·S)."""
    p = Path(rec_dir)
    grid = np.load(p / "grid.npy")          # (F, A, S) complex64
    t = np.load(p / "t.npy")
    node = np.load(p / "node_id.npy")
    F, A, S = grid.shape
    for i in range(F):
        fr = CsiFrame(A, S)
        fr.timestamp = float(t[i])
        fr.node_id = int(node[i])
        fr.grid[:, :] = grid[i]             # zero-copy write into the native buffer
        yield fr
