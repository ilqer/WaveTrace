import warnings

import numpy as np
import pytest

from wavetrace.Synthetic import generatePairedRecording
from wavetrace.Calibration import Calibration
from wavetrace.Config import ModelConfig
from wavetrace.groundtruth import ScriptedLabeler, buildDataset, presenceLabelFn
from wavetrace.adapters.recognition.heads import build_presence_head, build_weapon_head


def test_presence_head_rejects_single_class():
    head = build_presence_head(ModelConfig(stage="presence", k=2))
    X = np.random.default_rng(0).standard_normal((20, 18)).astype(np.float32)
    y = np.zeros(20, dtype=np.int64)  # all "absent"
    with pytest.raises(ValueError, match="single class"):
        head.fit(X, y)


def test_weapon_head_rejects_single_class():
    head = build_weapon_head(ModelConfig(stage="weapon", k=12, backend="variance"))
    X = np.random.default_rng(0).standard_normal((20, 27)).astype(np.float32)
    y = np.ones(20, dtype=np.int64)  # all "weapon"
    with pytest.raises(ValueError, match="single class"):
        head.fit(X, y)


def test_synthetic_weapon_zero_depth_warns():
    from wavetrace.Cli import _sourceFromArgs

    args = type("Args", (), dict(
        recording=None, synthetic=True, antennas=2, subcarriers=32, fs=100.0, duration=2.0,
        presence="", weapon="0:1", weapon_depth=0.0, seed=0,
    ))()
    with pytest.warns(UserWarning, match="weapon-depth"):
        _sourceFromArgs(args)


def test_build_dataset_accepts_generator():
    frames, _, _ = generatePairedRecording(
        numAntennas=2, numSubcarriers=32, sampleRateHz=100.0, durationS=4.0, cameraFps=30.0,
        presenceSpans=[(0.0, 4.0)], presenceTurbulenceStd=0.1, seed=1,
    )
    cal = Calibration(baseline_packets=50)
    for fr in frames[:50]:
        cal.observe(fr)
    result = cal.finalize()
    labeler = ScriptedLabeler([(0.0, 4.0, True)], label_fn=presenceLabelFn)
    # Pass a generator: buildDataset must materialize it before fs estimation.
    ds = buildDataset(iter(frames), result, cal.gainLock, labeler, window=128, hop=32)
    assert ds.meta["fs"] > 0.0 and ds.y.size > 0
