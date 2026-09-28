
import numpy as np
import pytest

from wavetrace.synthetic import generate_paired_recording
from wavetrace.calibration import Calibration
from wavetrace.config import ModelConfig
from wavetrace.groundtruth import ScriptedLabeler, build_dataset, presence_label_fn
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
    from wavetrace.adapters.source import build_csi_source

    args = type("Args", (), dict(
        recording=None, synthetic=True, antennas=2, subcarriers=32, fs=100.0, duration=2.0,
        presence="", weapon="0:1", weapon_depth=0.0, seed=0,
    ))()
    with pytest.warns(UserWarning, match="weapon-depth"):
        build_csi_source(args)


def test_build_dataset_accepts_generator():
    frames, _, _ = generate_paired_recording(
        num_antennas=2, num_subcarriers=32, sample_rate_hz=100.0, duration_s=4.0, camera_fps=30.0,
        presence_spans=[(0.0, 4.0)], presence_turbulence_std=0.1, seed=1,
    )
    cal = Calibration(baseline_packets=50)
    for fr in frames[:50]:
        cal.observe(fr)
    result = cal.finalize()
    labeler = ScriptedLabeler([(0.0, 4.0, True)], label_fn=presence_label_fn)
    # Pass a generator: build_dataset must materialize it before fs estimation.
    ds = build_dataset(iter(frames), result, cal.gain_lock, labeler, window=128, hop=32)
    assert ds.meta["fs"] > 0.0 and ds.y.size > 0
