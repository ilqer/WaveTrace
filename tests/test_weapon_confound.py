"""Carry-position confound axis in the weapon LOGO report."""
import numpy as np

from wavetrace.recognition.train import _carry_groups, _logo_metrics
from wavetrace.adapters.recognition.heads import build_weapon_head
from wavetrace.config import ModelConfig


def test_carry_groups_parses_weapon_session_ids():
    session_ids = np.array(["p0_chest_s0", "p0_chest_s1", "p1_waist_s0"])
    np.testing.assert_array_equal(_carry_groups(session_ids), ["chest", "chest", "waist"])


def test_carry_groups_none_for_non_weapon_ids():
    # presence-style ids without the <subject>_<carry>_s<n> shape get no carry axis
    assert _carry_groups(np.array(["sessionA", "sessionB"])) is None
    assert _carry_groups(np.array(["p0_chest_x0", "p0_chest_x1"])) is None  # not s<digit>


def test_logo_metrics_adds_carry_axis_when_separable():
    rng = np.random.default_rng(0)
    sample_count = 40
    # 27-feature IC block (variance backend keys on column 9 = the σ²-series mean); make it separable
    y = np.array([0, 1] * (sample_count // 2))
    X = rng.normal(0, 0.1, (sample_count, 27)).astype(np.float32)
    X[:, 9] += y * 5.0  # class signal in the variance-feature column
    carries = np.where(np.arange(sample_count) < sample_count // 2, "chest", "waist")
    subject_ids = np.where(np.arange(sample_count) % 4 < 2, "p0", "p1")
    session_ids = np.array([f"{subject}_{carry}_s{i % 2}"
                            for subject, carry, i in zip(subject_ids, carries, range(sample_count))])

    config = ModelConfig(stage="weapon", k=12, backend="variance")
    out = _logo_metrics(X, y, session_ids, subject_ids, lambda: build_weapon_head(config))
    assert "carry" in out
    assert set(out["carry"]) >= {"accuracy", "majority_accuracy"}
