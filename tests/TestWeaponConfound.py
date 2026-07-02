"""Item 13: Carry-position confound axis in the weapon LOGO report (CAUSE 5E)."""
import numpy as np

from wavetrace.recognition.Train import _carry_groups, _logo_metrics
from wavetrace.recognition.Weapon import WeaponHead
from wavetrace.Config import ModelConfig


def test_carry_groups_parses_weapon_session_ids():
    sess = np.array(["p0_chest_s0", "p0_chest_s1", "p1_waist_s0"])
    np.testing.assert_array_equal(_carry_groups(sess), ["chest", "chest", "waist"])


def test_carry_groups_none_for_non_weapon_ids():
    # presence-style ids without the <subject>_<carry>_s<n> shape get no carry axis
    assert _carry_groups(np.array(["sessionA", "sessionB"])) is None
    assert _carry_groups(np.array(["p0_chest_x0", "p0_chest_x1"])) is None  # not s<digit>


    """With >=2 carry positions, emits a 'carry' fold."""
    rng = np.random.default_rng(0)
    n = 40
    # 27-feature IC block (variance backend keys on column 9 = the σ²-series mean); make it separable
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(0, 0.1, (n, 27)).astype(np.float32)
    X[:, 9] += y * 5.0  # class signal in the variance-feature column
    carries = np.where(np.arange(n) < n // 2, "chest", "waist")
    subj = np.where(np.arange(n) % 4 < 2, "p0", "p1")
    sess = np.array([f"{s}_{c}_s{i % 2}" for s, c, i in zip(subj, carries, range(n))])

    cfg = ModelConfig(stage="weapon", k=12, backend="variance")
    out = _logo_metrics(X, y, sess, subj, lambda: WeaponHead(cfg))
    assert "carry" in out
    assert set(out["carry"]) >= {"accuracy", "majority_accuracy"}
