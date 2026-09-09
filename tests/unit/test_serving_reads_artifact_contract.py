"""`run_live_mesh.py`/`run_weapon.py`/`run_count.py`/`web/streamer.py` resample each link at
`m["session"].head.contract.target_sample_rate_hz` rather than a re-declared
`DEFAULT_TARGET_SAMPLE_RATE_HZ` constant, so a future change to the resample rate can't drift
between training and serving. All four call sites go through `modeSession` (Infer.py) to build the
`session` in their `m`/entry dict, so this pins the one fact that makes the swap
behaviour-identical today: nothing in the codebase overrides `target_sample_rate_hz` (confirmed by
`grep -rn "target_sample_rate_hz" --include='*.py' .` returning no call site outside
`contracts.py`/tests), so every existing artifact's contract carries exactly the number the
constant used to hardcode."""

import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.domain.contracts import DEFAULT_TARGET_SAMPLE_RATE_HZ
from wavetrace.recognition import modeSession
from wavetrace.recognition.Model import PresenceHead
from wavetrace.recognition.Weapon import WeaponHead


def test_presence_session_contract_rate_matches_old_hardcoded_default(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 18)).astype(np.float32)
    y = (np.arange(40) % 2).astype(np.int64)
    path = PresenceHead(ModelConfig(stage="presence", k=2)).fit(X, y).save(tmp_path / "model.joblib")

    session = modeSession("presence", str(path))
    assert session.head.contract.target_sample_rate_hz == DEFAULT_TARGET_SAMPLE_RATE_HZ == 100.0


def test_weapon_session_contract_rate_matches_old_hardcoded_default(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 27)).astype(np.float32)
    X[::2, 9] -= 3.0
    y = (np.arange(40) % 2).astype(np.int64)
    path = WeaponHead(ModelConfig(stage="weapon", k=12, backend="variance")).fit(X, y).save(
        tmp_path / "weapon.joblib")

    session = modeSession("weapon", str(path))
    assert session.head.contract.target_sample_rate_hz == DEFAULT_TARGET_SAMPLE_RATE_HZ == 100.0
