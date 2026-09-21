"""`run_live_mesh.py`/`run_weapon.py`/`run_count.py`/`web/streamer.py` resample each link at
`m["session"].head.contract.target_sample_rate_hz`, built via `modeSession` (Infer.py). Nothing in
the codebase overrides `target_sample_rate_hz` (confirmed by `grep -rn "target_sample_rate_hz"
--include='*.py' .` returning no call site outside `contracts.py`/tests), so every existing
artifact's contract carries `DEFAULT_TARGET_SAMPLE_RATE_HZ`."""

import numpy as np

from wavetrace.Config import ModelConfig
from wavetrace.adapters.recognition.heads import build_presence_head, build_weapon_head
from wavetrace.domain.contracts import DEFAULT_TARGET_SAMPLE_RATE_HZ
from wavetrace.recognition import modeSession


def test_presence_session_contract_rate_matches_old_hardcoded_default(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 18)).astype(np.float32)
    y = (np.arange(40) % 2).astype(np.int64)
    path = build_presence_head(ModelConfig(stage="presence", k=2)).fit(X, y).save(tmp_path / "model.joblib")

    session = modeSession("presence", str(path))
    assert session.head.contract.target_sample_rate_hz == DEFAULT_TARGET_SAMPLE_RATE_HZ == 100.0


def test_weapon_session_contract_rate_matches_old_hardcoded_default(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 27)).astype(np.float32)
    X[::2, 9] -= 3.0
    y = (np.arange(40) % 2).astype(np.int64)
    path = build_weapon_head(ModelConfig(stage="weapon", k=12, backend="variance")).fit(X, y).save(
        tmp_path / "weapon.joblib")

    session = modeSession("weapon", str(path))
    assert session.head.contract.target_sample_rate_hz == DEFAULT_TARGET_SAMPLE_RATE_HZ == 100.0
