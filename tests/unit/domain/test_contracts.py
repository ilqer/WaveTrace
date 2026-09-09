"""Unit tests for wavetrace.domain.contracts.PipelineContract — no I/O, milliseconds (§1.5 tests/unit/)."""

import pytest

from wavetrace.domain.contracts import (
    DEFAULT_HOP_FRAMES,
    DEFAULT_TARGET_SAMPLE_RATE_HZ,
    DEFAULT_WINDOW_FRAMES,
    PipelineContract,
    derive_pipeline_contract,
)


def _valid_kwargs(**overrides):
    kwargs = dict(target_sample_rate_hz=100.0, window_frames=128, hop_frames=32, subcarrier_width=12)
    kwargs.update(overrides)
    return kwargs


def test_subcarrier_width_defaults_to_none():
    contract = PipelineContract(target_sample_rate_hz=100.0, window_frames=128, hop_frames=32)
    assert contract.subcarrier_width is None


def test_constructs_with_todays_default_values():
    contract = PipelineContract(**_valid_kwargs())
    assert contract.target_sample_rate_hz == DEFAULT_TARGET_SAMPLE_RATE_HZ
    assert contract.window_frames == DEFAULT_WINDOW_FRAMES
    assert contract.hop_frames == DEFAULT_HOP_FRAMES


def test_rejects_non_positive_target_sample_rate_hz():
    with pytest.raises(ValueError, match="target_sample_rate_hz"):
        PipelineContract(**_valid_kwargs(target_sample_rate_hz=0.0))


def test_rejects_non_positive_window_frames():
    with pytest.raises(ValueError, match="window_frames"):
        PipelineContract(**_valid_kwargs(window_frames=0))


def test_rejects_non_positive_hop_frames():
    with pytest.raises(ValueError, match="hop_frames"):
        PipelineContract(**_valid_kwargs(hop_frames=0))


def test_rejects_hop_frames_exceeding_window_frames():
    with pytest.raises(ValueError, match="hop_frames"):
        PipelineContract(**_valid_kwargs(window_frames=32, hop_frames=64))


def test_accepts_hop_frames_equal_to_window_frames():
    contract = PipelineContract(**_valid_kwargs(window_frames=32, hop_frames=32))
    assert contract.hop_frames == contract.window_frames


def test_rejects_non_positive_subcarrier_width():
    with pytest.raises(ValueError, match="subcarrier_width"):
        PipelineContract(**_valid_kwargs(subcarrier_width=0))


def test_accepts_none_subcarrier_width():
    """None means "no capture width recorded" -- must stay valid, never coerced to a guess."""
    contract = PipelineContract(**_valid_kwargs(subcarrier_width=None))
    assert contract.subcarrier_width is None


def test_is_frozen():
    contract = PipelineContract(**_valid_kwargs())
    with pytest.raises(AttributeError):
        contract.window_frames = 64


def test_to_dict_from_dict_round_trip():
    contract = PipelineContract(**_valid_kwargs(subcarrier_width=6))
    assert PipelineContract.from_dict(contract.to_dict()) == contract


def test_to_dict_from_dict_round_trip_with_none_subcarrier_width():
    """A contract with no recorded capture width must round-trip as None, not get coerced or
    dropped -- this is the exact shape `PresenceHead`/`WeaponHead` persist to joblib."""
    contract = PipelineContract(**_valid_kwargs(subcarrier_width=None))
    assert PipelineContract.from_dict(contract.to_dict()) == contract
    assert PipelineContract.from_dict(contract.to_dict()).subcarrier_width is None


class _FakeModelConfig:
    """Duck-typed stand-in for wavetrace.Config.ModelConfig (only .window/.hop read;
    derive_pipeline_contract does not read .k -- see PipelineContract.subcarrier_width)."""

    def __init__(self, window, hop, k):
        self.window = window
        self.hop = hop
        self.k = k


def test_derive_pipeline_contract_reads_window_hop_from_config():
    config = _FakeModelConfig(window=128, hop=32, k=8)
    contract = derive_pipeline_contract(config)
    assert contract == PipelineContract(
        target_sample_rate_hz=DEFAULT_TARGET_SAMPLE_RATE_HZ,
        window_frames=128,
        hop_frames=32,
    )
    assert contract.subcarrier_width is None


def test_derive_pipeline_contract_accepts_explicit_sample_rate():
    config = _FakeModelConfig(window=64, hop=16, k=4)
    contract = derive_pipeline_contract(config, target_sample_rate_hz=50.0)
    assert contract.target_sample_rate_hz == 50.0
