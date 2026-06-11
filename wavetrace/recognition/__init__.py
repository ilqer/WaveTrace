"""Phases 6–7 — recognition heads + plumbing: presence head (P6), weapon head + soft segment voting
(P7). TWO INDEPENDENT OPERATING MODES — 'presence' and 'weapon' (`mode_session`), no cross-gating
(user decision 2026-06-11). Training OFFLINE; inference is the real-time path."""

from wavetrace.recognition.Evaluate import (
    binary_rates,
    evaluate_presence,
    evaluate_weapon,
    leave_one_group_out,
    segmenter_baseline,
    tier_verdict,
)
from wavetrace.recognition.Fusion import fuse
from wavetrace.recognition.Infer import InferenceSession, measure_latency, mode_session
from wavetrace.recognition.Model import PresenceHead, sklearn_pipeline
from wavetrace.recognition.Resample import accept_format, fs_ok, resample_uniform
from wavetrace.recognition.Train import concat_arrays, concat_datasets, train_presence, train_weapon
from wavetrace.recognition.Vote import SegmentVoter
from wavetrace.recognition.Weapon import WeaponHead

__all__ = [
    "PresenceHead",
    "WeaponHead",
    "sklearn_pipeline",
    "train_presence",
    "train_weapon",
    "concat_datasets",
    "concat_arrays",
    "leave_one_group_out",
    "segmenter_baseline",
    "evaluate_presence",
    "evaluate_weapon",
    "binary_rates",
    "tier_verdict",
    "InferenceSession",
    "measure_latency",
    "mode_session",
    "SegmentVoter",
    "fuse",
    "resample_uniform",
    "fs_ok",
    "accept_format",
]
