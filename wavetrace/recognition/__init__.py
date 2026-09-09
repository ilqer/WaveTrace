"""Phases 6–7 — recognition heads + plumbing: presence head (P6), weapon head + soft segment voting
(P7). TWO INDEPENDENT OPERATING MODES — 'presence' and 'weapon' (`modeSession`), no cross-gating
(user decision 2026-06-11). Training OFFLINE; inference is the real-time path."""

from wavetrace.recognition.Count import countName
from wavetrace.recognition.Evaluate import (
    binaryRates,
    evaluateConcealmentGap,
    evaluatePresence,
    evaluateWeapon,
    leaveOneGroupOut,
    segmenterBaseline,
    tierVerdict,
)
from wavetrace.recognition.Fusion import fuse
from wavetrace.recognition.Link import LinkVoter, accuracyWeights, evaluateLinkFusion
from wavetrace.recognition.Infer import InferenceSession, measureLatency, modeSession, planInferenceInput
from wavetrace.recognition.Model import PresenceHead, sklearnPipeline
from wavetrace.recognition.Resample import acceptFormat, fsOk, resampleUniform
from wavetrace.recognition.Train import concatArrays, concatDatasets, trainPresence, trainWeapon
from wavetrace.recognition.Vote import SegmentVoter
from wavetrace.recognition.Weapon import WeaponHead
from wavetrace.recognition.WeaponServing import dwellProbaDetailed, linkHealth, loadWeaponLinks

__all__ = [
    "countName",
    "PresenceHead",
    "WeaponHead",
    "sklearnPipeline",
    "trainPresence",
    "trainWeapon",
    "concatDatasets",
    "concatArrays",
    "leaveOneGroupOut",
    "segmenterBaseline",
    "evaluatePresence",
    "evaluateWeapon",
    "evaluateConcealmentGap",
    "binaryRates",
    "tierVerdict",
    "InferenceSession",
    "measureLatency",
    "modeSession",
    "planInferenceInput",
    "SegmentVoter",
    "fuse",
    "LinkVoter",
    "accuracyWeights",
    "evaluateLinkFusion",
    "resampleUniform",
    "fsOk",
    "acceptFormat",
    "dwellProbaDetailed",
    "loadWeaponLinks",
    "linkHealth",
]
