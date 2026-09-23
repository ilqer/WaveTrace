"""Training, evaluation and serving around the recognition heads.

'presence' and 'weapon' are independent modes with no gate between them."""

from wavetrace.domain.recognition import PresenceHead, WeaponHead
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
from wavetrace.recognition.Resample import acceptFormat, fsOk, resampleUniform
from wavetrace.recognition.Train import concatArrays, concatDatasets, trainPresence, trainWeapon
from wavetrace.recognition.Vote import SegmentVoter
from wavetrace.recognition.WeaponServing import dwellProbaDetailed, linkHealth, loadWeaponLinks

__all__ = [
    "countName",
    "PresenceHead",
    "WeaponHead",
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
