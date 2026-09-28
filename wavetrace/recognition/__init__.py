"""Training, evaluation and serving around the recognition heads.

'presence' and 'weapon' are independent modes with no gate between them."""

from wavetrace.domain.recognition import PresenceHead, WeaponHead
from wavetrace.recognition.count import count_name
from wavetrace.recognition.evaluate import (
    binary_rates,
    evaluate_concealment_gap,
    evaluate_presence,
    evaluate_weapon,
    leave_one_group_out,
    segmenter_baseline,
    tier_verdict,
)
from wavetrace.recognition.fusion import fuse
from wavetrace.recognition.link import LinkVoter, accuracy_weights, evaluate_link_fusion
from wavetrace.recognition.infer import InferenceSession, measure_latency, open_session, plan_input
from wavetrace.recognition.train import concat_arrays, concat_datasets, train_presence, train_weapon
from wavetrace.recognition.vote import SegmentVoter
from wavetrace.recognition.weapon_serving import (dwell_proba, find_weapon_entry, link_health, load_weapon_links,
                                                  make_link_tag, parse_link_tag)

__all__ = [
    "count_name",
    "PresenceHead",
    "WeaponHead",
    "train_presence",
    "train_weapon",
    "concat_datasets",
    "concat_arrays",
    "leave_one_group_out",
    "segmenter_baseline",
    "evaluate_presence",
    "evaluate_weapon",
    "evaluate_concealment_gap",
    "binary_rates",
    "tier_verdict",
    "InferenceSession",
    "measure_latency",
    "open_session",
    "plan_input",
    "SegmentVoter",
    "fuse",
    "LinkVoter",
    "accuracy_weights",
    "evaluate_link_fusion",
    "dwell_proba",
    "load_weapon_links",
    "find_weapon_entry",
    "make_link_tag",
    "parse_link_tag",
    "link_health",
]
