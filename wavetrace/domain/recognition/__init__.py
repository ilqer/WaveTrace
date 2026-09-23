"""The recognition heads: pure classifiers over a window of features, no I/O and no backend lookup."""

from wavetrace.domain.recognition.presence_head import PresenceHead
from wavetrace.domain.recognition.weapon_head import VARIANCE_FEATURE, WeaponHead

__all__ = ["PresenceHead", "WeaponHead", "VARIANCE_FEATURE"]
