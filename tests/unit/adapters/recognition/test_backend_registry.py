"""`RecognitionBackend` declares non-method members (`feature_mode`, `classes_`), so
`issubclass`/`runtime_checkable` either raises or does not check signatures against it; a plain
`hasattr` sweep is the honest mechanical check. `__protocol_attrs__` is a CPython/`typing`
implementation detail, not documented API."""

import pytest

from wavetrace.adapters.recognition import get_backend_class, registered_backend_names
from wavetrace.application.ports import RecognitionBackend


def test_every_registered_backend_satisfies_the_protocol():
    if not hasattr(RecognitionBackend, "__protocol_attrs__"):
        pytest.fail(
            "RecognitionBackend.__protocol_attrs__ is gone on this interpreter; this test derives "
            "the Protocol's member list from it and must be rewritten against whatever replaced "
            "it, not silently skipped"
        )
    protocol_members = RecognitionBackend.__protocol_attrs__
    for name in registered_backend_names():
        backend_cls = get_backend_class(name)
        for member in protocol_members:
            assert hasattr(backend_cls, member), (
                f"backend {name!r} ({backend_cls.__name__}) is missing "
                f"RecognitionBackend member {member!r}"
            )
