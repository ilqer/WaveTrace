"""Every adapter offers each member of its port in `wavetrace/application/ports.py` with the port's call
shape: parameter names, kinds, order, and which parameters have defaults. Annotations are not compared."""
import inspect
import itertools
from typing import NamedTuple, Protocol

import pytest

from wavetrace.adapters.recognition.cnn_backend import CnnBackend
from wavetrace.adapters.recognition.mlp_backend import MlpBackend
from wavetrace.adapters.recognition.svm_backend import SvmBackend
from wavetrace.adapters.recognition.variance_backend import VarianceBackend
from wavetrace.adapters.source.link_capture import UdpLinkCapture, UdpTimedLinkCapture
from wavetrace.adapters.source.nexmon import NexmonSource
from wavetrace.adapters.source.recording import RecordingSource
from wavetrace.adapters.source.serial_reader import SerialReader
from wavetrace.adapters.source.synthetic import SyntheticSource
from wavetrace.adapters.source.udp import UdpLinkBatchReceiver, UdpSource
from wavetrace.adapters.source.uwb_udp import UdpCirSource
from wavetrace.application import ports
from wavetrace.application.behaviours.drift import DriftMonitoredSource
from wavetrace.application.ports import (
    CameraLabelCapture,
    CirSource,
    CsiSource,
    LinkBatchReceiver,
    LinkCapture,
    RecognitionBackend,
    TimedLinkCapture,
)
from wavetrace.groundtruth.webcam import WebcamLabelCapture

PORT_ADAPTERS = [
    (CsiSource, SerialReader),
    (CsiSource, RecordingSource),
    (CsiSource, SyntheticSource),
    (CsiSource, NexmonSource),
    (CsiSource, UdpSource),
    (CsiSource, DriftMonitoredSource),
    (CirSource, UdpCirSource),
    (LinkBatchReceiver, UdpLinkBatchReceiver),
    (LinkCapture, UdpLinkCapture),
    (TimedLinkCapture, UdpTimedLinkCapture),
    (CameraLabelCapture, WebcamLabelCapture),
    (RecognitionBackend, MlpBackend),
    (RecognitionBackend, SvmBackend),
    (RecognitionBackend, VarianceBackend),
    (RecognitionBackend, CnnBackend),
]

_MISSING = object()


class ParameterShape(NamedTuple):
    name: str
    kind: inspect._ParameterKind
    has_default: bool

    def __str__(self) -> str:
        if self.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return f"{self.name} ({self.kind.description})"
        default = "has default" if self.has_default else "required"
        return f"{self.name!r} ({self.kind.description}, {default})"


def _port_classes() -> set[type]:
    return {value for value in vars(ports).values()
            if inspect.isclass(value) and Protocol in value.__bases__ and value.__module__ == ports.__name__}


def _port_member_names(port: type) -> list[str]:
    names = [name for name in vars(port) if not name.startswith("_")]
    names += [name for name in inspect.get_annotations(port) if name not in names]
    # typing swaps a Protocol's missing __init__ for its own placeholder, so only a declared one is a member
    if getattr(vars(port).get("__init__"), "__qualname__", "") == f"{port.__qualname__}.__init__":
        names.append("__init__")
    return names


def _member_kind(member) -> str:
    if isinstance(member, property):
        return "property"
    if isinstance(member, classmethod):
        return "classmethod"
    if isinstance(member, staticmethod):
        return "staticmethod"
    return "method" if inspect.isfunction(member) else "attribute"


def _parameter_function(member):
    return member.fget if isinstance(member, property) else getattr(member, "__func__", member)


def _parameter_shapes(member) -> list[ParameterShape]:
    shapes = []
    for parameter in inspect.signature(_parameter_function(member)).parameters.values():
        # a *args / **kwargs name can never be passed by a caller, so it is not part of the call shape
        name = {parameter.VAR_POSITIONAL: "*", parameter.VAR_KEYWORD: "**"}.get(parameter.kind, parameter.name)
        shapes.append(ParameterShape(name, parameter.kind, parameter.default is not parameter.empty))
    return shapes


# a port ending in **kwargs accepts any keyword, so an adapter also ending in ** may add keyword-only defaults
def _without_spelled_out_keywords(port_shapes, adapter_shapes) -> list[ParameterShape]:
    var_keyword = inspect.Parameter.VAR_KEYWORD
    if not (port_shapes and adapter_shapes and port_shapes[-1].kind is var_keyword
            and adapter_shapes[-1].kind is var_keyword):
        return adapter_shapes
    port_names = {shape.name for shape in port_shapes}
    return [shape for shape in adapter_shapes
            if not (shape.kind is inspect.Parameter.KEYWORD_ONLY and shape.has_default
                    and shape.name not in port_names)]


def _shape_differences(port_shapes, adapter_shapes) -> list[str]:
    differences = []
    pairs = itertools.zip_longest(port_shapes, _without_spelled_out_keywords(port_shapes, adapter_shapes))
    for position, (port_shape, adapter_shape) in enumerate(pairs):
        if port_shape != adapter_shape:
            differences.append(f"parameter {position}: port has {port_shape or 'nothing'}, "
                               f"adapter has {adapter_shape or 'nothing'}")
    return differences


def _member_problems(port: type, adapter: type, name: str) -> list[str]:
    where = f"{port.__name__} <- {adapter.__name__}.{name}"
    adapter_member = inspect.getattr_static(adapter, name, _MISSING)
    if adapter_member is _MISSING:
        return [f"{where}: missing"]
    port_member = inspect.getattr_static(port, name, _MISSING)
    if port_member is _MISSING:
        declared_type = inspect.get_annotations(port, eval_str=True)[name]
        if isinstance(adapter_member, declared_type):
            return []
        return [f"{where}: is {adapter_member!r}, the port declares {getattr(declared_type, '__name__', declared_type)}"]
    if adapter_member is port_member:
        return [f"{where}: inherits the port's unimplemented stub"]
    if _member_kind(adapter_member) != _member_kind(port_member):
        return [f"{where}: is a {_member_kind(adapter_member)}, the port declares a {_member_kind(port_member)}"]
    differences = _shape_differences(_parameter_shapes(port_member), _parameter_shapes(adapter_member))
    return [f"{where}{inspect.signature(_parameter_function(adapter_member))} vs port "
            f"{inspect.signature(_parameter_function(port_member))}: {difference}" for difference in differences]


def test_every_port_has_an_adapter_case():
    uncovered = _port_classes() - {port for port, _ in PORT_ADAPTERS}
    assert not uncovered, f"ports in {ports.__name__} with no PORT_ADAPTERS case: {sorted(port.__name__ for port in uncovered)}"


@pytest.mark.parametrize(("port", "adapter"), PORT_ADAPTERS, ids=lambda cls: cls.__name__)
def test_adapter_matches_port_call_shape(port, adapter):
    problems = [problem for name in _port_member_names(port) for problem in _member_problems(port, adapter, name)]
    assert not problems, "\n".join(problems)
