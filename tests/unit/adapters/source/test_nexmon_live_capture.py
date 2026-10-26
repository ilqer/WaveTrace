import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest

from wavetrace.adapters.source.nexmon import NexmonSource, NexmonSourceOptions


class _FakeTcpdump:
    """Writes one pcap-sized chunk to the capture path it is given, as a running tcpdump would."""

    def __init__(self, argv, **kwargs):
        self.capture_path = argv[argv.index("-w") + 1]
        with open(self.capture_path, "wb") as capture_file:
            capture_file.write(bytes(64))
        self.is_terminated = False
        self.is_reaped = False

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        self.wait()

    def terminate(self):
        self.is_terminated = True

    def wait(self, timeout=None):
        self.is_reaped = True
        return 0


@pytest.fixture
def spawned_tcpdumps(monkeypatch):
    processes = []

    def _spawn(argv, **kwargs):
        processes.append(_FakeTcpdump(argv, **kwargs))
        return processes[-1]

    monkeypatch.setattr(subprocess, "Popen", _spawn)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    return processes


def _live_source():
    return NexmonSource(NexmonSourceOptions(live=True, max_frames=1))


def test_a_decode_error_stops_and_reaps_tcpdump_and_removes_the_capture(spawned_tcpdumps, monkeypatch):
    source = _live_source()

    def _fail_to_decode():
        raise RuntimeError("csiread rejected the capture")

    monkeypatch.setattr(source, "_decode_file", _fail_to_decode)
    with pytest.raises(RuntimeError, match="csiread rejected the capture"):
        next(source.frames())
    [tcpdump] = spawned_tcpdumps
    assert tcpdump.is_terminated and tcpdump.is_reaped
    assert not os.path.exists(tcpdump.capture_path)


def test_a_missing_tcpdump_removes_the_capture_file(monkeypatch):
    created_paths = []
    original_mkstemp = tempfile.mkstemp

    def _recording_mkstemp(*args, **kwargs):
        descriptor, path = original_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return descriptor, path

    def _tcpdump_not_installed(argv, **kwargs):
        raise FileNotFoundError("tcpdump")

    monkeypatch.setattr(tempfile, "mkstemp", _recording_mkstemp)
    monkeypatch.setattr(subprocess, "Popen", _tcpdump_not_installed)
    with pytest.raises(FileNotFoundError):
        next(_live_source().frames())
    [capture_path] = created_paths
    assert not os.path.exists(capture_path)


def _install_fake_csiread(monkeypatch, **reader_fields):
    class _FakeNexmonReader:
        def __init__(self, pcap_path, chip, bw):
            self.csi = np.ones((3, 4), dtype=np.complex64)
            for name, value in reader_fields.items():
                setattr(self, name, value)

        def read(self):
            pass

    monkeypatch.setitem(sys.modules, "csiread", SimpleNamespace(Nexmon=_FakeNexmonReader))


def _decoded_timestamps_s():
    return [frame.timestamp for frame in NexmonSource(NexmonSourceOptions(pcap_path="capture.pcap")).frames()]


def test_timestamps_carry_the_microseconds_of_the_reader(monkeypatch):
    _install_fake_csiread(monkeypatch, sec=np.array([100, 100, 101]), usec=np.array([250_000, 750_000, 5]))
    assert _decoded_timestamps_s() == pytest.approx([100.25, 100.75, 101.000005])


def test_timestamps_are_whole_seconds_when_the_reader_has_no_microseconds(monkeypatch):
    _install_fake_csiread(monkeypatch, sec=np.array([100, 100, 101]))
    assert _decoded_timestamps_s() == [100.0, 100.0, 101.0]
