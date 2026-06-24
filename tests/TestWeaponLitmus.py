"""Unit tests for the static-σ²[p] weapon litmus tool (weapon_litmus.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weapon_litmus import (  # noqa: E402
    sigma2_per_frame, separation, gather_sigma2, _node_of, _verdict,
)


def test_sigma2_matches_numpy_ddof1():
    """σ²[p] == sample variance (ddof=1) of the antenna-collapsed magnitude, per frame."""
    rng = np.random.default_rng(0)
    grid = (rng.standard_normal((5, 2, 16)) + 1j * rng.standard_normal((5, 2, 16))).astype(np.complex64)
    got = sigma2_per_frame(grid)
    mag = np.abs(grid).mean(axis=1)
    assert got.shape == (5,)
    np.testing.assert_allclose(got, mag.var(axis=1, ddof=1), rtol=1e-5)


def test_separation_detects_lower_armed_variance():
    """Metal physics case: weapon σ² lower than clear -> high (folded) AUC, direction flagged ok."""
    rng = np.random.default_rng(1)
    clear = rng.normal(10.0, 1.0, 400)   # high inter-subcarrier variance
    weapon = rng.normal(4.0, 1.0, 400)   # metal flattens it -> lower
    s = separation(clear, weapon)
    assert s["auc"] > 0.95
    assert s["lower_when_armed"] is True
    assert s["cohens_d"] < 0  # armed mean below clear mean


def test_separation_chance_when_identical():
    """Overlapping distributions -> AUC ~0.5 -> NO-SEPARATION verdict (the go/no-go we care about)."""
    rng = np.random.default_rng(2)
    a = rng.normal(5.0, 1.0, 500)
    b = rng.normal(5.0, 1.0, 500)
    s = separation(a, b)
    assert s["auc"] == pytest.approx(0.5, abs=0.06)
    assert "NO SEPARATION" in _verdict(s["auc"])


def test_separation_none_when_one_side_empty():
    assert separation(np.array([1.0, 2.0]), np.array([])) is None


def test_node_of_parses_path():
    assert _node_of(os.path.join("data", "weapon_rec", "s0", "clear", "node3", "link_aa", "grid.npy")) == 3
    assert _node_of(os.path.join("data", "weapon_rec", "s0", "clear", "nodeX", "grid.npy")) is None


def test_gather_reads_both_conditions(tmp_path):
    """gather_sigma2 globs clear/weapon grids, maps them to the right node, and concatenates."""
    rng = np.random.default_rng(3)
    for cond, sd in (("clear", 3.0), ("weapon", 1.0)):
        d = tmp_path / "weapon_rec" / "p0_chest_s0" / cond / "node2" / "link_aabb"
        d.mkdir(parents=True)
        grid = (rng.normal(0, sd, (50, 1, 20)) + 1j * rng.normal(0, sd, (50, 1, 20))).astype(np.complex64)
        np.save(d / "grid.npy", grid)
    data = gather_sigma2(str(tmp_path))
    assert set(data) == {2}
    assert data[2]["clear"].shape == (50,)
    assert data[2]["weapon"].shape == (50,)
    # filtering by node excludes everything else
    assert gather_sigma2(str(tmp_path), node=9) == {}
