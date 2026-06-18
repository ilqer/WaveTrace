"""T6/P10 — LinkVoter probability fusion and accuracy_weights."""

import numpy as np
import pytest

from wavetrace.recognition.Link import LinkVoter, accuracy_weights, evaluate_link_fusion


def test_accuracy_weights_maps_correctly():
    """Chance (0.5) → 0, perfect (1.0) → 1.0, below chance → 0 (clamped)."""
    w = accuracy_weights({"a": 1.0, "b": 0.5, "c": 0.3, "d": 0.75})
    assert w["a"] == pytest.approx(1.0)   # (1.0 - 0.5) * 2
    assert w["b"] == pytest.approx(0.0)   # (0.5 - 0.5) * 2 = 0
    assert w["c"] == pytest.approx(0.0)   # max(0.3 - 0.5, 0) * 2 = 0
    assert w["d"] == pytest.approx(0.5)   # (0.75 - 0.5) * 2 = 0.5


def test_single_link_returns_input_proba():
    """With one link and default weight=1, blend equals the input proba."""
    voter = LinkVoter()
    p = np.array([0.3, 0.7], dtype=np.float32)
    voter.add(0, p)
    cls, blended = voter.finalize()
    assert cls == 1
    assert np.allclose(blended, p, atol=1e-6)


def test_two_links_equal_weights_average():
    """Two links with equal weight and quality → simple average of probas."""
    voter = LinkVoter()
    voter.add(0, np.array([0.8, 0.2]))
    voter.add(1, np.array([0.2, 0.8]))
    cls, blended = voter.finalize()
    assert np.allclose(blended, [0.5, 0.5], atol=1e-6)
    assert cls == 0  # argmax of [0.5, 0.5] → first index


def test_static_weights_shift_vote():
    """Static weight 3:1 in favour of node 1 shifts blend toward node 1's proba."""
    voter = LinkVoter({0: 1.0, 1: 3.0})
    p0 = np.array([0.9, 0.1])
    p1 = np.array([0.1, 0.9])
    voter.add(0, p0)
    voter.add(1, p1)
    cls, blended = voter.finalize()
    expected = (1.0 * p0 + 3.0 * p1) / 4.0
    assert np.allclose(blended, expected, atol=1e-5)
    assert cls == int(np.argmax(expected))


def test_quality_floor_applied():
    """Quality below floor is clamped to quality_floor; equal floors → equal blend."""
    voter = LinkVoter(quality_floor=0.1)
    p = np.array([0.0, 1.0], dtype=np.float64)
    voter.add(0, p, quality=0.001)  # below floor → clamped to 0.1
    voter.add(1, p, quality=0.1)    # exactly at floor
    cls, blended = voter.finalize()
    # Both have effective quality 0.1 → equal weights → blended == p
    assert np.allclose(blended, p, atol=1e-6)


def test_c_mismatch_raises():
    """Proba vectors of different sizes on subsequent add() raise ValueError."""
    voter = LinkVoter()
    voter.add(0, np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="C mismatch"):
        voter.add(1, np.array([0.33, 0.33, 0.34]))


def test_finalize_without_add_raises():
    """finalize() with no prior add() raises ValueError."""
    with pytest.raises(ValueError):
        LinkVoter().finalize()


def test_link_fusion_beats_best_single_link():
    """#6: two complementary above-chance links (2.4 GHz mesh vs 5 GHz Pi) fuse to beat either alone.
    Each link misses a different window; the accuracy-weighted blend recovers both."""
    y = np.array([0, 0, 1, 1])
    # link A: confident-correct except window 3; link B: confident-correct except window 0
    pa = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.6, 0.4]])  # wrong on 3 -> acc 0.75
    pb = np.array([[0.4, 0.6], [0.9, 0.1], [0.3, 0.7], [0.1, 0.9]])  # wrong on 0 -> acc 0.75
    rep = evaluate_link_fusion({24: (pa, 0.75), 5: (pb, 0.75)}, y)
    assert rep["per_link_accuracy"] == {24: pytest.approx(0.75), 5: pytest.approx(0.75)}
    assert rep["fused_accuracy"] == pytest.approx(1.0)  # complementary errors cancel
    assert rep["n"] == 4 and rep["weights"][24] == pytest.approx(0.5)


def test_link_fusion_uniform_when_all_at_chance():
    """All links at chance -> weights 0 -> falls back to a uniform blend instead of raising."""
    y = np.array([0, 1])
    p = np.array([[0.7, 0.3], [0.4, 0.6]])
    rep = evaluate_link_fusion({0: (p, 0.5), 1: (p, 0.5)}, y)
    assert rep["weights"] == {0: 0.0, 1: 0.0}
    assert rep["fused_accuracy"] == pytest.approx(1.0)  # uniform blend of identical probas = p


def test_reusable_after_finalize():
    """finalize() resets all state; subsequent add/finalize is independent."""
    voter = LinkVoter()
    voter.add(0, np.array([0.4, 0.6]))
    cls1, p1 = voter.finalize()

    voter.add(0, np.array([0.7, 0.3]))
    cls2, p2 = voter.finalize()

    assert cls1 == 1 and np.allclose(p1, [0.4, 0.6], atol=1e-6)
    assert cls2 == 0 and np.allclose(p2, [0.7, 0.3], atol=1e-6)
