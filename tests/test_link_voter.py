import numpy as np
import pytest

from wavetrace.recognition.link import LinkVoter, accuracy_weights, evaluate_link_fusion


def test_accuracy_weights_maps_correctly():
    """Chance (0.5) → 0, perfect (1.0) → 1.0, below chance → 0 (clamped)."""
    weights = accuracy_weights({"a": 1.0, "b": 0.5, "c": 0.3, "d": 0.75})
    assert weights["a"] == pytest.approx(1.0)   # (1.0 - 0.5) * 2
    assert weights["b"] == pytest.approx(0.0)   # (0.5 - 0.5) * 2 = 0
    assert weights["c"] == pytest.approx(0.0)   # max(0.3 - 0.5, 0) * 2 = 0
    assert weights["d"] == pytest.approx(0.5)   # (0.75 - 0.5) * 2 = 0.5


def test_single_link_returns_input_proba():
    voter = LinkVoter()
    probabilities = np.array([0.3, 0.7], dtype=np.float32)
    voter.add(0, probabilities)
    predicted_class, blended = voter.finalize()
    assert predicted_class == 1
    assert np.allclose(blended, probabilities, atol=1e-6)


def test_two_links_equal_weights_average():
    voter = LinkVoter()
    voter.add(0, np.array([0.8, 0.2]))
    voter.add(1, np.array([0.2, 0.8]))
    predicted_class, blended = voter.finalize()
    assert np.allclose(blended, [0.5, 0.5], atol=1e-6)
    assert predicted_class == 0  # argmax of [0.5, 0.5] → first index


def test_static_weights_shift_vote():
    voter = LinkVoter({0: 1.0, 1: 3.0})
    link_0_probabilities = np.array([0.9, 0.1])
    link_1_probabilities = np.array([0.1, 0.9])
    voter.add(0, link_0_probabilities)
    voter.add(1, link_1_probabilities)
    predicted_class, blended = voter.finalize()
    expected = (1.0 * link_0_probabilities + 3.0 * link_1_probabilities) / 4.0
    assert np.allclose(blended, expected, atol=1e-5)
    assert predicted_class == int(np.argmax(expected))


def test_quality_floor_applied():
    """Quality below floor is clamped to quality_floor; equal floors → equal blend."""
    voter = LinkVoter(quality_floor=0.1)
    probabilities = np.array([0.0, 1.0], dtype=np.float64)
    voter.add(0, probabilities, quality=0.001)  # below floor → clamped to 0.1
    voter.add(1, probabilities, quality=0.1)    # exactly at floor
    predicted_class, blended = voter.finalize()
    # Both have effective quality 0.1 → equal weights → blended == probabilities
    assert np.allclose(blended, probabilities, atol=1e-6)


def test_c_mismatch_raises():
    """Proba vectors of different sizes on subsequent add() raise ValueError."""
    voter = LinkVoter()
    voter.add(0, np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="C mismatch"):
        voter.add(1, np.array([0.33, 0.33, 0.34]))


def test_finalize_without_add_raises():
    with pytest.raises(ValueError):
        LinkVoter().finalize()


def test_link_fusion_beats_best_single_link():
    """Fusing two complementary links (2.4 GHz mesh vs 5 GHz Pi) beats either one alone.
    Accuracy-weighted blend recovers when each link misses a different window."""
    y = np.array([0, 0, 1, 1])
    # link A: confident-correct except window 3; link B: confident-correct except window 0
    # wrong on 3 -> acc 0.75
    link_a_probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.6, 0.4]])
    # wrong on 0 -> acc 0.75
    link_b_probabilities = np.array([[0.4, 0.6], [0.9, 0.1], [0.3, 0.7], [0.1, 0.9]])
    report = evaluate_link_fusion({24: (link_a_probabilities, 0.75), 5: (link_b_probabilities, 0.75)}, y)
    assert report.per_link_accuracy == {24: pytest.approx(0.75), 5: pytest.approx(0.75)}
    assert report.fused_accuracy == pytest.approx(1.0)  # complementary errors cancel
    assert report.sample_count == 4 and report.weights[24] == pytest.approx(0.5)


def test_link_fusion_uniform_when_all_at_chance():
    """All links at chance -> weights 0 -> falls back to a uniform blend instead of raising."""
    y = np.array([0, 1])
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    report = evaluate_link_fusion({0: (probabilities, 0.5), 1: (probabilities, 0.5)}, y)
    assert report.weights == {0: 0.0, 1: 0.0}
    assert report.fused_accuracy == pytest.approx(1.0)  # uniform blend of identical probas = probabilities


def test_reusable_after_finalize():
    voter = LinkVoter()
    voter.add(0, np.array([0.4, 0.6]))
    first_class, first_blended = voter.finalize()

    voter.add(0, np.array([0.7, 0.3]))
    second_class, second_blended = voter.finalize()

    assert first_class == 1 and np.allclose(first_blended, [0.4, 0.6], atol=1e-6)
    assert second_class == 0 and np.allclose(second_blended, [0.7, 0.3], atol=1e-6)
