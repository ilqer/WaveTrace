from types import SimpleNamespace

import numpy as np
import pytest

from wavetrace.recognition.heatmap import HeatmapHead, train_heatmap

GRID = 2


def _dataset(masks):
    labels = [SimpleNamespace(mask=mask) for mask in masks]
    X_image = np.arange(len(masks), dtype=np.float32).reshape(len(masks), 1, 1, 1)
    return SimpleNamespace(labels=labels, X_image=X_image, meta={"K": 1})


@pytest.fixture
def fit_calls(monkeypatch):
    calls = []

    def record_fit(head, X, Y, **kwargs):
        calls.append((head, X, Y))
        return head

    monkeypatch.setattr(HeatmapHead, "fit", record_fit)
    return calls


def test_masks_stay_paired_with_their_own_windows_across_a_maskless_gap(fit_calls):
    first_mask, last_mask = [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    _, masked_window_count = train_heatmap(_dataset([first_mask, [], last_mask]))
    head, X, Y = fit_calls[0]
    assert masked_window_count == 2
    assert X[:, 0, 0, 0].tolist() == [0.0, 2.0]
    assert Y.tolist() == [first_mask, last_mask]
    assert head.grid == GRID


def test_dataset_without_masks_raises(fit_calls):
    with pytest.raises(ValueError, match="mask"):
        train_heatmap(_dataset([[], []]))
    assert not fit_calls
