import torch
from unitrack.benchmarks.hota.types import FramePrediction


def test_frame_prediction_validates_row_counts():
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    categories = torch.tensor([11, 13], dtype=torch.int64)
    scores = torch.tensor([0.9, 0.5], dtype=torch.float32)
    fp = FramePrediction(masks=masks, categories=categories, scores=scores)
    assert fp.num_instances == 2
    assert fp.height == 4
    assert fp.width == 4


def test_frame_prediction_rejects_mismatched_rows():
    import pytest

    with pytest.raises(ValueError, match="row counts"):
        FramePrediction(
            masks=torch.zeros((2, 4, 4), dtype=torch.bool),
            categories=torch.tensor([11], dtype=torch.int64),
            scores=torch.tensor([0.9, 0.5], dtype=torch.float32),
        )


def test_frame_prediction_accepts_centroids():
    fp = FramePrediction(
        masks=torch.zeros((2, 4, 4), dtype=torch.bool),
        categories=torch.tensor([11, 13], dtype=torch.int64),
        scores=torch.tensor([0.9, 0.5], dtype=torch.float32),
        centroids=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )
    assert fp.centroids is not None
    assert fp.centroids.shape == (2, 2)


def test_frame_prediction_rejects_bad_centroids():
    import pytest

    with pytest.raises(ValueError, match="centroids"):
        FramePrediction(
            masks=torch.zeros((2, 4, 4), dtype=torch.bool),
            categories=torch.tensor([11, 13], dtype=torch.int64),
            scores=torch.tensor([0.9, 0.5], dtype=torch.float32),
            centroids=torch.zeros((1, 2), dtype=torch.float32),
        )
