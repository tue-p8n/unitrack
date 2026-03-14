import numpy as np
import pytest

pytest.importorskip("evaluators")

import torch
from unitrack.benchmarks.hota.render import TrackRemap
from unitrack.benchmarks.hota.runner import BenchmarkRunner
from unitrack.benchmarks.hota.tracker import (
    TRACKER_REGISTRY,
    default_tracker_factory,
)

from ._fakes import (
    EmbeddingModel,
    PerfectModel,
    ScoreSplitDataset,
    ScoreSplitModel,
    SyntheticPanopticDataset,
)


def test_perfect_model_scores_near_one():
    runner = BenchmarkRunner(device=torch.device("cpu"))
    dataset = SyntheticPanopticDataset(n_frames=3)
    results = runner.run(models=[PerfectModel()], dataset=dataset)
    (res,) = results
    assert res.model_key == "perfect"
    assert res.metrics["HOTA"] == pytest.approx(1.0, abs=1e-6)
    assert res.num_frames == 3


class _FlakyModel(PerfectModel):
    key = "flaky"

    def __init__(self):
        self._t = 0

    def __call__(self, image):
        import torch
        from unitrack.benchmarks.hota.types import FramePrediction

        self._t += 1
        if self._t == 2:  # drop the instance on frame 2 -> a gap
            h, w, _ = image.shape
            return FramePrediction(
                masks=torch.zeros((0, h, w), dtype=torch.bool),
                categories=torch.zeros((0,), dtype=torch.int64),
                scores=torch.zeros((0,), dtype=torch.float32),
            )
        return super().__call__(image)


def test_fragmented_model_scores_below_one():
    import torch

    res = BenchmarkRunner(device=torch.device("cpu")).run(
        models=[_FlakyModel()], dataset=SyntheticPanopticDataset(n_frames=3)
    )[0]
    assert 0.0 < res.metrics["HOTA"] < 1.0


def test_min_score_suppresses_below_threshold_detection():
    # The below-threshold class-13 instance must never appear in the predicted
    # panoptic, and the kept class-11 instance must hold one stable id.
    runner = BenchmarkRunner(device=torch.device("cpu"), min_score=0.5)
    dataset = ScoreSplitDataset(n_frames=3)
    model = ScoreSplitModel()
    model.load(runner.device)
    offset = dataset.offset
    tracker = runner.tracker_factory(8, 8)
    remap = TrackRemap(offset=offset)
    kept_ids: set[int] = set()
    for frame_idx in range(3):
        pred_panoptic = runner._predict_panoptic(
            model=model,
            image=np.zeros((8, 8, 3), np.uint8),
            tracker=tracker,
            remap=remap,
            offset=offset,
            height=8,
            width=8,
            frame_idx=frame_idx,
        )
        present = {int(v) for v in np.unique(pred_panoptic)}
        # No pixel encodes class 13 (the suppressed instance).
        assert not any(v // offset == 13 for v in present), present
        # Exactly the kept class-11 instance is painted.
        class11 = {v for v in present if v // offset == 11}
        assert len(class11) == 1, class11
        kept_ids |= class11
    # The kept instance kept ONE stable id across all three frames.
    assert kept_ids == {11 * offset + 1}


class _BoomModel:
    """A model that always fails, to exercise per-model sweep resilience."""

    key = "boom"

    def load(self, device):  # noqa: ARG002
        msg = "boom on load"
        raise RuntimeError(msg)

    def __call__(self, image):  # noqa: ARG002
        msg = "boom on call"
        raise RuntimeError(msg)


def test_failing_model_is_skipped_not_fatal():
    # One model blowing up must not lose the rest of the sweep.
    results = BenchmarkRunner(device=torch.device("cpu")).run(
        models=[_BoomModel(), PerfectModel()],
        dataset=SyntheticPanopticDataset(n_frames=2),
    )
    assert [r.model_key for r in results] == ["perfect"]
    assert results[0].metrics["HOTA"] == pytest.approx(1.0, abs=1e-6)


class _CapturingTracker:
    """Wrap a real MultiStream, recording each Detections it is stepped with."""

    def __init__(self, inner, captured: list) -> None:
        self._inner = inner
        self._captured = captured

    def step(self, stream_key, dets, ctx):
        self._captured.append(dets)
        return self._inner.step(stream_key, dets, ctx)


def test_predict_panoptic_carries_embedding_centroid_score():
    captured: list = []
    base = default_tracker_factory(cost_threshold=0.5)

    runner = BenchmarkRunner(device=torch.device("cpu"))
    dataset = SyntheticPanopticDataset(n_frames=1)
    model = EmbeddingModel(embed_dim=8)
    model.load(runner.device)
    tracker = _CapturingTracker(base(8, 8), captured)
    runner._predict_panoptic(
        model=model,
        image=np.zeros((8, 8, 3), np.uint8),
        tracker=tracker,
        remap=TrackRemap(offset=dataset.offset),
        offset=dataset.offset,
        height=8,
        width=8,
        frame_idx=0,
    )
    (dets,) = captured
    assert hasattr(dets, "embedding")
    assert dets.embedding.shape == (1, 8)
    assert torch.equal(dets.embedding[0], torch.tensor([1.0] + [0.0] * 7))
    assert hasattr(dets, "centroid")
    assert dets.centroid.shape == (1, 2)
    assert dets.centroid[0].tolist() == [1.5, 1.5]
    assert hasattr(dets, "score")
    assert dets.score.shape == (1,)
    assert dets.score[0].item() == pytest.approx(0.9)


def test_predict_panoptic_omits_optional_fields_when_absent():
    # The mask-only PerfectModel emits no embedding / centroid; the runner must
    # not invent those fields on the Detections (the mask-IoU path is unchanged).
    captured: list = []
    base = default_tracker_factory(cost_threshold=0.5)

    runner = BenchmarkRunner(device=torch.device("cpu"))
    dataset = SyntheticPanopticDataset(n_frames=1)
    model = PerfectModel()
    model.load(runner.device)
    tracker = _CapturingTracker(base(8, 8), captured)
    runner._predict_panoptic(
        model=model,
        image=np.zeros((8, 8, 3), np.uint8),
        tracker=tracker,
        remap=TrackRemap(offset=dataset.offset),
        offset=dataset.offset,
        height=8,
        width=8,
        frame_idx=0,
    )
    (dets,) = captured
    assert not hasattr(dets, "embedding")
    assert not hasattr(dets, "centroid")
    # score is always carried (cheap, and the cascade tracker reads it).
    assert hasattr(dets, "score")


def test_cosine_tracker_perfect_embeddings_scores_near_one():
    # End-to-end: a model emitting a stable per-instance embedding, tracked by
    # the cosine factory through the runner + HOTA, recovers a perfect score.
    runner = BenchmarkRunner(
        device=torch.device("cpu"), tracker_factory=TRACKER_REGISTRY["cosine"]
    )
    dataset = SyntheticPanopticDataset(n_frames=3)
    res = runner.run(models=[EmbeddingModel()], dataset=dataset)[0]
    assert res.metrics["HOTA"] == pytest.approx(1.0, abs=1e-6)
    assert res.num_frames == 3


def test_run_sweeps_model_by_tracker():
    # A single model swept over two trackers yields one result per (model,
    # tracker) combo, each carrying its tracker_key.
    runner = BenchmarkRunner(device=torch.device("cpu"))
    dataset = SyntheticPanopticDataset(n_frames=3)
    results = runner.run(
        models=[EmbeddingModel()],
        dataset=dataset,
        trackers={
            "maskiou": TRACKER_REGISTRY["maskiou"],
            "cosine": TRACKER_REGISTRY["cosine"],
        },
    )
    assert {r.tracker_key for r in results} == {"maskiou", "cosine"}
    assert {r.model_key for r in results} == {"embedding"}
    for res in results:
        assert res.metrics["HOTA"] == pytest.approx(1.0, abs=1e-6)


def test_run_default_tracker_uses_constructor_factory():
    # Without an explicit ``trackers`` mapping, ``run`` uses the constructor's
    # tracker_factory and labels the single result "maskiou".
    runner = BenchmarkRunner(device=torch.device("cpu"))
    res = runner.run(
        models=[PerfectModel()], dataset=SyntheticPanopticDataset(n_frames=2)
    )[0]
    assert res.tracker_key == "maskiou"


def test_run_failing_combo_is_skipped_not_fatal():
    # One (model, tracker) combo blowing up must not lose the rest of the sweep.
    runner = BenchmarkRunner(device=torch.device("cpu"))

    def boom_factory(height, width):  # noqa: ARG001
        msg = "boom building tracker"
        raise RuntimeError(msg)

    results = runner.run(
        models=[PerfectModel()],
        dataset=SyntheticPanopticDataset(n_frames=2),
        trackers={"boom": boom_factory, "maskiou": TRACKER_REGISTRY["maskiou"]},
    )
    assert [r.tracker_key for r in results] == ["maskiou"]


def test_custom_tracker_factory_is_invoked():
    calls: list[tuple[int, int]] = []
    base = default_tracker_factory(cost_threshold=0.5)

    def factory(height: int, width: int):
        calls.append((height, width))
        return base(height, width)

    runner = BenchmarkRunner(device=torch.device("cpu"), tracker_factory=factory)
    runner.run(models=[PerfectModel()], dataset=SyntheticPanopticDataset(n_frames=2))
    assert calls == [(8, 8)]  # built once per sequence at the frame resolution
