import os
from pathlib import Path

import numpy as np
import pytest
import torch
from unitrack.benchmarks.hota.models import build_model, segments_to_prediction


def test_segments_to_prediction_keeps_things_only():
    seg = torch.zeros((4, 4), dtype=torch.int64)
    seg[0:2, :] = 1  # thing (label 11)
    seg[2:4, :] = 2  # stuff (label 8) -> dropped
    info = [
        {"id": 1, "label_id": 11, "score": 0.9},
        {"id": 2, "label_id": 8, "score": 0.8},
    ]
    pred = segments_to_prediction(seg, info, thing_ids={11, 12, 13, 14, 15, 16, 17, 18})
    assert pred.num_instances == 1
    assert pred.categories.tolist() == [11]
    assert pred.masks[0][0, 0].item() is True
    assert pred.masks[0][3, 0].item() is False


def test_build_model_unknown_key_raises():
    with pytest.raises(KeyError):
        build_model("does-not-exist", thing_ids=range(11, 19))


def _make_outputs(cls, masks, emb):
    """Stand-in Mask2Former output exposing the three tensors the recipe reads."""
    from unittest.mock import MagicMock

    out = MagicMock()
    out.class_queries_logits = cls
    out.masks_queries_logits = masks
    out.transformer_decoder_last_hidden_state = emb
    return out


def test_adapter_call_extracts_embeddings_and_centroids():
    """The adapter's __call__ pipes processor -> model -> extract_segments ->
    thing instances carrying per-instance embeddings + centroids, without any
    network (processor/model are mocked)."""
    pytest.importorskip("transformers")
    from unittest.mock import MagicMock

    dim = 256
    num_labels = 19  # cityscapes
    num_queries = 1
    h, w = 4, 4

    # One query that wins label 11 (a thing) with high score over the whole frame.
    cls = torch.full((1, num_queries, num_labels + 1), -10.0)
    cls[0, 0, 11] = 10.0  # label 11 dominates softmax
    masks = torch.full((1, num_queries, h, w), 10.0)  # sigmoid -> ~1 everywhere
    emb = torch.arange(dim, dtype=torch.float32).view(1, num_queries, dim)

    out = _make_outputs(cls, masks, emb)

    adapter = build_model("mask2former-tiny", thing_ids=range(11, 19))
    proc = MagicMock()
    encoded = MagicMock()
    encoded.to.return_value = {}  # so model(**inputs) -> model()
    proc.return_value = encoded
    adapter._processor = proc
    adapter._model = MagicMock(return_value=out)
    adapter._device = torch.device("cpu")

    pred = adapter(np.zeros((h, w, 3), dtype=np.uint8))

    assert pred.num_instances == 1
    assert pred.categories.tolist() == [11]
    assert pred.embeddings is not None
    assert pred.embeddings.shape == (1, dim)
    # The instance embedding is the source query's decoder hidden state.
    assert torch.equal(pred.embeddings[0], emb[0, 0])
    assert pred.centroids is not None
    assert pred.centroids.shape == (1, 2)
    # Full-frame mask -> centroid at the image center, ordered [x, y].
    cx, cy = pred.centroids[0].tolist()
    assert cx == pytest.approx(1.5)
    assert cy == pytest.approx(1.5)


def test_extract_segments_with_queries_records_source_queries():
    """The reimplemented extraction returns segmentation/segments and records each
    segment's source query index."""
    pytest.importorskip("transformers")
    from unitrack.benchmarks.hota.models import extract_segments_with_queries

    dim = 8
    num_labels = 19
    num_queries = 2
    h, w = 6, 6

    cls = torch.full((1, num_queries, num_labels + 1), -10.0)
    cls[0, 0, 11] = 10.0
    cls[0, 1, 13] = 10.0
    masks = torch.full((1, num_queries, h, w), -10.0)
    masks[0, 0, :3, :] = 10.0  # query 0 -> top half
    masks[0, 1, 3:, :] = 10.0  # query 1 -> bottom half
    # Distinguishable per-query embedding rows so a shifted/permuted index is
    # caught: query 0 -> all ones, query 1 -> all twos.
    emb = torch.zeros(1, num_queries, dim)
    emb[0, 0] = 1.0
    emb[0, 1] = 2.0

    from unittest.mock import MagicMock

    out = MagicMock()
    out.class_queries_logits = cls
    out.masks_queries_logits = masks
    out.transformer_decoder_last_hidden_state = emb

    seg, segs, emb_keep = extract_segments_with_queries(out, (h, w))
    assert seg.dtype == torch.int32
    assert seg.shape == (h, w)
    assert len(segs) == 2
    labels = sorted(s["label_id"] for s in segs)
    assert labels == [11, 13]
    # The exact segment -> source-query mapping (label 11 from query 0, label 13
    # from query 1), not merely that some valid index is recorded.
    assert {s["label_id"]: s["q"] for s in segs} == {11: 0, 13: 1}
    # And the recovered per-instance embedding equals the correct query's row, so
    # a shifted/permuted emb index would be caught.
    by_label = {s["label_id"]: emb_keep[s["q"]] for s in segs}
    assert torch.equal(by_label[11], torch.ones(dim))
    assert torch.equal(by_label[13], 2.0 * torch.ones(dim))


@pytest.mark.skipif(
    os.environ.get("UNITRACK_BENCHMARK_LIVE") != "1",
    reason="set UNITRACK_BENCHMARK_LIVE=1 to run the live HF model test",
)
def test_live_mask2former_tiny_runs():
    pytest.importorskip("transformers")

    m = build_model("mask2former-tiny", thing_ids=range(11, 19))
    m.load(torch.device("cpu"))
    pred = m(np.zeros((256, 512, 3), dtype=np.uint8))
    assert pred.masks.ndim == 3


_REAL = Path.home() / "Datasets/cityscapes-dvps/cityscapes-dvps.val.lmdb"


@pytest.mark.skipif(
    os.environ.get("UNITRACK_BENCHMARK_LIVE") != "1" or not _REAL.exists(),
    reason="set UNITRACK_BENCHMARK_LIVE=1 and provide cityscapes-dvps to run parity",
)
def test_live_embedding_extraction_parity():
    """On a real frame, the reimplemented extraction is bit-identical to stock
    post_process_panoptic_segmentation, and same-class instances get distinct
    embeddings."""
    pytest.importorskip("transformers")
    pytest.importorskip("lmdb")
    from collections import defaultdict

    from PIL import Image
    from unitrack.benchmarks.hota.datasets import CityscapesDVPSDataset
    from unitrack.benchmarks.hota.models import extract_segments_with_queries

    ds = CityscapesDVPSDataset(_REAL, limit_seqs=1, max_frames=1)
    (seq,) = list(ds.sequences())
    image, _gt = next(seq.frames)
    h, w = image.shape[:2]

    m = build_model("mask2former-tiny", thing_ids=ds.thing_ids)
    m.load(torch.device("cpu"))

    pil = Image.fromarray(image)
    inputs = m._processor(images=pil, return_tensors="pt").to(m._device)
    with torch.inference_mode():
        outputs = m._model(**inputs)

    stock = m._processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[(h, w)]
    )[0]
    seg, segs, emb_keep = extract_segments_with_queries(outputs, (h, w))

    assert torch.equal(seg.to(torch.int64), stock["segmentation"].to(torch.int64))
    stock_pairs = sorted(
        (int(s["id"]), int(s["label_id"])) for s in stock["segments_info"]
    )
    ours_pairs = sorted((int(s["id"]), int(s["label_id"])) for s in segs)
    assert ours_pairs == stock_pairs

    # At least two same-class instances should have distinct embeddings.
    by_label = defaultdict(list)
    for s in segs:
        by_label[s["label_id"]].append(emb_keep[s["q"]])
    found_distinct = False
    for vecs in by_label.values():
        if len(vecs) >= 2 and not torch.equal(vecs[0], vecs[1]):
            found_distinct = True
            break
    assert found_distinct
