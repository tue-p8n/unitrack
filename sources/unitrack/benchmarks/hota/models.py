"""HuggingFace panoptic-segmentation model adapters."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .types import FramePrediction

# Mask2Former decoder hidden size for the Cityscapes Swin family.
EMBED_DIM = 256

# Short registry key -> HF repo id. The Mask2Former Swin family trained on
# Cityscapes panoptic, a clean tiny->large size-scaling sweep that all share one
# AutoImageProcessor + post_process_panoptic_segmentation path. (Architectures
# with a different processor contract, e.g. task-conditioned OneFormer, are a
# documented extension: add a registry entry + an adapter that yields the same
# FramePrediction.)
_MODEL_REPOS = {
    "mask2former-tiny": "facebook/mask2former-swin-tiny-cityscapes-panoptic",
    "mask2former-small": "facebook/mask2former-swin-small-cityscapes-panoptic",
    "mask2former-base": "facebook/mask2former-swin-base-IN21k-cityscapes-panoptic",
    "mask2former-large": "facebook/mask2former-swin-large-cityscapes-panoptic",
}


def segments_to_prediction(
    segmentation: torch.Tensor,
    segments_info: list[dict],
    *,
    thing_ids: Iterable[int],
) -> FramePrediction:
    """Convert HF panoptic output into a thing-only ``FramePrediction``."""
    things = {int(t) for t in thing_ids}
    masks, cats, scores = [], [], []
    for seg in segments_info:
        label = int(seg["label_id"])
        if label not in things:
            continue
        masks.append(segmentation == int(seg["id"]))
        cats.append(label)
        scores.append(float(seg.get("score", 1.0)))
    h, w = segmentation.shape
    if not masks:
        return FramePrediction(
            masks=torch.zeros((0, h, w), dtype=torch.bool),
            categories=torch.zeros((0,), dtype=torch.int64),
            scores=torch.zeros((0,), dtype=torch.float32),
        )
    return FramePrediction(
        masks=torch.stack(masks).bool(),
        categories=torch.tensor(cats, dtype=torch.int64),
        scores=torch.tensor(scores, dtype=torch.float32),
    )


def extract_segments_with_queries(
    outputs, target_size, *, threshold=0.5, mask_threshold=0.5, overlap=0.8
):
    """
    Reimplement Mask2Former panoptic post-processing, recording source queries.

    Additionally records each segment's source decoder-query index so its
    appearance embedding can be recovered.

    Returns ``(segmentation (H, W) int32, segments [{id,label_id,score,q}],
    emb_keep (K_kept, D))``. Bit-identical to
    ``post_process_panoptic_segmentation`` for the segmentation map and segment
    ids/labels; ``emb_keep`` is the kept-query decoder hidden state, indexed by
    each segment's recorded ``q``.
    """
    from transformers.models.mask2former.image_processing_mask2former import (
        check_segment_validity,
    )

    cls = outputs.class_queries_logits
    masks = outputs.masks_queries_logits
    emb = outputs.transformer_decoder_last_hidden_state
    num_labels = cls.shape[-1] - 1
    mp384 = nn.functional.interpolate(
        masks, size=(384, 384), mode="bilinear", align_corners=False
    ).sigmoid()[0]
    sc, lb = nn.functional.softmax(cls, dim=-1).max(-1)
    sc, lb = sc[0], lb[0]
    keep = lb.ne(num_labels) & (sc > threshold)
    mp, sc, lb, emb_keep = mp384[keep], sc[keep], lb[keep], emb[0][keep]
    h, w = target_size
    seg = torch.zeros((h, w), dtype=torch.int32)
    mpr = nn.functional.interpolate(
        mp.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
    )[0]
    mpr = mpr * sc.view(-1, 1, 1)
    mlab = mpr.argmax(0)
    segs, csid = [], 0
    for k in range(lb.shape[0]):
        exists, mk = check_segment_validity(mlab, mpr, k, mask_threshold, overlap)
        if exists:
            csid += 1
            seg[mk] = csid
            segs.append(
                {"id": csid, "label_id": int(lb[k]), "score": float(sc[k]), "q": k}
            )
    return seg, segs, emb_keep


def _mask_centroid(mask: torch.Tensor) -> tuple[float, float]:
    """Pixel-mean centroid ``(x, y)`` of a boolean mask (empty -> ``(0, 0)``)."""
    ys, xs = mask.nonzero(as_tuple=True)
    if xs.numel() == 0:
        return 0.0, 0.0
    return float(xs.float().mean()), float(ys.float().mean())


def segments_to_prediction_with_features(
    segmentation: torch.Tensor,
    segments: list[dict],
    emb_keep: torch.Tensor,
    *,
    thing_ids: Iterable[int],
) -> FramePrediction:
    """
    Build a thing-only ``FramePrediction`` with embeddings and centroids.

    Uses the same thing-filter logic as :func:`segments_to_prediction`.
    """
    things = {int(t) for t in thing_ids}
    masks, cats, scores, embeds, centroids = [], [], [], [], []
    for seg in segments:
        label = int(seg["label_id"])
        if label not in things:
            continue
        mask = segmentation == int(seg["id"])
        masks.append(mask)
        cats.append(label)
        scores.append(float(seg.get("score", 1.0)))
        embeds.append(emb_keep[int(seg["q"])])
        centroids.append(_mask_centroid(mask))
    h, w = segmentation.shape
    dim = emb_keep.shape[-1] if emb_keep.ndim == 2 else EMBED_DIM
    if not masks:
        return FramePrediction(
            masks=torch.zeros((0, h, w), dtype=torch.bool),
            categories=torch.zeros((0,), dtype=torch.int64),
            scores=torch.zeros((0,), dtype=torch.float32),
            embeddings=torch.zeros((0, dim), dtype=torch.float32),
            centroids=torch.zeros((0, 2), dtype=torch.float32),
        )
    return FramePrediction(
        masks=torch.stack(masks).bool(),
        categories=torch.tensor(cats, dtype=torch.int64),
        scores=torch.tensor(scores, dtype=torch.float32),
        embeddings=torch.stack(embeds).to(torch.float32),
        centroids=torch.tensor(centroids, dtype=torch.float32),
    )


class HFPanopticAdapter:
    """Loads a HF panoptic model and emits per-frame thing instances."""

    def __init__(self, key: str, *, thing_ids: Iterable[int]) -> None:
        if key not in _MODEL_REPOS:
            msg = f"unknown model key {key!r}; known: {sorted(_MODEL_REPOS)}"
            raise KeyError(msg)
        self.key = key
        self.repo_id = _MODEL_REPOS[key]
        self.thing_ids = {int(t) for t in thing_ids}
        self._model = None
        self._processor = None
        self._device = torch.device("cpu")

    def load(self, device: torch.device) -> None:
        """Download and instantiate the HF model + image processor on ``device``."""
        from transformers import (
            AutoImageProcessor,
            AutoModelForUniversalSegmentation,
        )

        self._device = device
        self._processor = AutoImageProcessor.from_pretrained(self.repo_id)
        self._model = (
            AutoModelForUniversalSegmentation.from_pretrained(self.repo_id)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> FramePrediction:
        """Run panoptic inference on one RGB frame and return thing instances."""
        from PIL import Image

        pil = Image.fromarray(image)
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        h, w = image.shape[:2]
        seg, segments, emb_keep = extract_segments_with_queries(outputs, (h, w))
        seg = seg.cpu()
        emb_keep = emb_keep.cpu()
        return segments_to_prediction_with_features(
            seg, segments, emb_keep, thing_ids=self.thing_ids
        )


def build_model(key: str, *, thing_ids: Iterable[int]) -> HFPanopticAdapter:
    """Construct an :class:`HFPanopticAdapter` for a registry ``key``."""
    return HFPanopticAdapter(key, thing_ids=thing_ids)


MODEL_REGISTRY = dict(_MODEL_REPOS)
