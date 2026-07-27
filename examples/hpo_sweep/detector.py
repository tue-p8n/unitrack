"""
Mask2Former → unitrack.Detections adapter.

Wraps a HuggingFace ``transformers.Mask2FormerForUniversalSegmentation`` model
and converts its outputs to the typed records that unitrack consumes.

Per-frame outputs:
- ``kernel``  : (M, hidden_dim) — post-decoder query embeddings
- ``mask``    : (M, H', W')     — boolean mask resampled to the schema shape
- ``klass``   : (M,)            — argmax over class logits
- ``score``   : (M,)            — softmax confidence of the argmax class
- ``bbox``    : (M, 4)          — xyxy bbox derived from mask
- ``bbox_centroid``  : (M, 2)   — mean (x, y) of mask pixels
"""

from __future__ import annotations

import dataclasses
import typing

import torch
from unitrack.data import Detections

if typing.TYPE_CHECKING:
    from PIL.Image import Image
    from transformers import (
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )

__all__ = ["Mask2FormerDetector"]


@dataclasses.dataclass(slots=True)
class Mask2FormerDetector:
    """
    Run Mask2Former on a single image and emit unitrack ``Detections``.

    Construct via :meth:`from_huggingface` to load weights from the HuggingFace
    Hub. The default checkpoint is a tiny Cityscapes-instance variant; see the
    README for notes on swapping to a ResNet-50 backbone.
    """

    model: Mask2FormerForUniversalSegmentation
    processor: Mask2FormerImageProcessor
    score_threshold: float = 0.3
    max_detections: int = 50
    target_mask_shape: tuple[int, int] = (96, 192)
    device: torch.device = dataclasses.field(
        default_factory=lambda: torch.device("cpu"),
    )

    @classmethod
    def from_huggingface(
        cls,
        model_name: str = "facebook/mask2former-swin-tiny-cityscapes-instance",
        *,
        device: str | torch.device = "cpu",
        score_threshold: float = 0.3,
        max_detections: int = 50,
        target_mask_shape: tuple[int, int] = (96, 192),
    ) -> Mask2FormerDetector:
        """Load model + processor from the HuggingFace Hub."""
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        torch_device = torch.device(device)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        model = model.to(torch_device).eval()
        processor = Mask2FormerImageProcessor.from_pretrained(model_name)

        return cls(
            model=model,
            processor=processor,
            score_threshold=score_threshold,
            max_detections=max_detections,
            target_mask_shape=target_mask_shape,
            device=torch_device,
        )

    @torch.inference_mode()
    def __call__(self, image: Image) -> Detections:
        """Run one forward pass and return the per-detection record."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # ``class_queries_logits``  : (1, Q, n_classes + 1)  (last is "no object")
        # ``masks_queries_logits``  : (1, Q, h, w)
        # ``transformer_decoder_last_hidden_state`` : (1, Q, hidden_dim)
        cls_logits = outputs.class_queries_logits[0]  # (Q, C+1)
        mask_logits = outputs.masks_queries_logits[0]  # (Q, h, w)
        kernels = outputs.transformer_decoder_last_hidden_state[0]  # (Q, D)

        # Per-query softmax — drop the "no object" channel.
        probs = torch.softmax(cls_logits, dim=-1)  # (Q, C+1)
        probs = probs[:, :-1]  # (Q, C)
        score, klass = probs.max(dim=-1)  # (Q,), (Q,)

        # Filter by confidence + take top-k.
        keep = score >= self.score_threshold
        idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        if idx.numel() > self.max_detections:
            _, top_idx = score[idx].topk(self.max_detections)
            idx = idx[top_idx]
        if idx.numel() == 0:
            return _empty_detections(
                kernel_dim=kernels.shape[-1],
                mask_shape=self.target_mask_shape,
                device=self.device,
            )

        kernel = kernels[idx]  # (M, D)
        klass = klass[idx]  # (M,)
        score = score[idx]  # (M,)
        mask_logits_kept = mask_logits[idx]  # (M, h, w)

        # Resample masks to a fixed schema shape so Tracklets can store them.
        masks = torch.nn.functional.interpolate(
            mask_logits_kept.unsqueeze(1),
            size=self.target_mask_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (M, H', W')
        masks_bin = masks > 0.0  # (M, H', W')

        bbox, centroid = _bboxes_and_centroids_from_masks(masks_bin)

        m = idx.numel()
        return Detections(
            index=torch.arange(m, dtype=torch.int64, device=self.device),
            kernel=kernel,
            mask=masks_bin,
            klass=klass,
            score=score,
            bbox=bbox,
            centroid=centroid,
            batch_size=[m],
        )


def _bboxes_and_centroids_from_masks(
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each mask, compute (xyxy bbox, (x, y) centroid).

    Empty masks (all-False) get a zero bbox + zero centroid; downstream
    cost modules should skip them via score-gate or class-gate.
    """
    m, _h, _w = masks.shape
    device = masks.device
    bboxes = torch.zeros((m, 4), dtype=torch.float32, device=device)
    centroids = torch.zeros((m, 2), dtype=torch.float32, device=device)

    for i in range(m):
        ys, xs = torch.nonzero(masks[i], as_tuple=True)
        if ys.numel() == 0:
            continue
        x_min, x_max = xs.min().float(), xs.max().float()
        y_min, y_max = ys.min().float(), ys.max().float()
        bboxes[i] = torch.stack([x_min, y_min, x_max, y_max])
        centroids[i] = torch.stack([xs.float().mean(), ys.float().mean()])

    return bboxes, centroids


def _empty_detections(
    *,
    kernel_dim: int,
    mask_shape: tuple[int, int],
    device: torch.device,
) -> Detections:
    """Return a zero-row Detections with the same schema as a real frame."""
    return Detections(
        index=torch.zeros(0, dtype=torch.int64, device=device),
        kernel=torch.zeros((0, kernel_dim), dtype=torch.float32, device=device),
        mask=torch.zeros((0, *mask_shape), dtype=torch.bool, device=device),
        klass=torch.zeros(0, dtype=torch.int64, device=device),
        score=torch.zeros(0, dtype=torch.float32, device=device),
        bbox=torch.zeros((0, 4), dtype=torch.float32, device=device),
        centroid=torch.zeros((0, 2), dtype=torch.float32, device=device),
        batch_size=[0],
    )
