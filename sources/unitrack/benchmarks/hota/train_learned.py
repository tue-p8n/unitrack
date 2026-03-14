"""
Train the learned MOTR-style appearance filter and write its checkpoint.

This is a one-time script that produces the small committed checkpoint
``benchmarks/hota/weights/learned_filter.safetensors`` consumed by
:func:`~unitrack.benchmarks.hota.tracker.build_learned_tracker`.

Pipeline
--------
1. Read a handful of train clips from the cityscapes-dvps LMDB; per frame run the
   panoptic model to extract ``(masks, embeddings, categories)`` and the GT
   panoptic, and assign each detection a GT track id by mask-IoU >= 0.5. Extracted
   embeddings are cached to a ``.pt`` keyed by ``(repo, n_clips)`` so re-runs skip
   the slow model inference.
2. Within a clip, each GT track id yields a frame-ordered sequence of its
   detection embeddings. The :func:`train_step` rolls a track state ``h`` forward
   through that sequence — ``h_pred = Propagator(h)`` then ``h = Fuser(h_pred,
   d_t)`` — and applies a per-step **InfoNCE** loss that asks ``h_pred`` to be
   more cosine-similar to its own next detection ``d_t`` than to every other
   detection in frame ``t``, by a temperature margin. This gives the Propagator a
   discriminative gradient and makes the Fuser useful: a bad fuse corrupts ``h``
   and hurts the next step's InfoNCE, so the degenerate hard-replace fuse is no
   longer optimal.
3. Save the two module state dicts (prefixed, flattened) via safetensors.

The numerically heavy step (model inference over the LMDB) lives in the script's
``main``; :func:`train_step` is a pure, in-memory function so it is covered by a
CI-safe smoke test on synthetic embeddings (no model, no LMDB).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

from .learned_modules import Fuser, Propagator

# Mask2Former decoder-query / appearance-embedding dimensionality.
EMBED_DIM = 256

# Committed checkpoint location (read by the learned factory).
DEFAULT_CKPT = Path(__file__).parent / "weights" / "learned_filter.safetensors"

# InfoNCE temperature for the contrastive rollout loss. The standard 0.07 from
# the contrastive-representation literature; small enough to demand a clear
# margin between the GT-correspondent detection and the in-frame distractors.
TEMPERATURE = 0.07

# Frame is ``(embeddings (n, D), gt_ids (n,))``; a clip is a list of frames.
Frame = tuple[torch.Tensor, torch.Tensor]
Clip = list[Frame]


def _normalize_clip(clip: Clip) -> Clip:
    """L2-normalize every frame's detection embeddings (match inference)."""
    out: Clip = []
    for emb, ids in clip:
        if emb.shape[0] == 0:
            out.append((emb, ids))
        else:
            out.append((F.normalize(emb.float(), dim=-1), ids))
    return out


def _track_sequences(clip: Clip) -> dict[int, list[tuple[int, int]]]:
    """
    Group a clip's detections by GT track id into frame-ordered sequences.

    Returns ``{gt_id: [(frame_idx, det_idx), ...]}`` for every positive GT id
    present in two or more frames (a single-frame track has no rollout step).
    """
    seqs: dict[int, list[tuple[int, int]]] = {}
    for f, (_emb, ids) in enumerate(clip):
        for d, gt in enumerate(ids.tolist()):
            if gt < 0:
                continue
            seqs.setdefault(gt, []).append((f, d))
    return {gt: occ for gt, occ in seqs.items() if len(occ) >= 2}


def train_step(
    propagator: Propagator,
    fuser: Fuser,
    clip: Clip,
    optimizer: torch.optim.Optimizer,
    *,
    temperature: float = TEMPERATURE,
) -> float:
    """
    Run one contrastive multi-frame optimization step on a single clip.

    Parameters
    ----------
    propagator, fuser
        The learned modules being trained.
    clip
        A list of frames ``[(embeddings (n_f, D), gt_ids (n_f,)), ...]`` for one
        clip in frame order. Detection embeddings are L2-normalized internally so
        training matches inference (where the Fuser emits unit norm and the
        cosine cost normalizes).
    optimizer
        An optimizer over both modules' parameters.
    temperature
        InfoNCE temperature; smaller demands a sharper margin.

    Returns
    -------
    float
        The scalar loss before the gradient step (``0.0`` if the clip has no
        multi-frame GT track to roll out).

    Notes
    -----
    For each GT track, the state ``h`` is initialized to its first detection and
    rolled forward: ``h_pred = Propagator(h)`` is scored against *all* detections
    in the next frame via cosine similarity, and the InfoNCE (cross-entropy) loss
    pushes ``h_pred`` toward its own GT-correspondent detection over the in-frame
    distractors. The state is then updated with ``h = Fuser(h_pred, d_t)``, so
    the loss flows through both modules across the whole rollout — a poor fuse
    corrupts later predictions. Losses accumulate over all steps of all tracks
    and backprop once.

    """
    optimizer.zero_grad()
    clip = _normalize_clip(clip)
    sequences = _track_sequences(clip)
    if not sequences:
        # Nothing to roll out; return a finite zero loss without a step.
        return 0.0

    losses: list[torch.Tensor] = []
    for occ in sequences.values():
        f0, d0 = occ[0]
        h = clip[f0][0][d0]
        for f, d in occ[1:]:
            emb_f = clip[f][0]
            h_pred = propagator(h.unsqueeze(0)).squeeze(0)
            logits = (emb_f @ h_pred) / temperature
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([d])))
            h = fuser(h_pred.unsqueeze(0), emb_f[d].unsqueeze(0)).squeeze(0)

    if not losses:
        return 0.0
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _flatten_state(propagator: Propagator, fuser: Fuser) -> dict[str, torch.Tensor]:
    """Flatten the two module state dicts into one prefixed ``str -> tensor`` map."""
    flat: dict[str, torch.Tensor] = {}
    for k, v in propagator.state_dict().items():
        flat[f"propagator.{k}"] = v
    for k, v in fuser.state_dict().items():
        flat[f"fuser.{k}"] = v
    return flat


def save_checkpoint(
    propagator: Propagator, fuser: Fuser, path: str | Path = DEFAULT_CKPT
) -> Path:
    """Write the flattened ``{propagator,fuser}`` state to a safetensors file."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(_flatten_state(propagator, fuser), str(path))
    return path


def _mask_iou_assign(
    masks: torch.Tensor, gt_panoptic: torch.Tensor, *, offset: int, thr: float = 0.5
) -> torch.Tensor:
    """
    Assign each detection mask the best-overlapping GT instance id (>= ``thr``).

    Returns a ``(n_dets,)`` long tensor of GT panoptic ids (``semantic*offset +
    instance``); ``-1`` where no GT instance reaches the IoU threshold.
    """
    gt = gt_panoptic.to(torch.int64)
    gt_ids = [int(v) for v in torch.unique(gt).tolist() if int(v) % offset != 0]
    out = torch.full((masks.shape[0],), -1, dtype=torch.int64)
    if not gt_ids:
        return out
    gt_masks = {g: (gt == g) for g in gt_ids}
    for d in range(masks.shape[0]):
        m = masks[d].bool()
        best_iou, best_id = 0.0, -1
        for g, gm in gt_masks.items():
            inter = float((m & gm).sum())
            union = float((m | gm).sum())
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_id = iou, g
        if best_iou >= thr:
            out[d] = best_id
    return out


def _extract_clips(model, dataset, *, offset: int, n_clips: int):
    """
    Run the model over the first ``n_clips`` sequences; return per-clip frames.

    Each frame is ``(embeddings (n, D), gt_ids (n,))`` with detection embeddings
    and their mask-IoU-assigned GT track ids.
    """
    clips: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for seq_idx, sample in enumerate(dataset.sequences()):
        if seq_idx >= n_clips:
            break
        frames: list[tuple[torch.Tensor, torch.Tensor]] = []
        for image, gt_panoptic in sample.frames:
            pred = model(image)
            if pred.embeddings is None or pred.embeddings.shape[0] == 0:
                frames.append(
                    (torch.zeros((0, EMBED_DIM)), torch.empty((0,), dtype=torch.long))
                )
                continue
            gt = torch.as_tensor(gt_panoptic)
            ids = _mask_iou_assign(pred.masks, gt, offset=offset)
            # The model runs under torch.inference_mode(), so its outputs are
            # "inference tensors" that cannot be saved for backward. Clone the
            # embeddings into normal tensors so they can be autograd constants
            # feeding the (trainable) Propagator/Fuser.
            frames.append((pred.embeddings.float().clone(), ids))
        clips.append(frames)
    return clips


def main(argv: list[str] | None = None) -> None:
    """Extract train clips, train the learned filter, and save the checkpoint."""
    p = argparse.ArgumentParser(prog="unitrack.benchmarks.hota.train_learned")
    p.add_argument("--model", default="mask2former-tiny")
    p.add_argument("--lmdb", default=None, help="path to the train LMDB")
    p.add_argument("--n-clips", type=int, default=8)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(DEFAULT_CKPT))
    p.add_argument(
        "--cache", default=None, help="path for the extracted-embedding cache"
    )
    a = p.parse_args(argv)

    from .datasets import CityscapesDVPSDataset
    from .models import build_model

    torch.manual_seed(a.seed)
    default_lmdb = Path.home() / "Datasets/cityscapes-dvps/cityscapes-dvps.train.lmdb"
    lmdb_path = Path(a.lmdb) if a.lmdb else default_lmdb
    dataset = CityscapesDVPSDataset(lmdb_path, limit_seqs=a.n_clips)

    cache_path = (
        Path(a.cache)
        if a.cache
        else Path(a.out).parent / f"_extract_{a.model}_{a.n_clips}.pt"
    )
    if cache_path.exists():
        clips = torch.load(cache_path, weights_only=False)
    else:
        model = build_model(a.model, thing_ids=dataset.thing_ids)
        model.load(torch.device("cpu"))
        clips = _extract_clips(model, dataset, offset=dataset.offset, n_clips=a.n_clips)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clips, cache_path)

    # Keep only clips that contain a multi-frame GT track to roll out.
    trainable = [clip for clip in clips if _track_sequences(clip)]
    if not trainable:
        msg = "no multi-frame GT tracks extracted; cannot train"
        raise RuntimeError(msg)

    prop = Propagator(EMBED_DIM)
    fuse = Fuser(EMBED_DIM)
    opt = torch.optim.Adam([*prop.parameters(), *fuse.parameters()], lr=a.lr)

    g = torch.Generator().manual_seed(a.seed)
    for step in range(a.steps):
        idx = int(torch.randint(len(trainable), (1,), generator=g))
        loss = train_step(prop, fuse, trainable[idx], opt)
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss:.4f}")  # noqa: T201

    out = save_checkpoint(prop, fuse, a.out)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
