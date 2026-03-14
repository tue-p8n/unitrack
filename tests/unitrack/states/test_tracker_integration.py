"""Each new embedding-filter state runs inside a real Tracker + MultiStream."""

from __future__ import annotations

import pytest
import torch
import unitrack
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine, GalleryCost
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import (
    FromDetectionField,
    LearnedObservation,
    LearnedProcess,
    State,
    enkf_state_entries,
    gallery_state_entries,
    information_state_entries,
    vmf_state_entries,
)

DIM = 8


def _clip(n_frames: int = 4, n_obj: int = 2):
    """Orthogonal per-identity embeddings, shuffled per frame with light noise."""
    g = torch.Generator().manual_seed(0)
    base = torch.linalg.qr(torch.randn(DIM, DIM, generator=g))[0][:n_obj]
    frames, order_per_frame = [], []
    for _ in range(n_frames):
        order = torch.randperm(n_obj, generator=g)
        order_per_frame.append(order)
        emb = base[order] + 0.01 * torch.randn(n_obj, DIM, generator=g)
        frames.append(
            Detections(
                index=torch.arange(n_obj, dtype=torch.int64),
                emb=emb.float(),
                batch_size=[n_obj],
            )
        )
    return frames, n_obj


def _learned_states():
    return {
        "emb": State(
            schema=TensorSpec(shape=(DIM,), dtype=torch.float32),
            process=LearnedProcess("emb", lambda v, _: v),  # identity propagate
            observation=LearnedObservation("emb", "emb", lambda _, m: m),  # adopt
            init=FromDetectionField("emb"),
        )
    }


CONFIGS = {
    "vmf": (vmf_state_entries("emb", dim=DIM), Cosine("emb")),
    "enkf": (enkf_state_entries("emb", dim=DIM, ensemble_size=16), Cosine("emb")),
    "information": (information_state_entries("emb", dim=DIM), Cosine("emb")),
    "gallery": (
        gallery_state_entries("emb", dim=DIM, capacity=4),
        GalleryCost("emb_gallery", "emb_count", "emb", reduce="max"),
    ),
    "learned": (_learned_states(), Cosine("emb")),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_state_runs_in_tracker_and_keeps_stable_ids(name):
    states, cost = CONFIGS[name]
    tracker = unitrack.Tracker(
        root=Pipe(cost=cost, assoc=Associate(Jonker(threshold=0.5))),
        states=states,
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )
    ms = unitrack.MultiStream(tracker)
    clip, n_obj = _clip()

    seen_ids = set()
    for k, dets in enumerate(clip):
        ctx = FrameContext.make(frame_idx=k, delta=1.0, fps=1.0, stream_key=0)
        res = ms.step(stream_key=0, detections=dets, ctx=ctx)
        seen_ids.update(res.snapshot.id.tolist())

    # Two identities, consistently re-associated: never spawns more than n_obj ids.
    assert res.snapshot.batch_size[0] == n_obj
    assert len(seen_ids) == n_obj
