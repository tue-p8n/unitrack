r"""Tests for ``unitrack.tracker``."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from unitrack import (
    MultiStageTracker,
    TrackletMemory,
    assignment,
    costs,
    stages,
    states,
)

KEY_CATEGORY = "object_category"
KEY_POSITION = "position_2d"


def test_tracker():
    device = torch.device("cpu")
    dtype = torch.float32

    with device:
        frames = [
            {
                KEY_CATEGORY: torch.ones(1 + frame * 2, dtype=torch.long),
                KEY_POSITION: (torch.arange(1 + frame * 2, dtype=dtype)).unsqueeze(1),
            }
            for frame in range(10)
        ]

    trk = MultiStageTracker(
        fields=[
            TensorDictModule(
                nn.Identity(), in_keys=[KEY_POSITION], out_keys=[KEY_POSITION]
            ),
            TensorDictModule(
                nn.Identity(), in_keys=[KEY_CATEGORY], out_keys=[KEY_CATEGORY]
            ),
        ],
        stages=[
            stages.Association(
                cost=costs.CDist(KEY_POSITION), assignment=assignment.Jonker(10)
            )
        ],
    )

    mem = TrackletMemory(
        states={
            KEY_POSITION: states.Value(dtype),
            KEY_CATEGORY: states.Value(dtype=torch.long),
        }
    )

    for frame, det in enumerate(frames):
        det = TensorDict.from_dict(det, device=device)
        det_amt = det.batch_size[0]

        ctx, obs = mem.read(frame)
        obs, new = trk(ctx, obs, det, det_amt)
        ids = mem.write(ctx, obs, new)

        assert len(ids) == len(det[KEY_POSITION])
        assert torch.all(
            ids == torch.arange(len(det[KEY_POSITION]), dtype=torch.long) + 1
        )
        assert isinstance(ids, torch.Tensor), type(ids)
