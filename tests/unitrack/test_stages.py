r"""Tests for ``unitrack.stages``."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tensordict import TensorDict
from unitrack import assignment, costs, stages


def test_lost_stage():
    for num_frames in range(1, 10):
        for max_lost in range(1, num_frames):
            stage_lost = stages.Lost(max_lost=max_lost)

            cs = TensorDict.from_dict(
                {"_frame": torch.arange(num_frames, dtype=torch.long)},
                batch_size=[num_frames],
            )
            ds = TensorDict.from_dict(
                {
                    "_frame": torch.empty((0,), dtype=torch.long),
                    "_index": torch.empty((0,), dtype=torch.long),
                },
                batch_size=[0],
            )

            assert len(cs) == num_frames, cs
            assert len(ds) == 0, ds

            ctx = TensorDict.from_dict(
                {"_frame": torch.tensor(num_frames) - 1, "_delta": torch.tensor(1.0)}
            )
            cs, ds = stage_lost(ctx, cs, ds)

            # Calculate expected kept count based on logic: time_lost = t - Tc - dt;
            # keep if time_lost <= max_lost
            time_lost = (num_frames - 1) - torch.arange(num_frames) - 1.0
            expected_count = (time_lost <= max_lost).sum().item()

            assert len(cs) == expected_count, (len(cs), expected_count)
            assert len(ds) == 0, ds


@settings(
    deadline=None,
)
@given(cs_num=st.integers(0, 10), ds_num=st.integers(0, 10))
def test_association_stage(cs_num: int, ds_num: int):
    x_cs = cs_num * 2.0
    cs = TensorDict.from_dict(
        {
            "x": torch.arange(cs_num).unsqueeze(1).float() * x_cs,
            "categories": torch.ones(cs_num),
            "_id": torch.arange(cs_num, dtype=torch.long) + 1,
            "_index": torch.full((cs_num,), -1, dtype=torch.long),
            "_frame": torch.zeros(cs_num, dtype=torch.long),
            "_active": torch.ones(cs_num, dtype=torch.bool),
        },
        batch_size=[cs_num],
    )
    assert len(cs) == cs_num, cs
    x_ds = ds_num * 3.0 + 1
    ds = TensorDict.from_dict(
        {
            "x": torch.arange(ds_num).unsqueeze(1).float() * x_ds,
            "categories": torch.ones(ds_num),
            "_index": torch.arange(ds_num, dtype=torch.long),
        },
        batch_size=[ds_num],
    )
    assert len(ds) == ds_num, ds

    cost = costs.CDist(field="x")
    ass_stage = stages.Association(cost, assignment.Jonker(torch.inf))
    ass_num = min(cs_num, ds_num)

    ctx = TensorDict.from_dict({})
    cs, ds = ass_stage(ctx, cs, ds)

    assert len(cs) == cs_num - ass_num
    assert len(ds) == ds_num - ass_num
