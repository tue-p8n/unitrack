r"""Implements an object tracker that uses the overlap between consecutive frames' object
bounding boxes to determine the association between them.
"""
import typing

import torch
import unitrack as ut
from torch import nn


class MinScoreGate(nn.Module):
    """Selects and filters the input tensor based on the class label and a minimum
    detection score.
    """

    def __init__(
        self,
        key_class: str,
        key_score: str,
        min_score: float = 0.0,
    ):
        super().__init__()

        self.min_score = min_score
        self.key_score = key_score

    @typing.override
    def forward(self, _, cs, ds) -> tuple[torch.Tensor, torch.Tensor]:
        cs_mask = torch.full(
            cs.batch_size[:1], True, dtype=torch.bool, device=cs.device
        )
        ds_mask = ds.get(self.key_score) > self.min_score
        return cs_mask, ds_mask


def build_overlap_tracker(
    *,
    key_score: str = "score",
    key_class: str = "class",
    key_bbox: str = "bbox",
    threshold: float = 0.5,
    min_score: float = 0.1,
    class_gate: bool = True,
) -> tuple[ut.MultiStageTracker, ut.TrackletMemory]:
    # Define the cost function, e.g. the IoU between the bounding boxes
    cost = ut.costs.BoxIoU(field=key_bbox)
    if class_gate:
        cost = ut.costs.GateCost(key_class).wrap(cost)

    # Define the tracker as a single-stage assignment process
    tracker = ut.MultiStageTracker(
        fields=[ut.SelectField(key_score, key_class, key_bbox)],
        stages=[
            ut.stages.Gate(
                gate=MinScoreGate(min_score=min_score),
                then=[
                    ut.stages.Association(
                        cost=cost,
                        assignment=ut.assignment.Jonker(threshold=threshold),
                    ),
                ],
            )
        ],
    )

    # The memory module stores the trackers' score, class and bounding box
    memory = ut.TrackletMemory(
        states={
            key_score: ut.states.Value(torch.float),
            key_class: ut.states.Value(torch.long),
            key_bbox: ut.states.Value(torch.float),
        },
    )

    return tracker, memory
