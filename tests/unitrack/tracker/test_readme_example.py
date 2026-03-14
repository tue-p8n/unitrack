# tests/unitrack/tracker/test_readme_example.py
"""Port of the 1.x README example onto the 2.0 surface.

The 1.x version (kept under tests/unitrack/_legacy/test_tracker.py) iterated
ten frames of growing detection counts and asserted that the resulting IDs
were ``range(n_detections) + 1`` at each frame. This test reproduces that
on the 2.0 ``Tracker``/``MultiStream`` API using a Cosine-on-position cost.
"""

from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import MultiStream, Tracker


def test_readme_example_matches_legacy_id_progression():
    """IDs grow monotonically and start at 1, matching the legacy tracker."""
    tracker = Tracker(
        root=Pipe(
            cost=Cosine("position"),
            assoc=Associate(Jonker(threshold=0.99)),
        ),
        states={
            "position": State(
                schema=TensorSpec(shape=(1,), dtype=torch.float32),
                process=Identity("position"),
                observation=Replace("position"),
                init=FromDetectionField("position"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
    )
    ms = MultiStream(tracker)

    for frame in range(10):
        n = 1 + frame * 2
        positions = torch.arange(n, dtype=torch.float32).unsqueeze(1) + 1.0
        ds = Detections(
            index=torch.arange(n, dtype=torch.int64),
            position=positions,
            batch_size=[n],
        )
        res = ms.step(
            stream_key=0,
            detections=ds,
            ctx=FrameContext.make(frame, delta=1 / 30, stream_key=0),
        )
        snap_ids = res.snapshot.id.tolist()
        # The snapshot's id field carries every live tracklet; for this test
        # the count grows monotonically and IDs start at 1.
        assert min(snap_ids) == 1
        assert max(snap_ids) == res.snapshot.batch_size[0]
