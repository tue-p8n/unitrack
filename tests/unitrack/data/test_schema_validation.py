"""Reserved-field schema validation on Tracklets / Detections construction.

The user-facing :class:`Tracklets` / :class:`Detections` constructors validate
that every reserved field is present with the documented dtype and a leading
dim matching ``batch_size[0]``. This file pins the rejection paths so a
regression in the validator can't silently re-open the foot-gun.

Internal TensorDict paths (slicing, ``torch.cat``, ``to``) bypass validation
by design — re-checking after every internal view is hot-loop cost for no
gain. The validator's job is to guard the explicit construction surface.
"""

from __future__ import annotations

import pytest
import torch
from unitrack.data import Detections, Tracklets

# --------- Tracklets reserved-field validation ---------------------------


def test_tracklets_missing_id_field_raises():
    with pytest.raises(ValueError, match="missing reserved field 'id'"):
        Tracklets(
            status=torch.zeros(2, dtype=torch.int8),
            hits=torch.zeros(2, dtype=torch.int32),
            time_since_update=torch.zeros(2, dtype=torch.int32),
            age=torch.zeros(2, dtype=torch.int32),
            frame_started=torch.zeros(2, dtype=torch.int32),
            frame_last_seen=torch.zeros(2, dtype=torch.int32),
            batch_size=[2],
        )


def test_tracklets_wrong_id_dtype_raises():
    with pytest.raises(ValueError, match=r"'id' must have dtype torch\.int64"):
        Tracklets(
            id=torch.zeros(2, dtype=torch.int32),  # wrong: should be int64
            status=torch.zeros(2, dtype=torch.int8),
            hits=torch.zeros(2, dtype=torch.int32),
            time_since_update=torch.zeros(2, dtype=torch.int32),
            age=torch.zeros(2, dtype=torch.int32),
            frame_started=torch.zeros(2, dtype=torch.int32),
            frame_last_seen=torch.zeros(2, dtype=torch.int32),
            batch_size=[2],
        )


def test_tracklets_wrong_status_dtype_raises():
    with pytest.raises(ValueError, match=r"'status' must have dtype torch\.int8"):
        Tracklets(
            id=torch.arange(2, dtype=torch.int64),
            status=torch.zeros(2, dtype=torch.int32),  # wrong: should be int8
            hits=torch.zeros(2, dtype=torch.int32),
            time_since_update=torch.zeros(2, dtype=torch.int32),
            age=torch.zeros(2, dtype=torch.int32),
            frame_started=torch.zeros(2, dtype=torch.int32),
            frame_last_seen=torch.zeros(2, dtype=torch.int32),
            batch_size=[2],
        )


def test_tracklets_wrong_id_shape_raises():
    with pytest.raises(ValueError, match="'id' must have leading dim 2"):
        Tracklets(
            id=torch.arange(
                3, dtype=torch.int64
            ),  # wrong: shape (3,) vs batch_size [2]
            status=torch.zeros(2, dtype=torch.int8),
            hits=torch.zeros(2, dtype=torch.int32),
            time_since_update=torch.zeros(2, dtype=torch.int32),
            age=torch.zeros(2, dtype=torch.int32),
            frame_started=torch.zeros(2, dtype=torch.int32),
            frame_last_seen=torch.zeros(2, dtype=torch.int32),
            batch_size=[2],
        )


def test_tracklets_valid_construction_passes_with_user_fields():
    """A correctly-typed construction with a user 'kernel' field works."""
    cs = Tracklets(
        id=torch.arange(2, dtype=torch.int64),
        status=torch.ones(2, dtype=torch.int8),
        hits=torch.ones(2, dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.ones(2, dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.zeros(2, dtype=torch.int32),
        kernel=torch.zeros(2, 4),
        batch_size=[2],
    )
    assert cs.batch_size[0] == 2
    assert cs.kernel.shape == (2, 4)


def test_tracklets_empty_factory_is_valid():
    cs = Tracklets.empty()
    assert cs.batch_size[0] == 0
    assert cs.id.dtype == torch.int64
    assert cs.status.dtype == torch.int8


def test_tracklets_bare_batch_size_no_fields_raises():
    """`Tracklets(batch_size=[N])` with no tensors must not silently produce
    a malformed snapshot — the validator runs unconditionally when batch_size
    is supplied. Use Tracklets.empty() if a zero-field record is wanted."""
    with pytest.raises(ValueError, match="missing reserved field 'id'"):
        Tracklets(batch_size=[2])


def test_tracklets_slicing_does_not_revalidate():
    """Internal TensorDict slicing produces a new Tracklets via internal
    paths that bypass our `__init__` — a regression there would otherwise
    trigger validator overhead on every cascade step."""
    cs = Tracklets(
        id=torch.arange(3, dtype=torch.int64),
        status=torch.ones(3, dtype=torch.int8),
        hits=torch.ones(3, dtype=torch.int32),
        time_since_update=torch.zeros(3, dtype=torch.int32),
        age=torch.ones(3, dtype=torch.int32),
        frame_started=torch.zeros(3, dtype=torch.int32),
        frame_last_seen=torch.zeros(3, dtype=torch.int32),
        batch_size=[3],
    )
    keep = torch.tensor([True, False, True])
    sub = cs[keep]
    assert sub.batch_size[0] == 2
    assert sub.id.tolist() == [0, 2]


# --------- Detections reserved-field validation --------------------------


def test_detections_missing_index_field_raises():
    with pytest.raises(ValueError, match="missing reserved field 'index'"):
        Detections(kernel=torch.zeros(2, 4), batch_size=[2])


def test_detections_wrong_index_dtype_raises():
    with pytest.raises(ValueError, match=r"'index' must have dtype torch\.int64"):
        Detections(
            index=torch.zeros(2, dtype=torch.int32),
            batch_size=[2],
        )


def test_detections_wrong_index_shape_raises():
    with pytest.raises(ValueError, match="'index' must have leading dim 3"):
        Detections(
            index=torch.arange(2, dtype=torch.int64),
            batch_size=[3],
        )


def test_detections_valid_construction_passes_with_user_fields():
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.zeros(2, 4),
        batch_size=[2],
    )
    assert ds.batch_size[0] == 2
    assert ds.kernel.shape == (2, 4)


def test_detections_empty_factory_is_valid():
    ds = Detections.empty()
    assert ds.batch_size[0] == 0
    assert ds.index.dtype == torch.int64


def test_detections_bare_batch_size_no_fields_raises():
    """`Detections(batch_size=[M])` with no tensors must not silently produce
    a malformed record — the validator runs unconditionally when batch_size
    is supplied. Use Detections.empty() if a zero-field record is wanted."""
    with pytest.raises(ValueError, match="missing reserved field 'index'"):
        Detections(batch_size=[2])
