"""N=0 / M=0 smoke for every public cost and gate leaf.

Empty-input coverage was previously only at the ``Associate`` layer; a
cost or gate leaf that crashed on zero-row inputs would surface only when
the full tracker stack ran. This module pins the (N, M) shape contract
at every leaf so a future regression fails locally rather than in
integration.
"""

from __future__ import annotations

import pytest
import torch
from unitrack.costs import (
    RBF,
    BiSoftmax,
    BoxCIoU,
    BoxGIoU,
    BoxIoU,
    CDist,
    Chamfer,
    Cosine,
    Mahalanobis,
    MaskIoU,
)
from unitrack.data import (
    CostExpression,
    Detections,
    FrameContext,
    Gate,
    Tracklets,
)
from unitrack.gates import (
    ClassGate,
    NoneGate,
    ScoreGate,
    SpatialGate2D,
    SpatialGate3D,
)

# --------- shape-builders -----------------------------------------------
# Build per-field cs/ds tensors at any (N, M) — including N=0 / M=0 — with
# the trailing feature shape each cost/gate leaf expects.


def _cs(n: int, **fields: torch.Tensor) -> Tracklets:
    return Tracklets(
        id=torch.arange(n, dtype=torch.int64),
        status=torch.ones(n, dtype=torch.int8),
        hits=torch.ones(n, dtype=torch.int32),
        time_since_update=torch.zeros(n, dtype=torch.int32),
        age=torch.ones(n, dtype=torch.int32),
        frame_started=torch.zeros(n, dtype=torch.int32),
        frame_last_seen=torch.zeros(n, dtype=torch.int32),
        **fields,
        batch_size=[n],
    )


def _ds(m: int, **fields: torch.Tensor) -> Detections:
    return Detections(
        index=torch.arange(m, dtype=torch.int64),
        **fields,
        batch_size=[m],
    )


# --------- cost leaves ---------------------------------------------------


_VEC_COST_LEAVES = [
    Cosine(field="vec"),
    CDist(field="vec", p_norm=2.0),
    BiSoftmax(field="vec"),
    RBF(field="vec", gamma=1.0),
]


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (2, 0)])
@pytest.mark.parametrize("leaf", _VEC_COST_LEAVES)
def test_vector_cost_leaf_empty_inputs_return_zero_shaped_matrix(leaf, n, m):
    cs = _cs(n, vec=torch.zeros(n, 4))
    ds = _ds(m, vec=torch.zeros(m, 4))
    out = leaf(cs, ds, FrameContext.make(0))
    assert isinstance(out, CostExpression)
    assert tuple(out.matrix.shape) == (n, m)
    # dtype/device contract: the matched solver consumes float32 on the
    # input device; a silent int64→float64 promotion at a leaf would
    # surface only in the LAP layer where the trace is opaque.
    assert out.matrix.dtype == torch.float32
    assert out.matrix.device == cs.id.device


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (2, 0)])
def test_mask_iou_empty_inputs_return_zero_shaped_matrix(n, m):
    cs = _cs(n, mask=torch.zeros(n, 4, 4, dtype=torch.bool))
    ds = _ds(m, mask=torch.zeros(m, 4, 4, dtype=torch.bool))
    out = MaskIoU(field="mask")(cs, ds, FrameContext.make(0))
    assert tuple(out.matrix.shape) == (n, m)
    assert out.matrix.dtype == torch.float32
    assert out.matrix.device == cs.id.device


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (2, 0)])
@pytest.mark.parametrize("leaf_cls", [BoxIoU, BoxGIoU, BoxCIoU])
def test_box_iou_family_empty_inputs_return_zero_shaped_matrix(leaf_cls, n, m):
    cs = _cs(n, bbox=torch.zeros(n, 4))
    ds = _ds(m, bbox=torch.zeros(m, 4))
    out = leaf_cls(field="bbox")(cs, ds, FrameContext.make(0))
    assert tuple(out.matrix.shape) == (n, m)
    assert out.matrix.dtype == torch.float32
    assert out.matrix.device == cs.id.device


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (2, 0)])
def test_chamfer_empty_inputs_return_zero_shaped_matrix(n, m):
    """Chamfer's 5-D broadcast over (N, M, K_a, K_b) is the highest-arity
    leaf; an empty N or M zeros out the broadcast and must not crash."""
    cs = _cs(n, cloud=torch.zeros(n, 3, 3))
    ds = _ds(m, cloud=torch.zeros(m, 3, 3))
    out = Chamfer(field="cloud")(cs, ds, FrameContext.make(0))
    assert tuple(out.matrix.shape) == (n, m)
    assert out.matrix.dtype == torch.float32
    assert out.matrix.device == cs.id.device


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (2, 0)])
def test_mahalanobis_empty_inputs_return_zero_shaped_matrix(n, m):
    """Mahalanobis projects through cs covariance; the empty path must
    survive both the projection truncation and the solve_psd call."""
    cs = _cs(n, pos=torch.zeros(n, 4), pos_cov=torch.eye(4).expand(n, 4, 4).clone())
    ds = _ds(m, pos=torch.zeros(m, 2))
    out = Mahalanobis(field="pos", cov_field="pos_cov")(cs, ds, FrameContext.make(0))
    assert tuple(out.matrix.shape) == (n, m)
    assert out.matrix.dtype == torch.float32
    assert out.matrix.device == cs.id.device


# --------- gate leaves ---------------------------------------------------


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (2, 0)])
def test_none_gate_empty_inputs_return_zero_shaped_per_pair(n, m):
    cs = _cs(n)
    ds = _ds(m)
    out = NoneGate()(cs, ds, FrameContext.make(0))
    assert out.kind == "per_pair"
    assert tuple(out.mask.shape) == (n, m)


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (2, 0)])
def test_class_gate_empty_inputs_return_zero_shaped_per_pair(n, m):
    cs = _cs(n, klass=torch.zeros(n, dtype=torch.int64))
    ds = _ds(m, klass=torch.zeros(m, dtype=torch.int64))
    out = ClassGate(field="klass")(cs, ds, FrameContext.make(0))
    assert out.kind == "per_pair"
    assert tuple(out.mask.shape) == (n, m)


@pytest.mark.parametrize("m", [0, 2])
def test_score_gate_empty_or_present_ds_returns_per_ds(m):
    cs = _cs(0)  # ScoreGate doesn't read cs
    ds = _ds(m, score=torch.ones(m, dtype=torch.float32))
    out = ScoreGate(field="score", threshold=0.5)(cs, ds, FrameContext.make(0))
    assert out.kind == "per_ds"
    assert tuple(out.mask.shape) == (m,)


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 2), (2, 0)])
@pytest.mark.parametrize(("gate_cls", "d"), [(SpatialGate2D, 2), (SpatialGate3D, 3)])
def test_spatial_gates_empty_inputs_return_zero_shaped_per_pair(gate_cls, d, n, m):
    cs = _cs(n, pos=torch.zeros(n, d))
    ds = _ds(m, pos=torch.zeros(m, d))
    out = gate_cls(field="pos", max_dist=1.0)(cs, ds, FrameContext.make(0))
    assert out.kind == "per_pair"
    assert tuple(out.mask.shape) == (n, m)


# --------- Gate algebra over empty inputs -------------------------------


def test_gate_combine_per_pair_on_zero_rows_or_cols():
    """`Gate.combine(PerPair, PerPair)` over zero-row / zero-col masks
    preserves the empty shape rather than crashing on `bool &` broadcast."""
    a = Gate.PerPair(mask=torch.zeros(0, 3, dtype=torch.bool))
    b = Gate.PerPair(mask=torch.zeros(0, 3, dtype=torch.bool))
    out = Gate.combine(a, b)
    assert out.kind == "per_pair"
    assert tuple(out.mask.shape) == (0, 3)
