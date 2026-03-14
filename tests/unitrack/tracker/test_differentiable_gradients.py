"""Gradient-correctness tests for the differentiable=True path.

These pin the *functional* claim — that ``differentiable=True`` actually
threads gradients through the soft path — separately from the structural
swap-registry tests in ``test_differentiable.py``.
"""

from __future__ import annotations

import torch
from unitrack.assignment import Associate, Jonker, SoftAssignment
from unitrack.assignment._soft import sinkhorn_log_plan
from unitrack.costs import Cosine
from unitrack.data import Detections, FrameContext, TensorSpec
from unitrack.gates.soft import SoftMotionGate
from unitrack.lifecycle import IncludeAll, NoLifecycle
from unitrack.pipeline import Pipe
from unitrack.states import FromDetectionField, Identity, Replace, State
from unitrack.tracker import Tracker


def test_sinkhorn_log_plan_gradcheck():
    """Finite-difference check on the log-domain Sinkhorn iteration."""
    torch.manual_seed(0)
    cost = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

    def fn(c: torch.Tensor) -> torch.Tensor:
        return sinkhorn_log_plan(c, epsilon=0.5, num_iter=30)

    assert torch.autograd.gradcheck(fn, (cost,), eps=1e-6, atol=1e-4)


def test_sinkhorn_log_plan_masked_row_finite_plan():
    """A fully-`inf` row must not poison the rest of the plan with NaN."""
    cost = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [float("inf"), float("inf"), float("inf")],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    log_plan = sinkhorn_log_plan(cost, epsilon=0.5, num_iter=30)
    plan = log_plan.exp()
    # Bad row → zero transport mass; other rows → finite, sum to ~1/N each.
    assert torch.isfinite(plan).all()
    assert plan[1].abs().sum() < 1e-6
    assert torch.isfinite(plan[0]).all()
    assert plan[0].sum() > 0
    assert torch.isfinite(plan[2]).all()
    assert plan[2].sum() > 0


def test_associate_attaches_soft_plan():
    """SoftAssignment path must propagate `soft_plan` through MatchOutcome."""
    from unitrack.data import CostExpression

    cs = _two_kernel_tracklets()
    ds = _two_kernel_detections()
    ctx = FrameContext.make(0)
    assoc = Associate(SoftAssignment(threshold=2.0, epsilon=0.1, num_iter=30))
    cost = CostExpression.from_matrix(torch.tensor([[0.1, 0.9], [0.9, 0.1]]))
    out = assoc(cs, ds, ctx, cost=cost)
    assert out.soft_plan is not None
    assert out.soft_plan.shape == (2, 2)
    # Row-stochastic up to the marginal scale 1/N.
    row_sums = out.soft_plan.sum(dim=1)
    assert torch.allclose(row_sums, torch.full((2,), 0.5), atol=1e-4)


def test_tracker_backward_propagates_to_detections():
    """End-to-end: a loss on the post-step snapshot reaches `detections.kernel`."""
    tr = Tracker(
        root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
        states={
            "kernel": State(
                schema=TensorSpec(shape=(4,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
        },
        lifecycle=NoLifecycle(),
        visibility=IncludeAll(),
        differentiable=True,
    )
    snap = tr.empty_snapshot()
    kernel = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], requires_grad=True
    )
    ds = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=kernel,
        batch_size=[2],
    )
    res0 = tr.step(snap, ds, FrameContext.make(0), next_id=1)
    # Second step so there's a non-empty cost matrix and SoftReplace blends.
    ds2 = Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=kernel * 2.0,
        batch_size=[2],
    )
    res1 = tr.step(res0.snapshot, ds2, FrameContext.make(1), next_id=res0.next_id)
    loss = res1.snapshot.kernel.sum()
    loss.backward()
    assert kernel.grad is not None
    assert torch.isfinite(kernel.grad).all()
    assert kernel.grad.abs().sum() > 0


def test_soft_motion_gate_gradient_wrt_state():
    """SoftMotionGate.chi2/temperature must propagate gradients to cs mean/cov."""
    torch.manual_seed(0)
    cs = _two_kernel_tracklets()
    ds = _two_kernel_detections()
    p = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32, requires_grad=True)
    p_cov = (torch.eye(2).expand(2, 2, 2).contiguous() * 0.5).requires_grad_()
    cs = cs.set("p", p).set("p_cov", p_cov)
    ds = ds.set("p", torch.tensor([[0.1, 0.0], [1.1, 0.0]], dtype=torch.float32))
    gate = SoftMotionGate("p", "p_cov", temperature=2.0)
    out = gate(cs, ds, FrameContext.make(0))
    assert out.kind == "cost_bias"  # type: ignore[attr-defined]
    # CostBias.matrix is (N, M); backward through it.
    out.matrix.sum().backward()  # type: ignore[union-attr]
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0
    assert p_cov.grad is not None
    assert torch.isfinite(p_cov.grad).all()
    assert p_cov.grad.abs().sum() > 0


def _two_kernel_tracklets():
    from unitrack.data import Tracklets

    return Tracklets(
        id=torch.arange(2, dtype=torch.int64),
        status=torch.ones(2, dtype=torch.int8),
        hits=torch.ones(2, dtype=torch.int32),
        time_since_update=torch.zeros(2, dtype=torch.int32),
        age=torch.ones(2, dtype=torch.int32),
        frame_started=torch.zeros(2, dtype=torch.int32),
        frame_last_seen=torch.zeros(2, dtype=torch.int32),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )


def _two_kernel_detections():
    return Detections(
        index=torch.arange(2, dtype=torch.int64),
        kernel=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch_size=[2],
    )
