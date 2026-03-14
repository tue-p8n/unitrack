import torch
from unitrack.benchmarks.hota.learned_modules import Fuser, Propagator


def test_propagator_output_shape_and_unit_norm():
    torch.manual_seed(0)
    dim = 16
    prop = Propagator(dim)
    x = torch.randn(5, dim)
    out = prop(x, 1.0)
    assert out.shape == (5, dim)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_propagator_accepts_dt_argument():
    # LearnedProcess calls module(value, dt); the signature must accept dt.
    prop = Propagator(8)
    x = torch.randn(3, 8)
    out = prop(x, 0.5)
    assert out.shape == (3, 8)


def test_propagator_gradient_flows():
    prop = Propagator(8)
    x = torch.randn(4, 8, requires_grad=True)
    loss = prop(x, 1.0).pow(2).sum()
    loss.backward()
    grads = [p.grad for p in prop.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_fuser_output_shape_and_unit_norm():
    torch.manual_seed(0)
    dim = 16
    fuse = Fuser(dim)
    track = torch.randn(5, dim)
    meas = torch.randn(5, dim)
    out = fuse(track, meas)
    assert out.shape == (5, dim)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_fuser_gradient_flows():
    fuse = Fuser(8)
    track = torch.randn(4, 8, requires_grad=True)
    meas = torch.randn(4, 8, requires_grad=True)
    loss = fuse(track, meas).pow(2).sum()
    loss.backward()
    grads = [p.grad for p in fuse.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
