import torch
from unitrack.benchmarks.hota.learned_modules import Fuser, Propagator
from unitrack.benchmarks.hota.train_learned import train_step


def _toy_clip(*, dim, seed, n_tracks=2, n_frames=3):
    """
    Fabricate a clip where a track's appearance drifts at a shared velocity.

    Each frame carries, per track, the true detection ``base_k + f*v`` (gt id
    ``k``) AND a stale distractor ``base_k + (f-1)*v`` (gt id ``-1``) sitting
    exactly where the *previous* appearance was. A near-identity Propagator maps
    ``h_pred ~ h ~ d_{f-1}`` straight onto that stale distractor, so at init the
    distractor — not the GT-correspondent — is the argmax and the InfoNCE loss is
    genuinely high. The Propagator must learn the constant shift ``+v`` (a single
    position-independent bias, since ``v`` is shared across tracks) to escape it.
    This is what makes ``last < first`` a real gradient signal, not the old floor.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n_tracks, dim, generator=g)
    v = torch.randn(dim, generator=g)
    v = 0.6 * v / v.norm() * base.norm(dim=-1).mean()  # shared drift
    clip = []
    for f in range(n_frames):
        rows, ids = [], []
        for k in range(n_tracks):
            rows.append(base[k] + f * v)  # true detection of track k
            ids.append(k)
            rows.append(base[k] + (f - 1) * v)  # stale copy = previous appearance
            ids.append(-1)
        emb = torch.nn.functional.normalize(torch.stack(rows), dim=-1)
        clip.append((emb, torch.tensor(ids, dtype=torch.long)))
    return clip


def test_train_step_returns_finite_loss_and_decreases():
    # A few dozen steps of the contrastive multi-frame rollout on a separable
    # (but confusable-at-init) toy clip should drive a finite loss down — the old
    # flat-floor objective could not.
    torch.manual_seed(0)
    dim = 32
    clip = _toy_clip(dim=dim, seed=1)

    prop = Propagator(dim)
    fuse = Fuser(dim)
    opt = torch.optim.Adam([*prop.parameters(), *fuse.parameters()], lr=1e-2)

    first = train_step(prop, fuse, clip, opt)
    assert torch.isfinite(torch.tensor(first))
    # The stale-distractor toy starts genuinely off the floor (the GT next
    # detection is NOT the argmax under near-identity propagation).
    assert first > 0.5
    last = first
    for _ in range(80):
        last = train_step(prop, fuse, clip, opt)
    assert torch.isfinite(torch.tensor(last))
    # Real gradient signal: the loss drops with a clear margin, not the old floor.
    assert last < first - 0.1


def test_train_step_requires_grad_and_trains_both_modules():
    # One step must build a grad-requiring loss and deposit gradients on BOTH the
    # Propagator and the Fuser (the rollout puts the Fuser on the gradient path).
    dim = 16
    clip = _toy_clip(dim=dim, seed=2)
    prop = Propagator(dim)
    fuse = Fuser(dim)
    opt = torch.optim.Adam([*prop.parameters(), *fuse.parameters()], lr=1e-2)

    train_step(prop, fuse, clip, opt)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in prop.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fuse.parameters())


def test_train_step_no_multiframe_track_is_safe():
    # A clip whose only GT track appears in a single frame (and a frame of pure
    # distractors) has no rollout step; the call returns a finite zero loss.
    dim = 8
    prop = Propagator(dim)
    fuse = Fuser(dim)
    opt = torch.optim.Adam([*prop.parameters(), *fuse.parameters()], lr=1e-2)
    g = torch.Generator().manual_seed(3)
    clip = [
        (
            torch.nn.functional.normalize(torch.randn(2, dim, generator=g), dim=-1),
            torch.tensor([0, -1], dtype=torch.long),
        ),
        (
            torch.nn.functional.normalize(torch.randn(2, dim, generator=g), dim=-1),
            torch.tensor([-1, -1], dtype=torch.long),
        ),
    ]
    loss = train_step(prop, fuse, clip, opt)
    assert loss == 0.0
    assert torch.isfinite(torch.tensor(loss))
