"""
Build the embedding-filter demonstration notebooks.

Six notebooks, one per recursive estimator for appearance/kernel
embeddings, each driving a *real* ``unitrack.Tracker`` so the demo doubles
as proof the library supports the method. Edit the ``NB_*`` lists and re-run:

    python docs/5.notebooks/embedding_filters/_build.py
"""

from __future__ import annotations

import json
import pathlib
import textwrap

HERE = pathlib.Path(__file__).resolve().parent


def _md(text: str) -> tuple[str, str]:
    return ("md", textwrap.dedent(text).strip("\n") + "\n")


def _py(text: str) -> tuple[str, str]:
    return ("py", textwrap.dedent(text).strip("\n") + "\n")


def _cell(idx: int, kind: str, source: str) -> dict:
    cid = f"emb-{idx:03d}"
    if kind == "md":
        return {
            "cell_type": "markdown",
            "id": cid,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    return {
        "cell_type": "code",
        "id": cid,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_notebook(path: pathlib.Path, cells: list[tuple[str, str]]) -> None:
    nb = {
        "cells": [_cell(i, k, s) for i, (k, s) in enumerate(cells)],
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))
    print(f"  wrote {path.relative_to(HERE.parent.parent.parent)}")


# Shared preamble: imports + a drifting-embedding clip + a tracker runner.
# Duplicated into every notebook so each is self-contained.
SETUP = _py("""
    import torch
    import matplotlib.pyplot as plt

    import unitrack
    from unitrack.assignment import Associate, Jonker
    from unitrack.costs import Cosine
    from unitrack.data import Detections, FrameContext, TensorSpec
    from unitrack.lifecycle import IncludeAll, NoLifecycle
    from unitrack.pipeline import Pipe

    torch.manual_seed(0)

    D = 16          # embedding dimensionality (256+ in practice; 16 plots fast)
    T = 44          # frames

    def make_clip(noise=0.15, seed=0, switch=None):
        \"\"\"
        A unit embedding that rotates slowly in the (e0, e1) plane plus
        per-frame noise in all D dims. Rotating in a known plane means
        projecting onto dims (0, 1) shows the true path as a circle arc.
        `switch` optionally rotates the plane mid-clip (an appearance change).
        \"\"\"
        g = torch.Generator().manual_seed(seed)
        t = torch.arange(T).float()
        theta = 0.10 * t
        truth = torch.zeros(T, D)
        truth[:, 0] = torch.cos(theta)
        truth[:, 1] = torch.sin(theta)
        if switch is not None:
            # after `switch`, swap appearance into the (e2, e3) plane.
            truth[switch:, :] = 0.0
            truth[switch:, 2] = torch.cos(theta[switch:])
            truth[switch:, 3] = torch.sin(theta[switch:])
        obs = truth + noise * torch.randn(T, D, generator=g)
        obs = torch.nn.functional.normalize(obs, dim=-1)
        dets = [Detections(index=torch.tensor([0]), emb=obs[k:k + 1].clone(),
                           batch_size=[1]) for k in range(T)]
        return t, truth, obs, dets

    def run(tracker, dets, fields=("emb",)):
        \"\"\"Run a single-object clip through a real Tracker; collect snapshot fields.\"\"\"
        ms = unitrack.MultiStream(tracker)
        rec = {f: [] for f in fields}
        for k, d in enumerate(dets):
            ctx = FrameContext.make(frame_idx=k, delta=1.0, fps=1.0, stream_key=0)
            res = ms.step(stream_key=0, detections=d, ctx=ctx)
            for f in fields:
                rec[f].append(getattr(res.snapshot, f)[0].clone())
        return {f: torch.stack(v) for f, v in rec.items()}

    def cos_to_truth(est, truth):
        e = torch.nn.functional.normalize(est, dim=-1)
        u = torch.nn.functional.normalize(truth, dim=-1)
        return (e * u).sum(-1)

    t, truth, obs, dets = make_clip()
    print(f"clip: {T} frames, D={D}; raw-detection mean cosine-to-truth "
          f"= {cos_to_truth(obs, truth).mean():.3f}")
""")


def _plot_track_cell(extra_lines: str = "") -> tuple[str, str]:
    return _py(f"""
        fig, (axp, axc) = plt.subplots(1, 2, figsize=(12, 4))
        axp.plot(truth[:, 0], truth[:, 1], "-", color="0.55", lw=2, label="truth")
        axp.scatter(obs[:, 0], obs[:, 1], marker="x", color="tab:red", s=22,
                    alpha=0.5, label="noisy detections")
        axp.plot(rec["emb"][:, 0], rec["emb"][:, 1], "o-", color="tab:blue",
                 ms=3, label="filtered estimate")
        axp.set_title("Embedding trajectory, projected to (dim 0, dim 1)")
        axp.set_xlabel("dim 0"); axp.set_ylabel("dim 1")
        axp.legend(fontsize=8); axp.grid(alpha=0.3); axp.set_aspect("equal")

        axc.plot(t, cos_to_truth(obs, truth), color="tab:red", alpha=0.6,
                 label="raw detections")
        axc.plot(t, cos_to_truth(rec["emb"], truth), color="tab:blue",
                 label="filtered")
        axc.set_title("Cosine similarity to ground truth (higher = better)")
        axc.set_xlabel("frame"); axc.set_ylabel("cosine"); axc.legend(fontsize=8)
        axc.grid(alpha=0.3)
        {extra_lines}
        plt.tight_layout(); plt.show()
        print(f"raw mean cos = {{cos_to_truth(obs, truth).mean():.3f}}   "
              f"filtered mean cos = {{cos_to_truth(rec['emb'], truth).mean():.3f}}")
    """)


# ---------------------------------------------------------------------------
# 01 — EMA
# ---------------------------------------------------------------------------
NB_EMA = [
    _md("""
        # Embedding filters 1 — Exponential moving average (EMA)

        The workhorse for appearance/ReID embeddings. An EMA blend
        ``e <- rho * e + (1 - rho) * z`` is a steady-state scalar Kalman
        filter: the gain ``(1 - rho)`` is constant rather than derived from a
        covariance. It is cheap (``O(D)``), stable, and is what DeepSORT,
        FairMOT and BoT-SORT use to smooth the per-track feature.

        unitrack ships it as `EMAFuse` (the blend, an `Observation`) paired
        with `EMATrack` (a no-op `Process`). Here we plug it into a real
        `Tracker` and watch it denoise a drifting embedding.
    """),
    SETUP,
    _py("""
        from unitrack.states import EMAFuse, EMATrack, FromDetectionField, State

        def ema_tracker(rho):
            states = {
                "emb": State(
                    schema=TensorSpec(shape=(D,), dtype=torch.float32),
                    process=EMATrack("emb"),
                    observation=EMAFuse("emb", rho=rho),
                    init=FromDetectionField("emb"),
                ),
            }
            return unitrack.Tracker(
                root=Pipe(cost=Cosine("emb"), assoc=Associate(Jonker(threshold=0.6))),
                states=states, lifecycle=NoLifecycle(), visibility=IncludeAll(),
            )

        rec = run(ema_tracker(rho=0.8), dets)
    """),
    _plot_track_cell(),
    _md("""
        ## The bias-variance knob

        `rho` trades responsiveness for smoothness: high `rho` rejects noise
        but lags the drift; low `rho` follows the drift but keeps more noise.
        There is a sweet spot — the same trade a Kalman filter makes
        automatically through its gain (next notebook).
    """),
    _py("""
        fig, ax = plt.subplots(figsize=(7, 4))
        for rho in (0.5, 0.8, 0.95):
            r = run(ema_tracker(rho), dets)
            ax.plot(t, cos_to_truth(r["emb"], truth), label=f"rho={rho} "
                    f"(mean {cos_to_truth(r['emb'], truth).mean():.3f})")
        ax.plot(t, cos_to_truth(obs, truth), color="0.6", ls=":", label="raw")
        ax.set_title("EMA rho sweep — cosine to truth")
        ax.set_xlabel("frame"); ax.set_ylabel("cosine")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    """),
    _md("""
        **Takeaway.** EMA is the cheap, robust default. It carries no
        uncertainty, so it cannot tell you *how sure* it is — for that, use a
        Kalman / information / vMF filter (notebooks 2-4), or a gallery
        (notebook 5) when one vector per track is too little memory.
    """),
]

# ---------------------------------------------------------------------------
# 02 — Diagonal Kalman
# ---------------------------------------------------------------------------
NB_KALMAN = [
    _md("""
        # Embedding filters 2 — Diagonal Kalman

        A Kalman filter on an embedding with a **random-walk** model
        (`F = H = I`) and *diagonal* process/measurement noise. With diagonal
        `Q`, `R` and a diagonal initial covariance the per-dimension filters
        are independent, so the cost stays `O(D)` while — unlike EMA — the
        filter carries a real uncertainty that *adapts* the gain: it trusts
        measurements more while uncertain (early frames) and less once
        confident.

        We build it from `KalmanLinear` + `KalmanUpdate` (the same primitives
        as the motion Kalman, just with identity dynamics) and track the
        per-dimension variance shrinking.
    """),
    SETUP,
    _py("""
        from unitrack.states import FromDetectionField, State
        from unitrack.states import EyeInitializer, NoopObservation, NoopProcess
        from unitrack.states.kalman import KalmanLinear, KalmanUpdate

        def diagonal_kalman_tracker(q=0.02, r=0.2, init_var=1.0):
            eye = torch.eye(D)
            states = {
                "emb": State(
                    schema=TensorSpec(shape=(D,), dtype=torch.float32),
                    process=KalmanLinear(field="emb", F=eye, H=eye,
                                         Q=eye * q, R=eye * r),
                    observation=KalmanUpdate(field="emb", cov_field="emb_cov",
                                             H=eye, R=eye * r),
                    init=FromDetectionField("emb"),
                ),
                "emb_cov": State(
                    schema=TensorSpec(shape=(D, D), dtype=torch.float32),
                    process=NoopProcess(), observation=NoopObservation(),
                    init=EyeInitializer(dim=D, scale=init_var),
                ),
            }
            return unitrack.Tracker(
                root=Pipe(cost=Cosine("emb"), assoc=Associate(Jonker(threshold=0.6))),
                states=states, lifecycle=NoLifecycle(), visibility=IncludeAll(),
            )

        rec = run(diagonal_kalman_tracker(), dets, fields=("emb", "emb_cov"))
    """),
    _plot_track_cell(),
    _md("""
        ## Adaptive gain via the covariance

        The variance starts high (the filter knows it is guessing) and
        collapses as evidence accumulates — that shrinking is what makes the
        Kalman gain large early and small later, the behaviour EMA's fixed
        `rho` only approximates. The covariance stays diagonal throughout
        (diagonal `Q`/`R`/`P0`), so this is genuinely `O(D)`.
    """),
    _py("""
        var = torch.stack([rec["emb_cov"][k].diagonal() for k in range(T)])  # (T, D)
        offdiag = torch.stack([
            (rec["emb_cov"][k] - torch.diag(rec["emb_cov"][k].diagonal())).abs().max()
            for k in range(T)])
        fig, ax = plt.subplots(figsize=(7, 4))
        for j in range(0, D, 4):
            ax.plot(t, var[:, j], label=f"Var(dim {j})")
        ax.set_title(f"Per-dimension variance shrinks under measurements\\n"
                     f"(max off-diagonal entry over the run: {offdiag.max():.1e} "
                     f"-> stays diagonal)")
        ax.set_xlabel("frame"); ax.set_ylabel("variance")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    """),
    _md("""
        **Takeaway.** A diagonal Kalman gives EMA-like cost with principled,
        adaptive, *measurable* uncertainty. When you need the full covariance
        but `D` is large, the information filter and the EnKF (notebook 4)
        scale better.
    """),
]

# ---------------------------------------------------------------------------
# 03 — vMF / directional
# ---------------------------------------------------------------------------
NB_VMF = [
    _md("""
        # Embedding filters 3 — von Mises-Fisher (directional)

        ReID embeddings are compared by cosine similarity, so they really
        live on the unit **sphere**, not in flat space — a Euclidean Kalman
        filter is the wrong geometry. The von Mises-Fisher distribution is
        the sphere's Gaussian: a mean direction `mu` (unit) and a
        concentration `kappa` (certainty). Its mean direction has a conjugate
        vMF prior, so fusing an observation is an exact, closed-form update —
        the resultant of the concentration-weighted directions.

        `vmf_state_entries` wires this up: match on `mu` with the ordinary
        `Cosine` cost, and read `kappa` as a live confidence.
    """),
    SETUP,
    _py("""
        from unitrack.states import vmf_state_entries

        tracker = unitrack.Tracker(
            root=Pipe(cost=Cosine("emb"), assoc=Associate(Jonker(threshold=0.6))),
            states=vmf_state_entries("emb", dim=D, init_kappa=5.0,
                                     kappa_obs=12.0, tau=8.0),
            lifecycle=NoLifecycle(), visibility=IncludeAll(),
        )
        rec = run(tracker, dets, fields=("emb", "emb_kappa"))
    """),
    _plot_track_cell(),
    _md("""
        ## Concentration = confidence

        The update sharpens the belief (kappa grows) as consistent
        detections accumulate; the predict step lets it decay, so a track
        that stops being seen becomes *less* certain of its appearance — a
        principled, geometry-aware confidence you can feed to a gate.
    """),
    _py("""
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, rec["emb_kappa"], "o-", color="tab:green", ms=3)
        ax.set_title("vMF concentration kappa over the clip (rises as evidence builds)")
        ax.set_xlabel("frame"); ax.set_ylabel("kappa")
        ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
        print(f"mu stays unit: max |‖mu‖ - 1| = "
              f"{(rec['emb'].norm(dim=-1) - 1).abs().max():.2e}")
    """),
    _md("""
        **Takeaway.** vMF is the *right* filter for normalised embeddings:
        the estimate provably stays on the sphere and `kappa` is an honest
        directional confidence. It is the directional analogue of the Kalman
        filter in notebook 2.
    """),
]

# ---------------------------------------------------------------------------
# 04 — EnKF + information filter
# ---------------------------------------------------------------------------
NB_ENKF = [
    _md("""
        # Embedding filters 4 — Ensemble Kalman & information filters

        Two ways to carry a *full* covariance on a high-dimensional embedding
        without the `O(D^2)`-`O(D^3)` cost of a dense Kalman filter:

        - **Information filter** — the exact Kalman dual. It stores the
          *inverse* covariance, which turns the measurement update into a
          plain addition. Its posterior is identical to the Kalman filter's
          (the library test asserts this to 1e-4).
        - **Ensemble Kalman filter (EnKF)** — represents the covariance
          *implicitly* by an ensemble of sample states. Cost scales with the
          ensemble size, not `D`-squared; this is the method built for very
          high-dimensional filtering. unitrack uses a deterministic ETKF
          transform, so the step is reproducible.
    """),
    SETUP,
    _py("""
        from unitrack.states import enkf_state_entries, information_state_entries

        def tracker_for(states):
            return unitrack.Tracker(
                root=Pipe(cost=Cosine("emb"), assoc=Associate(Jonker(threshold=0.6))),
                states=states, lifecycle=NoLifecycle(), visibility=IncludeAll(),
            )

        info = run(tracker_for(information_state_entries("emb", dim=D, q=0.02, r=0.2)),
                   dets, fields=("emb", "emb_infomat"))
        enkf = run(tracker_for(enkf_state_entries("emb", dim=D, ensemble_size=24,
                                                  q=0.02, r=0.2, init_std=0.4)),
                   dets, fields=("emb", "emb_ensemble"))
    """),
    _py("""
        # Both track the drift; compare to raw.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, cos_to_truth(obs, truth), color="0.6", ls=":", label="raw detections")
        ax.plot(t, cos_to_truth(info["emb"], truth), color="tab:blue",
                label=f"information filter ({cos_to_truth(info['emb'], truth).mean():.3f})")
        ax.plot(t, cos_to_truth(enkf["emb"], truth), color="tab:orange",
                label=f"EnKF ({cos_to_truth(enkf['emb'], truth).mean():.3f})")
        ax.set_title("Information filter vs EnKF — cosine to truth")
        ax.set_xlabel("frame"); ax.set_ylabel("cosine"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    """),
    _md("""
        ## Two views of the same shrinking uncertainty

        Left: the EnKF ensemble, projected to 2-D — the cloud of members
        contracts as measurements arrive (that contraction *is* the
        covariance update; no `D`-by-`D` matrix is ever formed). Right: the
        information filter's covariance trace (`tr(Y^{-1})`) falling in
        lockstep — the explicit dual of the same belief.
    """),
    _py("""
        fig, (axe, axt) = plt.subplots(1, 2, figsize=(12, 4))
        cmap = plt.get_cmap("viridis")
        for k in range(0, T, 6):
            members = enkf["emb_ensemble"][k]  # (E, D)
            axe.scatter(members[:, 0], members[:, 1], s=14, color=cmap(k / T),
                        alpha=0.6)
        axe.plot(enkf["emb"][:, 0], enkf["emb"][:, 1], "-", color="0.3", lw=1,
                 label="ensemble mean")
        axe.set_title("EnKF ensemble (proj. to dims 0,1) contracts over time\\n"
                      "(purple = early, yellow = late)")
        axe.set_xlabel("dim 0"); axe.set_ylabel("dim 1"); axe.legend(fontsize=8)
        axe.grid(alpha=0.3)

        cov_trace = torch.stack([
            torch.linalg.inv(info["emb_infomat"][k]).diagonal().sum() for k in range(T)])
        ens_trace = torch.stack([enkf["emb_ensemble"][k].var(dim=0).sum() for k in range(T)])
        axt.plot(t, cov_trace, color="tab:blue", label="information filter  tr(P)")
        axt.plot(t, ens_trace, color="tab:orange", label="EnKF ensemble spread")
        axt.set_title("Total uncertainty falls under measurements")
        axt.set_xlabel("frame"); axt.set_ylabel("trace / spread"); axt.legend(fontsize=8)
        axt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    """),
    _md("""
        **Takeaway.** Use the **information filter** when you want the exact
        Gaussian posterior and `D` is moderate (its update is a cheap
        addition, ideal for fusing many cues). Use the **EnKF** when `D` is
        large enough that a dense covariance is impractical — the ensemble
        buys you covariance information at a cost set by the ensemble size.
    """),
]

# ---------------------------------------------------------------------------
# 05 — Gallery (memory bank) + learned (MOTR-style)
# ---------------------------------------------------------------------------
NB_GALLERY = [
    _md("""
        # Embedding filters 5 — Memory bank & learned propagation

        Two answers to "one filtered vector per track is not enough memory":

        1. **Gallery (feature bank)** — store the last `K` embeddings per
           track and match a detection against the *best* of them
           (DeepSORT / MeMOT). One good past view re-associates an object
           whose current appearance has drifted.
        2. **Learned propagation (MOTR-style)** — let a trained module
           propagate the track embedding frame to frame, instead of a
           hand-written filter. unitrack exposes this as `LearnedProcess` /
           `LearnedObservation` hooks around any `nn.Module`.
    """),
    SETUP,
    _md("""
        ## Gallery beats a single embedding across an appearance switch

        We build a two-object clip where object A's appearance **switches**
        partway (a viewpoint change), then later a detection resembling its
        *old* appearance returns. A single-embedding matcher (`Cosine` on the
        latest vector) has forgotten the old look; a `GalleryCost` that keeps
        `K` past views still recognises it.
    """),
    _py("""
        from unitrack.costs import GalleryCost
        from unitrack.states import gallery_state_entries

        # Tracklet A: appearance in plane (e0,e1). A returning query that looks
        # like A's *early* appearance should still match A's gallery.
        torch.manual_seed(1)
        early = torch.zeros(1, D); early[0, 0] = 1.0
        late = torch.zeros(1, D); late[0, 2] = 1.0          # A after a switch
        views = [early, 0.5 * (early + late), late]          # A's history, oldest first

        gstates = gallery_state_entries("emb", dim=D, capacity=6)
        from unitrack.states import FromDetectionField, NoopProcess, State, Replace
        single = {"emb": State(schema=TensorSpec(shape=(D,), dtype=torch.float32),
                               process=NoopProcess(), observation=Replace("emb"),
                               init=FromDetectionField("emb"))}

        def feed(states, cost):
            tr = unitrack.Tracker(root=Pipe(cost=cost, assoc=Associate(Jonker(threshold=0.6))),
                                  states=states, lifecycle=NoLifecycle(), visibility=IncludeAll())
            ms = unitrack.MultiStream(tr)
            snap = None
            for k, v in enumerate(views):
                res = ms.step(stream_key=0, detections=Detections(index=torch.tensor([0]),
                              emb=torch.nn.functional.normalize(v, dim=-1), batch_size=[1]),
                              ctx=FrameContext.make(k, stream_key=0))
                snap = res.snapshot
            return snap

        # After observing A's history, a query resembling the EARLY view returns.
        query = torch.nn.functional.normalize(early + 0.05 * torch.randn(1, D), dim=-1)
        snap_g = feed(dict(gstates), GalleryCost("emb_gallery", "emb_count", "emb", reduce="max"))
        snap_s = feed(dict(single), Cosine("emb"))
        ds_q = Detections(index=torch.tensor([0]), emb=query, batch_size=[1])
        ctx = FrameContext.make(99)
        cost_gallery = GalleryCost("emb_gallery", "emb_count", "emb", reduce="max")(
            snap_g, ds_q, ctx).matrix.item()
        cost_single = Cosine("emb")(snap_s, ds_q, ctx).matrix.item()
        n_views = int(snap_g.emb_count[0].item())
        print("returning early-appearance query:")
        print(f"  single-embedding cost (latest view only): {cost_single:.3f}  (high -> missed)")
        print(f"  gallery cost (best of {n_views} stored views):  "
              f"{cost_gallery:.3f}  (low -> re-associated)")
    """),
    _py("""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["single\\n(latest)", "gallery\\n(best view)"],
               [cost_single, cost_gallery],
               color=["tab:red", "tab:green"])
        ax.axhline(0.4, color="0.5", ls="--", label="example gate threshold")
        ax.set_ylabel("matching cost (lower = match)")
        ax.set_title("Re-associating a returning appearance")
        ax.legend(fontsize=8)
        plt.tight_layout(); plt.show()
    """),
    _md("""
        ## Learned propagation, trained MOTR-style (through the association)

        MOTR/MeMOTR do not regress the track query toward a target embedding;
        they train it **through the data association** — the query is
        propagated, matched against the next frame's detections, and the loss
        rewards keeping each identity bound to its detection. The gradient
        flows back through the (differentiable) matcher into the propagation
        module. We do exactly that here, using unitrack's differentiable
        `sinkhorn_log_plan` as the soft matcher.

        The scenario is built so propagation is *necessary*: four identities
        sit on a circle in feature space and all rotate fast (~54° / frame).
        A track that does not anticipate the rotation lags past the midpoint
        to its neighbour, so a "match-the-last-embedding" tracker
        systematically assigns the wrong identity. Only a module that learns
        to rotate the query forward keeps the association correct — and the
        only supervision is the association itself.
    """),
    _py("""
        import torch.nn as nn
        from unitrack.assignment import sinkhorn_log_plan
        from unitrack.states import (
            FromDetectionField, Identity, LearnedObservation, LearnedProcess,
            Replace, State,
        )

        N, ROT_T, OMEGA, NZ = 4, 10, 0.95, 0.10   # ids, frames, rad/frame, noise

        def rot_clip(seed, *, shuffle=True):
            \"\"\"Four embeddings on a circle, rotating fast; shuffled per frame.\"\"\"
            g = torch.Generator().manual_seed(seed)
            phases = torch.arange(N).float() * (2 * torch.pi / N)
            frames, gts = [], []
            for k in range(ROT_T):
                ang = phases + OMEGA * k
                emb = torch.zeros(N, D); emb[:, 0] = torch.cos(ang); emb[:, 1] = torch.sin(ang)
                emb = torch.nn.functional.normalize(emb + NZ * torch.randn(N, D, generator=g), dim=-1)
                order = torch.randperm(N, generator=g) if shuffle else torch.arange(N)
                frames.append(emb[order])
                gt = torch.empty(N, dtype=torch.long); gt[order] = torch.arange(N)
                gts.append(gt)                          # gt[i] = column of identity i
            return frames, gts

        def cosdist(a, b):
            a = torch.nn.functional.normalize(a, dim=-1)
            b = torch.nn.functional.normalize(b, dim=-1)
            return 1.0 - a @ b.T

        class Propagator(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, d))
            def forward(self, x, dt=1.0):
                return torch.nn.functional.normalize(x + self.net(x), dim=-1)

        class Fuser(nn.Module):                          # learned update (observation)
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(2 * d, 64), nn.Tanh(), nn.Linear(64, d))
            def forward(self, track, meas):
                return torch.nn.functional.normalize(
                    track + self.net(torch.cat([track, meas], -1)), dim=-1)

        def rollout(prop, fuse, frames, gts, *, eps=0.05):
            \"\"\"Propagate -> soft-match -> association loss; fuse with gt match (BPTT).\"\"\"
            track = frames[0][gts[0]].clone()           # track i := identity i
            loss = 0.0
            for k in range(1, ROT_T):
                pred = prop(track)
                logP = sinkhorn_log_plan(cosdist(pred, frames[k]), epsilon=eps, num_iter=50)
                loss = loss - logP[torch.arange(N), gts[k]].mean()   # NLL of correct cells
                track = fuse(pred, frames[k][gts[k]])   # teacher-forced identity update
            return loss / (ROT_T - 1)

        @torch.no_grad()
        def assoc_accuracy(propagate, seeds):
            \"\"\"One-step hard-association accuracy (argmax of the soft plan vs gt).\"\"\"
            correct = total = 0
            for s in seeds:
                frames, gts = rot_clip(s)
                for k in range(1, ROT_T):
                    pred = propagate(frames[k - 1][gts[k - 1]])   # from the true prev detection
                    logP = sinkhorn_log_plan(cosdist(pred, frames[k]), epsilon=0.05, num_iter=50)
                    correct += int((logP.argmax(1) == gts[k]).sum()); total += N
            return correct / total
    """),
    _py("""
        prop, fuse = Propagator(D), Fuser(D)
        eval_seeds = list(range(5000, 5016))
        acc_identity = assoc_accuracy(lambda x: x, eval_seeds)          # no propagation
        acc_untrained = assoc_accuracy(prop, eval_seeds)               # random init

        opt = torch.optim.Adam(list(prop.parameters()) + list(fuse.parameters()), lr=3e-3)
        losses = []
        for epoch in range(150):
            loss = rollout(prop, fuse, *rot_clip(epoch))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        acc_trained = assoc_accuracy(prop, eval_seeds)

        print(f"association loss {losses[0]:.3f} -> {losses[-1]:.3f}  "
              f"(optimum = log N = {torch.tensor(float(N)).log():.3f})")
        print(f"association accuracy   no-propagation: {acc_identity:.3f}")
        print(f"                       untrained:      {acc_untrained:.3f}")
        print(f"                       TRAINED:        {acc_trained:.3f}")

        fig, (axl, axb) = plt.subplots(1, 2, figsize=(12, 4))
        axl.plot(losses, color="tab:purple")
        axl.axhline(float(torch.tensor(float(N)).log()), ls="--", color="0.5",
                    label="log N (optimal soft assignment)")
        axl.set_title("Association loss (Sinkhorn NLL) through training")
        axl.set_xlabel("epoch"); axl.set_ylabel("NLL"); axl.legend(fontsize=8)
        axl.grid(alpha=0.3)
        axb.bar(["no\\npropagation", "untrained", "trained"],
                [acc_identity, acc_untrained, acc_trained],
                color=["tab:red", "tab:orange", "tab:green"])
        axb.set_ylim(0, 1.05); axb.set_ylabel("identity-association accuracy")
        axb.set_title("Learned propagation is trained *by* the association")
        plt.tight_layout(); plt.show()
    """),
    _py("""
        # Deploy the trained modules in a REAL unitrack.Tracker (hard Jonker
        # matching) and measure how often each track keeps its own identity.
        def emb_state(process, observation):
            return {"emb": State(schema=TensorSpec(shape=(D,), dtype=torch.float32),
                                 process=process, observation=observation,
                                 init=FromDetectionField("emb"))}

        def deploy_accuracy(states):
            tr = unitrack.Tracker(
                root=Pipe(cost=Cosine("emb"), assoc=Associate(Jonker(threshold=0.95))),
                states=states, lifecycle=NoLifecycle(), visibility=IncludeAll())
            ms = unitrack.MultiStream(tr)
            frames, _ = rot_clip(99, shuffle=False)     # column j == identity j
            correct = total = 0
            for k in range(ROT_T):
                d = Detections(index=torch.arange(N, dtype=torch.int64),
                               emb=frames[k].clone(), batch_size=[N])
                res = ms.step(stream_key=0, detections=d,
                              ctx=FrameContext.make(k, delta=1.0, stream_key=0))
                if k > 0:                               # correct pair is (row i, col i)
                    p = res.match.matched_pairs
                    correct += int((p[:, 0] == p[:, 1]).sum()); total += p.shape[0]
            return correct / max(total, 1)

        with torch.no_grad():
            dep_learned = deploy_accuracy(emb_state(
                LearnedProcess("emb", prop),
                LearnedObservation("emb", "emb", fuse)))
            dep_plain = deploy_accuracy(emb_state(Identity("emb"), Replace("emb")))
        print(f"deployed in a real Tracker (hard matching):")
        print(f"  identity process + replace : {dep_plain:.3f}  (lags the rotation -> swaps)")
        print(f"  learned process + learned fuse: {dep_learned:.3f}  (anticipates -> holds)")
    """),
    _md("""
        **Takeaway.** A **gallery** adds memory the single-vector filters lack
        — decisive across appearance changes and re-identification gaps.
        **Learned propagation** is the MOTR-style option: the propagation and
        update modules are trained end-to-end *through the differentiable
        association*, not by regressing to a target embedding — the only
        supervision is "keep each identity matched". With that signal the
        module learns to anticipate motion that a closed-form filter cannot,
        and it drops into the same `State` interface (`LearnedProcess` /
        `LearnedObservation`) as every other method here.
    """),
]


# ---------------------------------------------------------------------------
# 06 — Summary & side-by-side benchmark
# ---------------------------------------------------------------------------
NB_SUMMARY = [
    _md("""
        # Embedding filters 6 — summary & benchmark

        Seven recursive estimators for a tracker's appearance embedding, all
        behind the same `State` interface. This notebook puts them
        side-by-side on **compute** (FLOPs + wall-clock, and how they scale
        with the embedding dimension `D`), **accuracy** (on two
        complementary tasks), and **effort** (tuning / training), and ends
        with a fail/success cheat-sheet.

        | Method | one-line idea |
        |---|---|
        | EMA | constant-gain blend (steady-state scalar Kalman) |
        | diagonal Kalman | random-walk Kalman, adaptive gain |
        | vMF | recursive Bayes on the unit sphere |
        | information filter | exact Kalman dual; additive fusion |
        | EnKF | covariance via an ensemble (built for large `D`) |
        | gallery | feature bank of the last `K` views |
        | learned | MOTR-style module trained through the association |

        **Methodology.** Compute is measured on the *state update* in
        isolation (the matching cost is shared). Accuracy is measured on two
        tasks because no single number is fair to all: a **denoising** task
        (smooth a drifting embedding) and an **association-under-fast-motion**
        task (keep four rotating identities matched). Each method is set to a
        reasonable operating point — and *finding* that point is itself part
        of the "effort" comparison (vMF needs its decay tuned; the EnKF needs
        enough inflation or its ensemble collapses).
    """),
    _py("""
        import time
        import torch
        import torch.nn as nn
        import matplotlib.pyplot as plt

        import unitrack
        from unitrack.assignment import Associate, Jonker, sinkhorn_log_plan
        from unitrack.costs import Cosine, GalleryCost
        from unitrack.data import (
            Detections, FrameContext, MatchOutcome, TensorSpec, Tracklets,
        )
        from unitrack.lifecycle import IncludeAll, NoLifecycle
        from unitrack.pipeline import Pipe
        from unitrack.states import (
            EMAFuse, EMATrack, FromDetectionField, Identity, LearnedObservation,
            LearnedProcess, NoopObservation, NoopProcess, Replace, State,
            EyeInitializer, vmf_state_entries, gallery_state_entries,
        )
        from unitrack.states.kalman import (
            KalmanLinear, KalmanUpdate, enkf_state_entries, information_state_entries,
        )
        torch.manual_seed(0)

        def _reserved(n):
            return dict(
                id=torch.arange(n), status=torch.full((n,), 1, dtype=torch.int8),
                hits=torch.ones(n, dtype=torch.int32),
                time_since_update=torch.zeros(n, dtype=torch.int32),
                age=torch.ones(n, dtype=torch.int32),
                frame_started=torch.zeros(n, dtype=torch.int32),
                frame_last_seen=torch.zeros(n, dtype=torch.int32))

        def spawn(states, n, dim):
            ds = Detections(index=torch.arange(n), emb=torch.randn(n, dim), batch_size=[n])
            user = {k: st.init(ds, FrameContext.make(0)) for k, st in states.items()}
            return Tracklets(**_reserved(n), **user, batch_size=[n]), ds

        def match_n(n):
            p = torch.stack([torch.arange(n), torch.arange(n)], 1)
            return MatchOutcome(matched_pairs=p,
                tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
                detections_residual_index=torch.zeros(0, dtype=torch.int64),
                per_match_cost=torch.zeros(n), batch_size=[])

        def step_update(states, cs, ds, m, ctx):
            out = cs
            for st in states.values():
                out = st.process(out, ctx)
            for st in states.values():
                out = st.observation(out, ds, m, ctx)
            return out

        def time_us(fn, iters=40, warmup=5):
            for _ in range(warmup):
                fn()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            return (time.perf_counter() - t0) / iters * 1e6
    """),
    _py("""
        # Method registry: a factory (states at dim D) + metadata, each at a
        # reasonable operating point (see the per-method notebooks).
        def st_ema(D):
            return {"emb": State(TensorSpec((D,), torch.float32),
                    EMATrack("emb"), EMAFuse("emb", 0.8), FromDetectionField("emb"))}
        def st_kalman(D):
            e = torch.eye(D)
            return {"emb": State(TensorSpec((D,), torch.float32),
                        KalmanLinear("emb", e, e, e * 0.02, e * 0.2),
                        KalmanUpdate("emb", "emb_cov", e, e * 0.2), FromDetectionField("emb")),
                    "emb_cov": State(TensorSpec((D, D), torch.float32),
                        NoopProcess(), NoopObservation(), EyeInitializer(D, 1.0))}
        def st_vmf(D):
            return vmf_state_entries("emb", dim=D, init_kappa=3.0, kappa_obs=5.0, tau=3.0)
        def st_info(D):
            return information_state_entries("emb", dim=D, q=0.02, r=0.2)
        def st_enkf(D):
            return enkf_state_entries("emb", dim=D, ensemble_size=32, q=1.0, r=0.2, init_std=0.5)
        def st_gallery(D):
            return gallery_state_entries("emb", dim=D, capacity=8)
        def st_learned(D):
            prop = nn.Sequential(nn.Linear(D, 64), nn.Tanh(), nn.Linear(64, D))
            fuse = nn.Sequential(nn.Linear(2 * D, 64), nn.Tanh(), nn.Linear(64, D))
            nrm = torch.nn.functional.normalize
            return {"emb": State(TensorSpec((D,), torch.float32),
                    LearnedProcess("emb", lambda x, dt: nrm(x + prop(x), dim=-1)),
                    LearnedObservation("emb", "emb",
                        lambda tr, me: nrm(tr + fuse(torch.cat([tr, me], -1)), dim=-1)),
                    FromDetectionField("emb"))}

        METHODS = {
            "EMA":          dict(make=st_ema,     flops="2·N·D",            effort=1),
            "diag-Kalman":  dict(make=st_kalman,  flops="~N·D³ (dense)",    effort=2),
            "vMF":          dict(make=st_vmf,     flops="~4·N·D",           effort=3),
            "info-filter":  dict(make=st_info,    flops="~N·D³",            effort=3),
            "EnKF (E=32)":  dict(make=st_enkf,    flops="N·(E²D+E³)",       effort=4),
            "gallery (K=8)": dict(make=st_gallery, flops="N·D  (+K·matching)", effort=2),
            "learned":      dict(make=st_learned, flops="2·N·D·H  (+train)", effort=5),
        }
        print("methods:", list(METHODS))
    """),
    _md("""
        ## 1 — Compute: FLOPs and how they scale with `D`

        The dominant FLOPs of one **state update** (`N` tracks, `D` dims;
        `E` ensemble members, `K` gallery slots, `H` MLP width). The split is
        the whole point: the dense Gaussian filters are `O(D³)` per update
        (a `D×D` solve), which is fine at `D=16` but ruinous at the `D=256+`
        of a real ReID/DETR embedding. EMA, vMF and the learned MLP are
        `O(D)`; the EnKF trades the `D³` for `E²D+E³`, paying off once
        `E ≪ D`.
    """),
    _py("""
        D_BENCH, N_BENCH = 256, 8
        rows = []
        with torch.no_grad():
            for name, meta in METHODS.items():
                st = meta["make"](D_BENCH)
                cs, ds = spawn(st, N_BENCH, D_BENCH)
                m = match_n(N_BENCH)
                ctx = FrameContext.make(1, delta=1.0)
                meta["us256"] = time_us(lambda st=st, cs=cs, ds=ds, m=m, ctx=ctx:
                                        step_update(st, cs, ds, m, ctx))
                rows.append((name, meta["flops"], meta["us256"]))
        print(f"state-update cost at D={D_BENCH}, N={N_BENCH}:")
        for n, f, u in rows:
            print(f"  {n:14s} {f:22s} {u:9.1f} us")

        fig, ax = plt.subplots(figsize=(9, 4))
        names = [r[0] for r in rows]; us = [r[2] for r in rows]
        ax.barh(names, us, color="tab:blue")
        ax.set_xscale("log"); ax.set_xlabel("microseconds / update (log scale)")
        ax.set_title(f"State-update wall-clock at D={D_BENCH} (N={N_BENCH} tracks)")
        for i, u in enumerate(us):
            ax.text(u, i, f" {u:.0f}", va="center", fontsize=8)
        ax.grid(alpha=0.3, axis="x"); plt.tight_layout(); plt.show()
    """),
    _py("""
        # Scaling with D: the O(D^3) filters curve up; the rest stay flat.
        DIMS = [16, 32, 64, 128, 256]
        curves = {name: [] for name in METHODS}
        with torch.no_grad():
            for D in DIMS:
                for name, meta in METHODS.items():
                    st = meta["make"](D); cs, ds = spawn(st, N_BENCH, D); m = match_n(N_BENCH)
                    ctx = FrameContext.make(1, delta=1.0)
                    curves[name].append(time_us(
                        lambda st=st, cs=cs, ds=ds, m=m, ctx=ctx:
                        step_update(st, cs, ds, m, ctx), iters=25))
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, ys in curves.items():
            ax.plot(DIMS, ys, "o-", ms=4, label=name)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("embedding dim D"); ax.set_ylabel("microseconds / update")
        ax.set_title("Update cost vs D — dense Kalman/info are O(D³); EnKF & O(D) methods are not")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        plt.tight_layout(); plt.show()
    """),
    _md("""
        Two honest caveats the benchmark exposes:

        - The shipped **diagonal Kalman** uses the generic dense
          `KalmanLinear`, so it is `O(D³)` here. A diagonal-specialised
          implementation would be `O(D)` — the *math* is cheap, the generic
          *code* is not.
        - The **gallery**'s update is trivial (`O(D)` append); its real cost
          is in *matching*, where `GalleryCost` compares against `K` stored
          views — `K×` an ordinary cosine. We measure that separately:
    """),
    _py("""
        D, N, M, K = 256, 8, 8, 8
        gst = st_gallery(D); gcs, _ = spawn(gst, N, D)
        dets = Detections(index=torch.arange(M), emb=torch.randn(M, D), batch_size=[M])
        ctx = FrameContext.make(0)
        cos = Cosine("emb"); gcost = GalleryCost("emb_gallery", "emb_count", "emb")
        t_cos = time_us(lambda: cos(gcs, dets, ctx))
        t_gal = time_us(lambda: gcost(gcs, dets, ctx))
        print(f"matching {N}x{M} at D={D}:  Cosine {t_cos:.1f} us   "
              f"GalleryCost(K={K}) {t_gal:.1f} us   ({t_gal / t_cos:.1f}x)")
    """),
    _md("""
        ## 2 — Accuracy on two complementary tasks

        **Denoising** (smooth a drifting embedding): the genuine smoothers do
        well; the **gallery** sits at the raw level because it stores views
        rather than averaging them (it is memory, not a smoother), and
        **learned** is not built for this. **Association under fast motion**
        (four identities rotating ~54°/frame): every static filter lags into
        its neighbour and mis-associates; only the **learned** module, trained
        through the differentiable matcher, anticipates the motion.
    """),
    _py("""
        D = 16
        def dn_clip(T=44, noise=0.15, seed=0):
            g = torch.Generator().manual_seed(seed); t = torch.arange(T).float(); th = 0.10 * t
            truth = torch.zeros(T, D); truth[:, 0] = torch.cos(th); truth[:, 1] = torch.sin(th)
            obs = torch.nn.functional.normalize(truth + noise * torch.randn(T, D, generator=g), dim=-1)
            return truth, obs

        def denoise_cos(states):
            \"\"\"Forced-match single-track driver: isolates filter quality from matching.\"\"\"
            truth, obs = dn_clip()
            d0 = Detections(index=torch.tensor([0]), emb=obs[0:1].clone(), batch_size=[1])
            cs = Tracklets(**_reserved(1),
                           **{k: st.init(d0, FrameContext.make(0)) for k, st in states.items()},
                           batch_size=[1])
            m = match_n(1); coss = []
            for k in range(1, len(obs)):
                ctx = FrameContext.make(k, delta=1.0)
                for st in states.values():
                    cs = st.process(cs, ctx)
                ds = Detections(index=torch.tensor([0]), emb=obs[k:k + 1].clone(), batch_size=[1])
                for st in states.values():
                    cs = st.observation(cs, ds, m, ctx)
                coss.append((torch.nn.functional.normalize(cs.emb[0], dim=0) @ truth[k]).item())
            return sum(coss) / len(coss)

        truth, obs = dn_clip()
        raw = (torch.nn.functional.normalize(obs, dim=-1) * truth).sum(-1).mean().item()
        denoise = {}
        with torch.no_grad():
            for name in ["EMA", "diag-Kalman", "vMF", "info-filter", "EnKF (E=32)", "gallery (K=8)"]:
                denoise[name] = denoise_cos(METHODS[name]["make"](D))
        METHODS_denoise = denoise
        fig, ax = plt.subplots(figsize=(9, 4))
        names = list(denoise); vals = [denoise[n] for n in names]
        ax.bar(names, vals, color="tab:green")
        ax.axhline(raw, ls="--", color="0.5", label=f"raw detections ({raw:.3f})")
        ax.set_ylim(0.7, 1.0); ax.set_ylabel("mean cosine to truth")
        ax.set_title("Denoising a drifting embedding (forced-match, D=16)")
        ax.legend(fontsize=8); plt.xticks(rotation=15)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        plt.tight_layout(); plt.show()
    """),
    _py("""
        # Association task: rotate fast; learned anticipates, statics lag.
        N, RT, OM, NZ = 4, 10, 0.95, 0.10
        def rot_clip(seed, shuffle=True):
            g = torch.Generator().manual_seed(seed); ph = torch.arange(N).float() * (2 * torch.pi / N)
            frames, gts = [], []
            for k in range(RT):
                a = ph + OM * k; e = torch.zeros(N, D); e[:, 0] = torch.cos(a); e[:, 1] = torch.sin(a)
                e = torch.nn.functional.normalize(e + NZ * torch.randn(N, D, generator=g), dim=-1)
                order = torch.randperm(N, generator=g) if shuffle else torch.arange(N)
                frames.append(e[order])
                gt = torch.empty(N, dtype=torch.long); gt[order] = torch.arange(N); gts.append(gt)
            return frames, gts

        def assoc_acc(states):
            tr = unitrack.Tracker(root=Pipe(cost=Cosine("emb"), assoc=Associate(Jonker(threshold=0.95))),
                                  states=states, lifecycle=NoLifecycle(), visibility=IncludeAll())
            ms = unitrack.MultiStream(tr); frames, _ = rot_clip(99, shuffle=False)
            c = t = 0
            for k in range(RT):
                d = Detections(index=torch.arange(N), emb=frames[k].clone(), batch_size=[N])
                res = ms.step(stream_key=0, detections=d, ctx=FrameContext.make(k, delta=1.0, stream_key=0))
                if k > 0:
                    p = res.match.matched_pairs; c += int((p[:, 0] == p[:, 1]).sum()); t += p.shape[0]
            return c / max(t, 1)

        # Train the learned propagator/fuser through the soft association (MOTR-style).
        def cosdist(a, b):
            a = torch.nn.functional.normalize(a, dim=-1); b = torch.nn.functional.normalize(b, dim=-1)
            return 1.0 - a @ b.T
        prop = nn.Sequential(nn.Linear(D, 64), nn.Tanh(), nn.Linear(64, D))
        fuse = nn.Sequential(nn.Linear(2 * D, 64), nn.Tanh(), nn.Linear(64, D))
        nrm = torch.nn.functional.normalize
        pf = lambda x: nrm(x + prop(x), dim=-1)
        ff = lambda tr, me: nrm(tr + fuse(torch.cat([tr, me], -1)), dim=-1)
        opt = torch.optim.Adam(list(prop.parameters()) + list(fuse.parameters()), lr=3e-3)
        for epoch in range(150):
            frames, gts = rot_clip(epoch); track = frames[0][gts[0]].clone(); loss = 0.0
            for k in range(1, RT):
                pred = pf(track)
                logP = sinkhorn_log_plan(cosdist(pred, frames[k]), epsilon=0.05, num_iter=50)
                loss = loss - logP[torch.arange(N), gts[k]].mean()
                track = ff(pred, frames[k][gts[k]])
            opt.zero_grad(); (loss / (RT - 1)).backward(); opt.step()

        learned_states = {"emb": State(TensorSpec((D,), torch.float32),
            LearnedProcess("emb", lambda x, dt: pf(x)),
            LearnedObservation("emb", "emb", ff), FromDetectionField("emb"))}

        assoc = {}
        with torch.no_grad():
            for name in ["EMA", "diag-Kalman", "vMF", "info-filter", "EnKF (E=32)", "gallery (K=8)"]:
                assoc[name] = assoc_acc(METHODS[name]["make"](D))
            assoc["learned"] = assoc_acc(learned_states)
        fig, ax = plt.subplots(figsize=(9, 4))
        names = list(assoc); vals = [assoc[n] for n in names]
        cols = ["tab:red" if v < 0.5 else "tab:green" for v in vals]
        ax.bar(names, vals, color=cols)
        ax.set_ylim(0, 1.05); ax.set_ylabel("identity-association accuracy")
        ax.set_title("Association under fast motion — only learned anticipates (D=16)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        plt.xticks(rotation=15); plt.tight_layout(); plt.show()
    """),
    _py("""
        # ---- the side-by-side table ----
        FAILSUCCESS = {
            "EMA":          ("cheap default smoothing", "no uncertainty; reactive"),
            "diag-Kalman":  ("adaptive uncertainty, small D", "dense O(D³) at large D"),
            "vMF":          ("cosine/normalised embeddings", "needs decay tuning"),
            "info-filter":  ("exact posterior; fuse cues", "O(D³); large D"),
            "EnKF (E=32)":  ("full covariance at large D", "collapses w/o inflation"),
            "gallery (K=8)": ("appearance change / re-ID", "no smoothing; K× match"),
            "learned":      ("anticipate complex motion", "needs training data"),
        }
        order = ["EMA", "diag-Kalman", "vMF", "info-filter", "EnKF (E=32)", "gallery (K=8)", "learned"]
        header = ["method", "FLOPs/update", "us@D256", "denoise", "assoc", "effort", "best for", "fails at"]
        cells = []
        for n in order:
            cells.append([
                n, METHODS[n]["flops"], f"{METHODS[n]['us256']:.0f}",
                f"{METHODS_denoise.get(n, float('nan')):.3f}" if n in METHODS_denoise else "—",
                f"{assoc.get(n, float('nan')):.2f}" if n in assoc else "—",
                "★" * METHODS[n]["effort"], FAILSUCCESS[n][0], FAILSUCCESS[n][1],
            ])
        fig, ax = plt.subplots(figsize=(16, 3.2)); ax.axis("off")
        tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="left",
                       colWidths=[0.085, 0.13, 0.06, 0.06, 0.05, 0.07, 0.21, 0.21])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
        for j in range(len(header)):
            tbl[0, j].set_facecolor("#dddddd"); tbl[0, j].set_text_props(weight="bold")
        ax.set_title("Embedding filters, side by side", fontsize=11, pad=12)
        plt.tight_layout(); plt.show()
    """),
    _md("""
        ## How to choose

        - **Default to EMA.** Cheapest, robust, one knob. Reach further only
          when it isn't enough.
        - Want **calibrated uncertainty** (for gating) and `D` is small →
          **diagonal Kalman** or the **information filter** (exact, and the
          IF fuses many cues by addition).
        - Embeddings are **cosine-normalised** and you want a confidence →
          **vMF** (just tune the decay to the drift rate).
        - `D` is **large** and you still want covariance → **EnKF** (give it
          enough inflation), the only `O(D)`-friendly full-covariance option.
        - Appearance **changes / re-ID gaps** → a **gallery**; it does not
          smooth, it remembers.
        - Motion is **complex and you have data** → a **learned** module,
          trained through the association — the only method that anticipates
          rather than reacts, at the cost of a training pipeline.

        No method dominates: the dense filters denoise best but don't scale;
        the EnKF scales but must be inflated; the gallery and learned modules
        win tasks the smoothers structurally cannot. The shared `State`
        interface means you can swap among them — or compose them — without
        touching the rest of the tracker.
    """),
]


NOTEBOOKS = {
    "1.ema.ipynb": NB_EMA,
    "2.kalman_diagonal.ipynb": NB_KALMAN,
    "3.vmf_directional.ipynb": NB_VMF,
    "4.enkf_information.ipynb": NB_ENKF,
    "5.gallery_and_learned.ipynb": NB_GALLERY,
    "6.summary_benchmark.ipynb": NB_SUMMARY,
}


def main() -> None:
    print(f"Writing {len(NOTEBOOKS)} notebooks to {HERE}/")
    for filename, cells in NOTEBOOKS.items():
        write_notebook(HERE / filename, cells)
    print("Done.")


if __name__ == "__main__":
    main()
