"""
Build the unitrack tutorial notebooks.

Each tutorial is defined here as a list of (cell_type, source) tuples
where cell_type is "md" or "py". This script writes valid .ipynb JSON
files into the same directory.

Run with:
    python notebooks/tutorials/_build.py
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


def _cell(kind: str, source: str) -> dict:
    if kind == "md":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_notebook(path: pathlib.Path, cells: list[tuple[str, str]]) -> None:
    nb = {
        "cells": [_cell(k, s) for k, s in cells],
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
    print(f"  wrote {path.relative_to(HERE.parent.parent)}")


# ---------------------------------------------------------------------------
# Notebook 1 — Quickstart
# ---------------------------------------------------------------------------

NB_01_QUICKSTART: list[tuple[str, str]] = [
    _md("""
        # 1. Quickstart — your first tracker

        This is a 10-minute introduction to **unitrack**, the PyTorch-native
        multi-object tracking library. By the end of this notebook you will:

        - understand what a unitrack `Tracker` *is* (a pure step function),
        - build a minimal tracker that matches detections by appearance,
        - run it on a synthetic clip and visualize the resulting tracks.

        Subsequent tutorials drill down into each part of the library:
        the data model (notebook 2), the cost & gate zoos (3), the
        composable pipeline tree (4), state evolution and lifecycle (5),
        and a paper-style end-to-end reproduction (6).
    """),
    _md("""
        ## What is a tracker?

        A tracker associates **detections** (per-frame outputs of an object
        detector) into **tracklets** (per-identity sequences over time).
        unitrack's central abstraction is:

        ```
        Tracker.step(snapshot, detections, ctx, next_id) → (snapshot', match, ids, next_id')
        ```

        - `snapshot` is the **current state** of the tracker — a typed,
          immutable record of every live tracklet's fields (id, status,
          age, kernel embedding, position, mask, …).
        - `detections` is one frame's worth of new observations.
        - `ctx` carries timing and stream-identity metadata.
        - `next_id` is a counter passed in by the caller for assigning new
          tracklet IDs.

        The tracker is a **pure function**. It doesn't mutate its inputs;
        it returns a fresh snapshot. The convenience wrapper
        `MultiStream` holds the snapshot for you, so you don't have to
        thread `next_id` and the snapshot through every call.
    """),
    _py("""
        # The full set of imports we'll use across this notebook.
        import torch
        import matplotlib.pyplot as plt

        import unitrack
        from unitrack.assignment import Associate, Jonker
        from unitrack.costs import Cosine
        from unitrack.data import Detections, FrameContext, TensorSpec
        from unitrack.lifecycle import IncludeAll, NoLifecycle
        from unitrack.pipeline import Pipe
        from unitrack.states import FromDetectionField, Identity, Replace, State

        torch.manual_seed(0)
        plt.rcParams["figure.figsize"] = (6, 4)
    """),
    _md("""
        ## A minimal tracker

        We'll build the simplest meaningful tracker: it matches detections
        across frames by **cosine similarity over a learned kernel embedding**.

        The construction reads top-down like a recipe:

        - **State**: each tracklet carries a `kernel` field (a 4-dim float
          vector). `Identity` means the predict step is a no-op (the
          embedding doesn't drift between frames). `Replace` means matched
          tracklets adopt the new detection's embedding. `FromDetectionField`
          says "when a new tracklet is created, copy this field from the
          detection that spawned it."
        - **Stage tree**: a single `Pipe` that computes a cosine-distance
          cost matrix and hands it to `Associate`, which runs Jonker–Volgenant
          assignment with a 0.5 threshold.
        - **Lifecycle / Visibility**: `NoLifecycle` (no Tentative→Active
          transitions) and `IncludeAll` (every tracklet is visible to the
          caller). The next tutorial introduces the proper lifecycle policy.
    """),
    _py("""
        tracker = unitrack.Tracker(
            root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
            states={
                "kernel": State(
                    schema=TensorSpec(shape=(4,), dtype=torch.float32),
                    process=Identity("kernel"),
                    observation=Replace("kernel"),
                    init=FromDetectionField("kernel"),
                ),
                # `position` isn't used for matching here; we carry it on
                # the snapshot so the visualization below has a 2D point
                # to draw per tracklet.
                "position": State(
                    schema=TensorSpec(shape=(2,), dtype=torch.float32),
                    process=Identity("position"),
                    observation=Replace("position"),
                    init=FromDetectionField("position"),
                ),
            },
            lifecycle=NoLifecycle(),
            visibility=IncludeAll(),
        )

        ms = unitrack.MultiStream(tracker)
        print(tracker)
    """),
    _md("""
        ## A tiny synthetic clip

        We'll generate three "ground-truth" objects, each with a unique
        kernel embedding plus a 2D position that drifts at constant velocity.
        Across frames the detection order is **shuffled** so the tracker
        can't trivially exploit row alignment — it has to use the kernel
        embedding to reassociate.
    """),
    _py("""
        N_FRAMES, N_OBJS, K_DIM = 8, 3, 4

        # Per-identity ground truth.
        kernels = torch.randn(N_OBJS, K_DIM)
        kernels = kernels / kernels.norm(dim=-1, keepdim=True)
        positions = torch.tensor([[20.0, 50.0], [100.0, 30.0], [180.0, 80.0]])
        velocities = torch.tensor([[3.0, 1.0], [-1.0, 2.0], [-2.0, -1.0]])

        clip = []
        gt_per_frame = []
        for k in range(N_FRAMES):
            order = torch.randperm(N_OBJS)
            gt_per_frame.append(order)

            kernel_obs = kernels[order] + 0.02 * torch.randn(N_OBJS, K_DIM)
            kernel_obs = kernel_obs / kernel_obs.norm(dim=-1, keepdim=True)
            pos_obs = positions[order] + k * velocities[order]

            clip.append(
                Detections(
                    index=torch.arange(N_OBJS, dtype=torch.int64),
                    kernel=kernel_obs.float(),
                    position=pos_obs.float(),
                    batch_size=[N_OBJS],
                )
            )
        gt_per_frame = torch.stack(gt_per_frame)
        print(f"Generated {N_FRAMES} frames of {N_OBJS} detections each.")
    """),
    _md("""
        Note `Detections` accepts arbitrary user fields (`kernel`, `position`)
        beyond its single reserved field `index`. The same is true of
        `Tracklets`. unitrack's typed-record story is intentionally
        permissive about user fields so you can plug in whatever your
        detector emits — kernels, masks, depth, classes, scores, …
    """),
    _md("""
        ## What does the clip look like?

        Before we track anything, let's *see* what we just generated —
        the two cues the tracker will rely on:

        - **Left (2D motion)**: each object's ground-truth position over
          the eight frames. This is the world the tracker observes,
          modulo per-frame shuffling.
        - **Right (appearance space)**: the cosine similarity between the
          three identities' kernel embeddings. The near-identity matrix
          (1 on the diagonal, ≈0 off it) is exactly what lets the cosine
          cost tell the objects apart even when their positions cross.
    """),
    _py("""
        fig, (ax_pos, ax_emb) = plt.subplots(1, 2, figsize=(11, 4))
        cmap = plt.get_cmap("tab10")

        # (left) ground-truth motion: where each identity actually is.
        for o in range(N_OBJS):
            track = torch.stack(
                [positions[o] + k * velocities[o] for k in range(N_FRAMES)]
            )
            ax_pos.plot(track[:, 0], track[:, 1], "-", color=cmap(o), alpha=0.4)
            ax_pos.scatter(track[:, 0], track[:, 1], color=cmap(o), s=25,
                           edgecolor="black", linewidth=0.4, label=f"object {o}")
            ax_pos.scatter(track[0, 0], track[0, 1], color=cmap(o), s=160,
                           marker="*", edgecolor="black", zorder=3)
        ax_pos.set_title("Ground-truth motion in 2D (★ = frame 0)")
        ax_pos.set_xlabel("x"); ax_pos.set_ylabel("y")
        ax_pos.legend(fontsize=8); ax_pos.grid(alpha=0.3)

        # (right) appearance space: cosine similarity between identity kernels.
        sim = kernels @ kernels.T
        im = ax_emb.imshow(sim.numpy(), cmap="viridis", vmin=-1, vmax=1)
        ax_emb.set_title("Appearance space: kernel cosine similarity")
        ax_emb.set_xlabel("object"); ax_emb.set_ylabel("object")
        ax_emb.set_xticks(range(N_OBJS)); ax_emb.set_yticks(range(N_OBJS))
        for i in range(N_OBJS):
            for j in range(N_OBJS):
                ax_emb.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                            color="white", fontsize=9)
        fig.colorbar(im, ax=ax_emb, fraction=0.046)
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        ## Run the tracker

        Each step takes one frame's detections and a `FrameContext` (which
        carries the frame index and a delta-t for state evolution). The
        wrapper holds the snapshot internally and threads `next_id` through.
    """),
    _py("""
        all_results = []
        for k, dets in enumerate(clip):
            ctx = FrameContext.make(frame_idx=k, delta=1/15.0, fps=15.0, stream_key=0)
            res = ms.step(stream_key=0, detections=dets, ctx=ctx)
            all_results.append(res)
            print(f"frame {k}: snapshot={res.snapshot.batch_size[0]} live, "
                  f"ids={res.ids.tolist()}")
    """),
    _md("""
        Three tracklets live across all eight frames. Their IDs (1, 2, 3)
        are stable — the tracker correctly matches every shuffled detection
        back to the right identity by cosine similarity on the kernel.
    """),
    _md("""
        ## Visualize

        We'll draw the trajectories color-coded by tracker-assigned ID.
        Each marker is one detection, located at its 2D position; the
        color tells us which tracklet the tracker thinks it belongs to.
    """),
    _py("""
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("tab10")

        for k, (dets, res) in enumerate(zip(clip, all_results)):
            # The snapshot's id field aligns with the matched-then-appended order.
            # For this minimal example we re-derive per-detection IDs by walking
            # the snapshot in detection-index order. (Notebook 6 shows a robust
            # version of this for ground-truth evaluation.)
            ids = res.snapshot.id
            pos = res.snapshot.position
            for n, tid in enumerate(ids):
                ax.scatter(pos[n, 0], pos[n, 1],
                           color=cmap(int(tid) % 10), s=40,
                           edgecolor="black", linewidth=0.5)
                if k == 0:
                    ax.text(pos[n, 0] + 5, pos[n, 1], f"id={int(tid)}",
                            fontsize=9)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Per-detection track IDs over 8 frames (color = tracker ID)")
        ax.grid(alpha=0.3)
        plt.show()
    """),
    _md("""
        Each color forms a coherent trajectory in (x, y) — the tracker
        preserved identity across all 8 frames, despite per-frame
        detection shuffling.

        ## What's next

        - **Notebook 2** unpacks the typed records (`Tracklets`,
          `Detections`, `CostExpression`, `MatchOutcome`, `Gate`) and
          explains *why* unitrack uses immutable snapshots.
        - **Notebook 3** visualizes every cost and gate primitive in the library.
        - **Notebook 4** builds richer pipelines with `Sequential`,
          `Parallel`, `Gated`, `Filter`, and `Iterate`.
        - **Notebook 5** introduces state evolution (Kalman, EMA) and
          lifecycle (Tentative → Active → Lost → Removed).
        - **Notebook 6** assembles a K=2 cascaded canonical configuration
          and compares cascaded vs parallel fusion end-to-end.
    """),
]

# ---------------------------------------------------------------------------
# Notebook 2 — Data model
# ---------------------------------------------------------------------------

NB_02_DATA_MODEL: list[tuple[str, str]] = [
    _md("""
        # 2. The data model — typed records that flow through a tracker

        unitrack 2.0 is built around a small set of **immutable, typed
        records** backed by `tensordict.tensorclass`. Every interesting
        data shape passing between stages of a tracker is one of these
        types. This makes the pipeline easy to reason about: a stage's
        signature tells you exactly what it consumes and produces.

        The five canonical records are:

        | Record | Shape | Role |
        |---|---|---|
        | `Tracklets` | (N,) batched | the tracker's snapshot of live identities |
        | `Detections` | (M,) batched | one frame's new observations |
        | `FrameContext` | scalar | timing + stream metadata |
        | `CostExpression` | (N, M) cost matrix + optional gates | cost producer's output |
        | `MatchOutcome` | matched pairs + residual indices | associator's output |

        Plus an algebraic variant type `Gate = PerPair | PerCs | PerDs | CostBias`.

        We'll walk through each in turn.
    """),
    _py("""
        import torch
        import matplotlib.pyplot as plt

        from unitrack.data import (
            CostExpression, Detections, FrameContext, Gate, MatchOutcome, Tracklets,
        )
        from unitrack.lifecycle import TrackletStatus

        torch.manual_seed(0)
    """),
    _md("""
        ## Tracklets — the snapshot

        A `Tracklets` record holds the tracker's view of every live
        identity. The reserved fields are common to every tracker:
        `id`, `status`, `hits`, `time_since_update`, `age`,
        `frame_started`, `frame_last_seen`. User fields (kernel, mask,
        bbox, …) are added at construction time and live alongside the
        reserved ones.

        Status is a `TrackletStatus` enum stored as `int8`:
    """),
    _py("""
        list(TrackletStatus)
    """),
    _py("""
        # Construct three tracklets — two Active, one Tentative.
        t = Tracklets(
            id=torch.tensor([10, 11, 12], dtype=torch.int64),
            status=torch.tensor(
                [TrackletStatus.Active, TrackletStatus.Active, TrackletStatus.Tentative],
                dtype=torch.int8,
            ),
            hits=torch.tensor([5, 5, 1], dtype=torch.int32),
            time_since_update=torch.zeros(3, dtype=torch.int32),
            age=torch.tensor([10, 8, 1], dtype=torch.int32),
            frame_started=torch.tensor([0, 2, 9], dtype=torch.int32),
            frame_last_seen=torch.tensor([10, 10, 10], dtype=torch.int32),
            # Two user fields:
            kernel=torch.randn(3, 8),
            position=torch.tensor([[100.0, 50.0], [200.0, 80.0], [50.0, 30.0]]),
            batch_size=[3],
        )
        print(f"batch_size: {t.batch_size}")
        print(f"ids:        {t.id.tolist()}")
        print(f"statuses:   {t.status.tolist()}  (0=Tentative, 1=Active)")
        print(f"kernel.shape: {t.kernel.shape}")
        print(f"position:    {t.position.tolist()}")
    """),
    _md("""
        Snapshots are batch-aware: indexing returns a Tracklets of the
        sub-batch, with all fields sliced consistently.
    """),
    _py("""
        # Subset to only Active tracklets:
        active_only = t[t.status == int(TrackletStatus.Active)]
        print(f"active tracklets:  {active_only.batch_size[0]}")
        print(f"ids:               {active_only.id.tolist()}")
        print(f"position.shape:    {active_only.position.shape}")
    """),
    _md("""
        ## Detections — one frame's observations

        `Detections` mirrors `Tracklets` but only carries one reserved
        field, `index` (the caller-supplied per-detection ordering).
        User fields match the Tracklets schema for the same Tracker.
    """),
    _py("""
        d = Detections(
            index=torch.arange(4, dtype=torch.int64),
            kernel=torch.randn(4, 8),
            position=torch.tensor([[110.0, 52.0],   # close to tracklet 0
                                    [201.0, 81.0],   # close to tracklet 1
                                    [300.0, 200.0],  # new
                                    [49.0, 31.0]]),  # close to tracklet 2
            batch_size=[4],
        )
        print(f"detections:  {d.batch_size[0]}")
        print(f"index:       {d.index.tolist()}")
        print(f"position:    {d.position.tolist()}")
    """),
    _md("""
        ## What's being matched?

        Before we compute any cost, picture the problem in 2D: the
        snapshot's three tracklets (squares) and this frame's four
        detections (circles). The associator's whole job is to decide
        which circle continues which square — and which circle is a
        brand-new object. The spatial layout below is what the cost
        matrix in the next section measures.
    """),
    _py("""
        fig, ax = plt.subplots(figsize=(6, 4))
        tp, dp = t.position, d.position
        ax.scatter(tp[:, 0], tp[:, 1], s=160, marker="s", color="tab:blue",
                   edgecolor="black", label="tracklets (snapshot)", zorder=3)
        for i in range(t.batch_size[0]):
            ax.annotate(f"id={int(t.id[i])}", (tp[i, 0], tp[i, 1]),
                        textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.scatter(dp[:, 0], dp[:, 1], s=90, marker="o", color="tab:orange",
                   edgecolor="black", label="detections (this frame)", zorder=3)
        for j in range(d.batch_size[0]):
            ax.annotate(f"det {j}", (dp[j, 0], dp[j, 1]),
                        textcoords="offset points", xytext=(8, -12), fontsize=9)
        ax.set_title("The matching problem in 2D — which detection continues which tracklet?")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.show()
    """),
    _md("""
        Detections 0, 1, 3 each sit beside a tracklet; detection 2 (far
        top-right) has no nearby tracklet — it should fall through as a
        new identity. Keep that picture in mind as we build the cost.

        ## CostExpression — cost matrix with un-applied gates

        After a `CostProducer` runs, it returns a `CostExpression`: an
        (N, M) cost matrix together with optional **un-applied** gates
        and bias. Carrying the gates separately lets a downstream
        node decide *when* to apply them — handy if you want to merge
        costs from multiple branches before applying gates.

        Below: a simple 3×4 cost matrix (Euclidean distance between
        the tracklets' and detections' positions), with no gates yet.
    """),
    _py("""
        dist = torch.cdist(t.position, d.position, p=2.0)
        cost = CostExpression.from_matrix(dist)
        print("matrix (lower = closer):")
        print(cost.matrix.round(decimals=2))
        print(f"\\ngates attached: pair={cost.gate_pair}, cs={cost.gate_cs}, ds={cost.gate_ds}")
    """),
    _md("""
        Visualizing the cost matrix as a heatmap makes the structure
        obvious: tracklet *i* should match detection *j* where the
        cell is dark.
    """),
    _py("""
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cost.matrix.numpy(), cmap="viridis")
        ax.set_xlabel("detection index")
        ax.set_ylabel("tracklet index")
        ax.set_title("CostExpression.matrix — Euclidean distance over positions")
        ax.set_xticks(range(d.batch_size[0]))
        ax.set_yticks(range(t.batch_size[0]))
        for i in range(t.batch_size[0]):
            for j in range(d.batch_size[0]):
                ax.text(j, i, f"{cost.matrix[i, j]:.0f}",
                        ha="center", va="center", color="white", fontsize=9)
        plt.colorbar(im, ax=ax, label="cost (lower = match)")
        plt.show()
    """),
    _md("""
        ## Gate — algebraic variant for filtering pairs

        A `Gate` is a small union type with four constructors:

        - `Gate.PerPair(mask: (N, M))` — pairwise boolean accept/reject.
        - `Gate.PerCs(mask: (N,))` — drop entire tracklet rows.
        - `Gate.PerDs(mask: (M,))` — drop entire detection columns.
        - `Gate.CostBias(matrix: (N, M))` — additive cost penalty.

        Gates compose under conjunction via `Gate.combine(a, b)` —
        cross-kind combinations are promoted to the smallest common
        representation (e.g., `PerCs ∧ PerDs → PerPair`).

        Below: build a per-pair gate that only allows tracklet *i* to
        match detection *j* if their classes agree, then visualize
        the gated cost.
    """),
    _py("""
        # Pretend the user adds a class field to both tracklets and detections.
        t_class = torch.tensor([0, 1, 0])      # tracklet classes
        d_class = torch.tensor([0, 1, 2, 0])   # detection classes
        same_class = (t_class[:, None] == d_class[None, :])  # (N, M) bool

        gate = Gate.PerPair(mask=same_class)
        cost_gated = gate.apply(cost)

        # ``materialize`` applies all attached gates to produce a final cost
        # matrix where blocked pairs become +inf.
        materialized = cost_gated.materialize()
        print("After ClassGate (only same-class pairs allowed):")
        print(materialized.round(decimals=1))
    """),
    _py("""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (m, title) in zip(axes, [(cost.matrix, "before gate"),
                                          (materialized, "after class gate")]):
            disp = m.clone()
            disp[torch.isinf(disp)] = m[~torch.isinf(disp)].max() * 2
            im = ax.imshow(disp.numpy(), cmap="viridis")
            ax.set_xlabel("detection index")
            ax.set_ylabel("tracklet index")
            ax.set_title(title)
            for i in range(t.batch_size[0]):
                for j in range(d.batch_size[0]):
                    label = "∞" if torch.isinf(m[i, j]) else f"{m[i, j]:.0f}"
                    ax.text(j, i, label, ha="center", va="center",
                            color="white", fontsize=9)
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        ## MatchOutcome — what the associator returns

        After running an `Associator` (Jonker, Hungarian, Greedy, …),
        you get a `MatchOutcome` that records:

        - `matched_pairs`: (K, 2) — `(tracklet_index, detection_index)` rows.
        - `tracklets_residual_index`: which tracklets did NOT match.
        - `detections_residual_index`: which detections did NOT match.
        - `per_match_cost`: per-pair assignment cost (telemetry/HPO).
    """),
    _py("""
        from unitrack.assignment import Associate, Jonker

        m = MatchOutcome.empty()
        print(f"empty MatchOutcome: matched_pairs={m.matched_pairs.shape}")

        # Build an Associator and run it on the gated cost.
        ctx = FrameContext.make(frame_idx=0, delta=0.0)
        outcome = Associate(Jonker(threshold=200.0))(t, d, ctx, cost_gated)
        print(f"matched_pairs: {outcome.matched_pairs.tolist()}")
        print(f"residual tracklets: {outcome.tracklets_residual_index.tolist()}")
        print(f"residual detections: {outcome.detections_residual_index.tolist()}")
    """),
    _md("""
        Reading the matched pairs:

        - tracklet 0 (class 0) ↔ detection 0 (class 0)
        - tracklet 1 (class 1) ↔ detection 1 (class 1)
        - tracklet 2 (class 0) ↔ detection 3 (class 0)

        Detection 2 (class 2) had no class-matching tracklet, so it lands
        in `detections_residual_index` — ready to be promoted to a brand-
        new tracklet by the rest of `Tracker.step`.

        ## What's next

        Notebook **3** visualizes the cost zoo (cosine, IoU, BiSoftmax,
        Mahalanobis, …) and the gate zoo (Class, Score, Spatial, Motion)
        side by side, showing how each shape fits the algebra above.
    """),
]

# ---------------------------------------------------------------------------
# Notebook 3 — Costs and gates
# ---------------------------------------------------------------------------

NB_03_COSTS_GATES: list[tuple[str, str]] = [
    _md("""
        # 3. The cost & gate zoos

        unitrack ships a catalogue of **cost producers** (functions
        from `(tracklets, detections)` to a `CostExpression`) and
        **gate producers** (functions to a `Gate` variant). Mixing
        and matching them is most of the design space the paper
        explores.

        This notebook visualizes each one on small toy inputs so
        you can see at a glance what shape each module produces.
    """),
    _py("""
        import torch
        import matplotlib.pyplot as plt
        import numpy as np

        from unitrack.costs import (
            BiSoftmax, BoxCIoU, BoxGIoU, BoxIoU, CDist, Chamfer,
            Cosine, Mahalanobis, MaskIoU, RBF, Reduce, Weighted,
        )
        from unitrack.data import Detections, FrameContext, Gate, Tracklets
        from unitrack.gates import (
            ClassGate, MotionGate, NoneGate, ScoreGate,
            SpatialGate2D, SpatialGate3D,
        )
        from unitrack.lifecycle import TrackletStatus

        torch.manual_seed(0)
        ctx = FrameContext.make(frame_idx=0, delta=0.0)
    """),
    _md("""
        ## A common toy fixture

        Throughout this notebook we'll use 3 tracklets vs 4 detections
        with synthetic kernel embeddings, masks, bboxes, and centroids.
    """),
    _py("""
        N, M, D = 3, 4, 8

        def make_tracklets(*, kernel=None, mask=None, bbox=None, centroid=None,
                           cov=None, klass=None):
            base = dict(
                id=torch.arange(N, dtype=torch.int64),
                status=torch.full((N,), int(TrackletStatus.Active), dtype=torch.int8),
                hits=torch.ones(N, dtype=torch.int32),
                time_since_update=torch.zeros(N, dtype=torch.int32),
                age=torch.ones(N, dtype=torch.int32),
                frame_started=torch.zeros(N, dtype=torch.int32),
                frame_last_seen=torch.zeros(N, dtype=torch.int32),
                batch_size=[N],
            )
            if kernel is not None:    base["kernel"] = kernel
            if mask is not None:       base["mask"] = mask
            if bbox is not None:       base["bbox"] = bbox
            if centroid is not None:   base["centroid"] = centroid
            if cov is not None:        base["centroid_cov"] = cov
            if klass is not None:      base["klass"] = klass
            return Tracklets(**base)

        def make_dets(*, kernel=None, mask=None, bbox=None, centroid=None,
                       klass=None, score=None):
            base = dict(
                index=torch.arange(M, dtype=torch.int64),
                batch_size=[M],
            )
            if kernel is not None:    base["kernel"] = kernel
            if mask is not None:       base["mask"] = mask
            if bbox is not None:       base["bbox"] = bbox
            if centroid is not None:   base["centroid"] = centroid
            if klass is not None:      base["klass"] = klass
            if score is not None:      base["score"] = score
            return Detections(**base)

        # Three tracklets with somewhat-distinct kernel embeddings:
        cs_kernel = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)
        # Four detections — first 3 close to corresponding tracklets, last is novel.
        ds_kernel = torch.tensor([
            [0.9, 0.1, 0, 0, 0, 0, 0, 0],   # close to cs[0]
            [0.1, 0.9, 0, 0, 0, 0, 0, 0],   # close to cs[1]
            [0, 0.1, 0.9, 0, 0, 0, 0, 0],   # close to cs[2]
            [0, 0, 0, 1, 0, 0, 0, 0],       # novel
        ], dtype=torch.float32)
    """),
    _md("""
        ## Heatmap helper

        We'll plot every cost matrix the same way: tracklets on the
        rows, detections on the columns, colorbar with low = better
        match.
    """),
    _py("""
        def plot_cost(matrix: torch.Tensor, title: str, ax=None, cmap="viridis"):
            if ax is None:
                fig, ax = plt.subplots(figsize=(4, 3))
            disp = matrix.detach().clone()
            if torch.isinf(disp).any():
                disp[torch.isinf(disp)] = disp[~torch.isinf(disp)].max() * 2
            im = ax.imshow(disp.numpy(), cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("ds")
            ax.set_ylabel("cs")
            ax.set_xticks(range(matrix.shape[1]))
            ax.set_yticks(range(matrix.shape[0]))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    v = matrix[i, j].item()
                    label = "∞" if not np.isfinite(v) else f"{v:.2f}"
                    ax.text(j, i, label, ha="center", va="center",
                            color="white", fontsize=8)
            return ax
    """),
    _md("""
        ## The appearance fixture, before any cost

        Every distance/similarity cost below reads the 8-dim `kernel`
        field. Here are the raw embeddings the costs see: three tracklet
        rows (`cs`) and four detection rows (`ds`). Rows 0–2 of `ds` are
        near-copies of the matching `cs` row; `ds` row 3 lights up a
        different dimension — a novel object with no tracklet to match.
    """),
    _py("""
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for ax, emb, title in [
            (axes[0], cs_kernel, "Tracklet kernels (cs)"),
            (axes[1], ds_kernel, "Detection kernels (ds)"),
        ]:
            im = ax.imshow(emb.numpy(), cmap="magma", vmin=0, vmax=1, aspect="auto")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("embedding dim"); ax.set_ylabel("row")
            ax.set_yticks(range(emb.shape[0]))
            ax.set_xticks(range(emb.shape[1]))
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, label="value")
        fig.suptitle("Appearance feature space — what the distance/similarity costs see")
        plt.show()
    """),
    _md("""
        ## Distance / similarity costs over kernel embeddings

        - **Cosine** — `1 − cos(a, b)`. Direction-only; magnitude irrelevant.
        - **CDist** — `‖a − b‖_p` Minkowski distance (default p=2).
        - **BiSoftmax** — bidirectional softmax similarity (paper-aligned).
        - **RBF** — `1 − exp(−γ ‖a − b‖²)`; kernel-style similarity.
    """),
    _py("""
        cs = make_tracklets(kernel=cs_kernel)
        ds = make_dets(kernel=ds_kernel)

        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
        plot_cost(Cosine("kernel")(cs, ds, ctx).matrix,    "Cosine",    ax=axes[0])
        plot_cost(CDist("kernel")(cs, ds, ctx).matrix,     "CDist (p=2)", ax=axes[1])
        plot_cost(BiSoftmax("kernel")(cs, ds, ctx).matrix, "BiSoftmax", ax=axes[2])
        plot_cost(RBF("kernel", gamma=1.0)(cs, ds, ctx).matrix, "RBF (γ=1)", ax=axes[3])
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        Notice the diagonals of the first three: each tracklet has its
        lowest cost at the matching detection (0, 1, 2) and a high
        cost at the novel detection (3). RBF compresses the contrast
        because exp(0) ≈ 1 for already-close pairs.
    """),
    _md("""
        ## Overlap costs over masks and bboxes

        - **MaskIoU** — `1 − IoU` over bitmasks.
        - **BoxIoU** — plain `1 − IoU`.
        - **BoxGIoU** — `1 − GIoU` (penalizes enclosing area).
        - **BoxCIoU** — `1 − CIoU` (penalizes centre offset + aspect ratio).

        Below we test on bounding boxes that are mostly aligned but
        have one outlier.
    """),
    _py("""
        # Three tracklets, four detections; identity bboxes near-overlap.
        cs_box = torch.tensor([
            [10, 10, 30, 30],
            [50, 10, 70, 30],
            [90, 10, 110, 30],
        ], dtype=torch.float32)
        ds_box = torch.tensor([
            [12, 11, 32, 31],     # close to cs[0]
            [49, 12, 70, 32],     # close to cs[1]
            [88, 11, 109, 30],    # close to cs[2]
            [200, 200, 220, 220], # far from all
        ], dtype=torch.float32)

        # Draw the boxes themselves before scoring them.
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (x0, y0, x1, y1) in enumerate(cs_box.tolist()):
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, edgecolor="tab:blue", linewidth=2))
            ax.text(x0, y0 - 2, f"cs{i}", color="tab:blue", fontsize=9)
        for j, (x0, y0, x1, y1) in enumerate(ds_box.tolist()):
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, edgecolor="tab:orange", linewidth=2,
                         linestyle="--"))
            ax.text(x1, y1 + 2, f"ds{j}", color="tab:orange", fontsize=9)
        ax.set_xlim(0, 240); ax.set_ylim(240, 0)  # image convention: y down
        ax.set_aspect("equal")
        ax.set_title("Box fixture — solid = tracklets (cs), dashed = detections (ds)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.show()
    """),
    _md("""
        Three detection boxes hug their matching tracklet; `ds3` sits far
        away with no overlap. The IoU-family costs below turn that picture
        into numbers.
    """),
    _py("""
        cs_b = make_tracklets(bbox=cs_box)
        ds_b = make_dets(bbox=ds_box)

        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        plot_cost(BoxIoU("bbox")(cs_b, ds_b, ctx).matrix,  "BoxIoU",  ax=axes[0])
        plot_cost(BoxGIoU("bbox")(cs_b, ds_b, ctx).matrix, "BoxGIoU", ax=axes[1])
        plot_cost(BoxCIoU("bbox")(cs_b, ds_b, ctx).matrix, "BoxCIoU", ax=axes[2])
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        ## Mahalanobis distance — Kalman-style gating

        `Mahalanobis(field, cov_field)` reads a per-tracklet covariance
        matrix from the snapshot and computes
        $(a-b)^T \\Sigma^{-1} (a-b)$ over each pair. With identity
        covariance this is squared L2; with anisotropic covariance,
        elongation along the dominant axis is "free."
    """),
    _py("""
        from matplotlib.patches import Ellipse

        cs_pos = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        ds_pos = torch.tensor([[0.5, 0.5], [9.0, 1.0], [0.0, 11.0], [50.0, 50.0]])

        eye_cov = torch.eye(2).expand(N, 2, 2).contiguous()
        aniso = torch.eye(2).clone()
        aniso[0, 0] = 100.0           # 100x the variance along x
        aniso_cov = aniso.expand(N, 2, 2).contiguous()

        # The geometry: centroids, plus each tracklet's 1-sigma covariance
        # ellipse. Mahalanobis distance measures how many sigmas a detection
        # sits from a tracklet — so points inside a wide ellipse are "cheap."
        def draw_centroids(ax, cov, title):
            ax.scatter(cs_pos[:, 0], cs_pos[:, 1], marker="s", s=120,
                       color="tab:blue", edgecolor="black", label="cs", zorder=3)
            ax.scatter(ds_pos[:, 0], ds_pos[:, 1], marker="o", s=70,
                       color="tab:orange", edgecolor="black", label="ds", zorder=3)
            for i in range(N):
                w = 2 * float(cov[i, 0, 0]) ** 0.5   # 1-sigma full width
                h = 2 * float(cov[i, 1, 1]) ** 0.5
                ax.add_patch(Ellipse(cs_pos[i].tolist(), w, h, angle=0,
                             fill=False, edgecolor="tab:blue", alpha=0.6))
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
            ax.set_xlim(-8, 22); ax.set_ylim(-8, 22)  # ds3 (50,50) is off-plot
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        draw_centroids(axes[0], eye_cov, "Σ = I  (isotropic 1σ)")
        draw_centroids(axes[1], aniso_cov, "Σ_x = 100·Σ_y  (stretched in x)")
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        Now the cost matrices for the same two covariance settings (the
        far-away `ds3` is the easy reject in both):
    """),
    _py("""
        ds = make_dets(centroid=ds_pos)

        cs = make_tracklets(centroid=cs_pos, cov=eye_cov)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        plot_cost(Mahalanobis("centroid", "centroid_cov")(cs, ds, ctx).matrix,
                   "Mahalanobis (Σ = I)", ax=axes[0])

        cs = make_tracklets(centroid=cs_pos, cov=aniso_cov)
        plot_cost(Mahalanobis("centroid", "centroid_cov")(cs, ds, ctx).matrix,
                   "Mahalanobis (Σ_x = 100·Σ_y)", ax=axes[1])
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        Notice how the high-variance-x covariance flattens the cost
        differences along the x axis — a tracklet that's been tracked
        with much position uncertainty in x cares less about a
        detection that's a few units off in x.
    """),
    _md("""
        ## Combinators — mixing costs

        - **Reduce** combines K cost matrices with a reduction
          (`sum`, `mean`, `min`, `max`, `product`).
        - **Weighted** scales an inner cost by a constant.
        - **Sinkhorn** (not visualized here) renormalises a cost via
          entropy-regularized OT — relevant for differentiable tracking.
    """),
    _py("""
        cs = make_tracklets(kernel=cs_kernel)
        ds = make_dets(kernel=ds_kernel)
        cosine = Cosine("kernel")
        cdist = CDist("kernel")

        sum_cost = Reduce([cosine, cdist], "sum")(cs, ds, ctx).matrix
        weighted = Reduce(
            [cosine, Weighted(cdist, weight=0.1)], "sum"
        )(cs, ds, ctx).matrix

        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
        plot_cost(cosine(cs, ds, ctx).matrix, "Cosine", ax=axes[0])
        plot_cost(cdist(cs, ds, ctx).matrix,  "CDist",  ax=axes[1])
        plot_cost(sum_cost,  "Reduce(sum)",      ax=axes[2])
        plot_cost(weighted, "Reduce(Cosine + 0.1·CDist)", ax=axes[3])
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        ## Gates — boolean (or biased) acceptance over pairs

        Gates filter or penalise (cs, ds) pairs *before* the
        associator runs. The unitrack catalogue:

        - **NoneGate** — identity. Every pair survives.
        - **ClassGate(field)** — pairs allowed iff their class fields agree.
        - **ScoreGate(field, threshold)** — drops detections whose score is below threshold (per-side).
        - **SpatialGate2D / 3D** — drops pairs whose Euclidean distance exceeds a threshold.
        - **MotionGate** — Mahalanobis χ² gate (Kalman-aware).
    """),
    _py("""
        # Build a richer fixture so each gate has something to act on.
        # Centroids carry a z-axis so the 3-D spatial gate has a third
        # component to act on; the 2-D gate ignores the trailing coordinate.
        cs_kernel_g = cs_kernel
        ds_kernel_g = ds_kernel
        cs_klass = torch.tensor([0, 1, 0])
        ds_klass = torch.tensor([0, 1, 0, 2])
        ds_score = torch.tensor([0.9, 0.6, 0.4, 0.95])
        cs_pos_g = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        ds_pos_g = torch.tensor(
            [[2.0, 0.0, 0.0], [49.0, 1.0, 0.0], [98.0, 0.0, 0.0], [200.0, 0.0, 0.0]]
        )

        cs = make_tracklets(kernel=cs_kernel_g, centroid=cs_pos_g, klass=cs_klass)
        ds = make_dets(kernel=ds_kernel_g, centroid=ds_pos_g,
                        klass=ds_klass, score=ds_score)
    """),
    _py("""
        def show_gate(g_call, title, ax):
            g = g_call(cs, ds, ctx)
            if g.kind == "per_pair":
                m = g.mask
            elif g.kind == "per_cs":
                m = g.mask[:, None].expand(-1, M)
            elif g.kind == "per_ds":
                m = g.mask[None, :].expand(N, -1)
            else:
                m = torch.ones((N, M), dtype=torch.bool)
            ax.imshow(m.numpy().astype(float), cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_title(f"{title}\\n[{g.kind}]", fontsize=9)
            ax.set_xlabel("ds")
            ax.set_ylabel("cs")
            ax.set_xticks(range(M))
            ax.set_yticks(range(N))

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        show_gate(NoneGate(),                            "NoneGate",          axes[0])
        show_gate(ClassGate("klass"),                    "ClassGate('klass')", axes[1])
        show_gate(ScoreGate("score", threshold=0.7),    "ScoreGate(>0.7)",   axes[2])
        show_gate(SpatialGate2D("centroid", max_dist=10.0),
                   "Spatial2D(<10px)", axes[3])
        show_gate(SpatialGate3D("centroid", max_dist=50.0),
                   "Spatial3D(<50px)", axes[4])
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        Reading the heatmaps: green = pair allowed; red = pair rejected.

        ## Gate composition

        Gates form a closed algebra under conjunction. Cross-kind
        pairs are promoted to the smallest variant that holds the
        result:

        | a            | b            | result |
        |--------------|--------------|--------|
        | PerCs        | PerCs        | PerCs |
        | PerDs        | PerDs        | PerDs |
        | PerCs        | PerDs        | PerPair (outer-AND) |
        | any          | PerPair      | PerPair |
        | CostBias     | CostBias     | CostBias (sum) |

        Combining `ClassGate` (per-pair) with `ScoreGate` (per-ds)
        promotes to per-pair:
    """),
    _py("""
        gA = ClassGate("klass")(cs, ds, ctx)             # per_pair
        gB = ScoreGate("score", threshold=0.7)(cs, ds, ctx)  # per_ds
        combined = Gate.combine(gA, gB)
        print(f"ClassGate kind:  {gA.kind}")
        print(f"ScoreGate kind:  {gB.kind}")
        print(f"Combined kind:   {combined.kind}")

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(combined.mask.numpy().astype(float), cmap="RdYlGn",
                   vmin=0, vmax=1)
        ax.set_title("ClassGate ∧ ScoreGate(>0.7)\\n(green = both allow)")
        ax.set_xlabel("ds")
        ax.set_ylabel("cs")
        plt.show()
    """),
    _md("""
        ## What's next

        Notebook **4** wires costs and gates into composable pipelines
        with `Pipe`, `Sequential`, `Parallel`, `Gated`, `Filter`, and
        `Iterate` — these are the combinators that build up actual
        trackers.
    """),
]

# ---------------------------------------------------------------------------
# Notebook 4 — Pipeline tree
# ---------------------------------------------------------------------------

NB_04_PIPELINE: list[tuple[str, str]] = [
    _md("""
        # 4. The composable pipeline tree

        unitrack 2.0 builds trackers from a **typed tree of stages**.
        There are three leaf signatures:

        - `GateProducer: (cs, ds, ctx) → Gate`
        - `CostProducer: (cs, ds, ctx) → CostExpression`
        - `Associator: (cs, ds, ctx, cost?) → MatchOutcome`

        and a small set of combinators that compose them:

        - `Pipe(cost, assoc)` — bridge: cost → match.
        - `Sequential[T]([s1, s2, …])` — chain of T-producers.
          For T=MatchOutcome: cascaded matching (residuals chain through).
          For T=Gate: gates fold under `Gate.combine`.
        - `Parallel(children, merge)` — cost-level merge of K branches.
        - `Gated(gate, then)` — apply a gate, then run the body.
        - `Filter(predicate, on, then)` — drop rows of cs and/or ds.
        - `Iterate(n, body)` — repeat a body n times.

        This notebook builds and visualizes each combinator.
    """),
    _py("""
        import torch
        import matplotlib.pyplot as plt

        from unitrack.assignment import Associate, Greedy, Jonker
        from unitrack.costs import CDist, Cosine, MaskIoU, Reduce
        from unitrack.data import Detections, FrameContext, Tracklets
        from unitrack.gates import ClassGate, ScoreGate, SpatialGate2D
        from unitrack.lifecycle import (
            MaxAgeFilter, StatusFilter, TrackletStatus,
        )
        from unitrack.pipeline import (
            Filter, Gated, Iterate, Parallel, Pipe, Sequential,
        )
        from unitrack.pipeline.merge import WeightedSum

        torch.manual_seed(0)
        ctx = FrameContext.make(0, delta=0.0)
    """),
    _md("""
        ## Common fixture

        Three tracklets, four detections; all share a kernel field and
        a class. We use the same data across each combinator so the
        differences in their behavior are visible.
    """),
    _py("""
        N, M = 3, 4
        cs = Tracklets(
            id=torch.arange(N, dtype=torch.int64),
            status=torch.full((N,), int(TrackletStatus.Active), dtype=torch.int8),
            hits=torch.ones(N, dtype=torch.int32),
            time_since_update=torch.zeros(N, dtype=torch.int32),
            age=torch.tensor([1, 1, 6], dtype=torch.int32),  # last one is older
            frame_started=torch.zeros(N, dtype=torch.int32),
            frame_last_seen=torch.zeros(N, dtype=torch.int32),
            kernel=torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ], dtype=torch.float32),
            klass=torch.tensor([0, 1, 0]),
            position=torch.tensor([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]),
            batch_size=[N],
        )
        ds = Detections(
            index=torch.arange(M, dtype=torch.int64),
            kernel=torch.tensor([
                [0.95, 0.05, 0, 0, 0, 0, 0, 0],
                [0.05, 0.95, 0, 0, 0, 0, 0, 0],
                [0, 0.05, 0.95, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ], dtype=torch.float32),
            klass=torch.tensor([0, 1, 0, 2]),
            score=torch.tensor([0.9, 0.6, 0.4, 0.95]),
            position=torch.tensor([
                [2.0, 0.0], [49.0, 1.0], [98.0, 0.0], [200.0, 0.0]
            ]),
            batch_size=[M],
        )
    """),
    _py("""
        def report(name: str, outcome):
            pairs = outcome.matched_pairs.tolist()
            res_cs = outcome.tracklets_residual_index.tolist()
            res_ds = outcome.detections_residual_index.tolist()
            print(f"{name:30s} matched={pairs}, residual cs={res_cs}, ds={res_ds}")
    """),
    _md("""
        ## The fixture, at a glance

        Every combinator below runs on this same data, so it's worth
        seeing it once. Squares are tracklets, circles are detections,
        colored by class. The right panel is the appearance similarity
        (cosine over `kernel`) between every tracklet and detection — the
        diagonal three are near-matches, and `ds3` (class 2, novel
        kernel) matches nothing. Watch how each combinator carves up
        exactly this picture.
    """),
    _py("""
        fig, (axp, axk) = plt.subplots(1, 2, figsize=(12, 4))
        cmap = plt.get_cmap("tab10")

        # (left) spatial layout, annotated with class / score / age.
        for i in range(N):
            p = cs.position[i]
            axp.scatter(p[0], p[1], marker="s", s=170, color=cmap(int(cs.klass[i])),
                        edgecolor="black", zorder=3)
            axp.annotate(f"cs{i} cls={int(cs.klass[i])} age={int(cs.age[i])}",
                         (p[0], p[1]), textcoords="offset points",
                         xytext=(8, 8), fontsize=8)
        for j in range(M):
            p = ds.position[j]
            axp.scatter(p[0], p[1], marker="o", s=110, color=cmap(int(ds.klass[j])),
                        edgecolor="black", zorder=3)
            axp.annotate(f"ds{j} cls={int(ds.klass[j])} s={ds.score[j]:.2f}",
                         (p[0], p[1]), textcoords="offset points",
                         xytext=(8, -16), fontsize=8)
        axp.set_title("Fixture in 2D — squares=tracklets, circles=detections (color=class)")
        axp.set_xlabel("x"); axp.set_ylabel("y"); axp.grid(alpha=0.3)

        # (right) appearance similarity, cs x ds.
        kn = torch.nn.functional.normalize
        sim = kn(cs.kernel, dim=-1) @ kn(ds.kernel, dim=-1).T
        im = axk.imshow(sim.numpy(), cmap="viridis", vmin=0, vmax=1)
        axk.set_title("Appearance similarity (cs × ds)")
        axk.set_xlabel("ds"); axk.set_ylabel("cs")
        axk.set_xticks(range(M)); axk.set_yticks(range(N))
        for i in range(N):
            for j in range(M):
                axk.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                         color="white", fontsize=8)
        fig.colorbar(im, ax=axk, fraction=0.046)
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        ## Pipe — the cost→match bridge

        The simplest end-to-end stage: compute a cost, run an
        assignment.
    """),
    _py("""
        pipe = Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5)))
        report("Pipe(Cosine + Jonker)", pipe(cs, ds, ctx))
    """),
    _md("""
        ## Gated — apply a gate before the body

        Gated is the wrap-this-with-a-gate combinator. With a per-pair
        ClassGate, only same-class pairs are allowed before matching.
    """),
    _py("""
        gated = Gated(gate=ClassGate("klass"), then=pipe)
        report("Gated(ClassGate, Pipe)", gated(cs, ds, ctx))

        # Same Pipe gated by ScoreGate (per-detection)
        gated_score = Gated(gate=ScoreGate("score", threshold=0.7), then=pipe)
        report("Gated(ScoreGate>0.7, Pipe)", gated_score(cs, ds, ctx))
    """),
    _md("""
        ## Sequential — cascaded matching

        `Sequential[MatchOutcome]([s1, s2, …])` chains stages: each
        child's residuals (unmatched cs, unmatched ds) feed the next.
        This is cascaded fusion.
    """),
    _py("""
        # Stage 1: strict cosine match (only near-perfect kernel similarity).
        # Stage 2: relaxed CDist on position for the leftovers.
        seq = Sequential([
            Pipe(cost=Cosine("kernel"),  assoc=Associate(Jonker(threshold=0.06))),
            Pipe(cost=CDist("position"), assoc=Associate(Jonker(threshold=10.0))),
        ])
        report("Sequential(strict, then relaxed)", seq(cs, ds, ctx))
    """),
    _md("""
        Reading the result: stage 1 found two strong cosine matches;
        stage 2 picked up the remaining tracklet via position
        proximity.

        ## Sequential[Gate] — folding gates

        Sequential of gate-producers folds children with `Gate.combine`.
        That gives you a single gate that's the conjunction of all the
        child gates.
    """),
    _py("""
        gate_chain = Sequential([
            ClassGate("klass"),
            ScoreGate("score", threshold=0.5),
        ])
        composed = gate_chain(cs, ds, ctx)
        print(f"composed gate kind: {composed.kind}")
        print(f"composed mask:")
        print(composed.mask.int())
    """),
    _md("""
        ## Parallel — cost-level merge

        `Parallel(children, merge)` runs each child *on the same input*
        and combines their CostExpressions via a `Merge` strategy.
        This is parallel fusion. Branches must be cost-
        producers (not associators).
    """),
    _py("""
        parallel = Parallel(
            children=[Cosine("kernel"), CDist("position")],
            merge=WeightedSum([1.0, 0.05]),
        )
        # Wrap in a Pipe so we can run it end-to-end.
        pipe_parallel = Pipe(cost=parallel, assoc=Associate(Jonker(threshold=0.6)))
        report("Parallel(Cosine, 0.05·CDist)", pipe_parallel(cs, ds, ctx))
    """),
    _md("""
        ## Filter — drop rows by predicate

        `Filter(predicate, on, then)` drops cs (or ds, or both) before
        the body runs. Common predicates live in
        `unitrack.lifecycle`:

        - `MaxAgeFilter(max_age=N)` — keep tracklets matched within
          the last N frames.
        - `StatusFilter(*statuses)` — keep tracklets in the listed
          status set.
    """),
    _py("""
        # Drop tracklets aged > 4 (only cs[2] in our fixture has age=6).
        filtered = Filter(MaxAgeFilter(max_age=4), on="cs", then=pipe)
        report("Filter(MaxAgeFilter(<=4), Pipe)", filtered(cs, ds, ctx))

        # Inspect what was filtered:
        print("\\nages:", cs.age.tolist())
        print("After Filter, residual cs includes the filtered-out tracklet.")
    """),
    _md("""
        Note that the filtered-out tracklet (index 2, age=6) appears
        in `tracklets_residual_index` — Filter remaps indices back to
        the unfiltered space and adds the dropped rows to the residual,
        so the parent stage doesn't see "missing" tracklets.

        ## Iterate — repeat a body

        `Iterate(n, body)` is `Sequential[MatchOutcome]([body] * n)` as
        a single configuration knob — handy for HPO over the number of
        cascaded stages.
    """),
    _py("""
        loop = Iterate(n=2, body=pipe)
        report("Iterate(n=2, Pipe(Cosine))", loop(cs, ds, ctx))
    """),
    _md("""
        ## Side-by-side: cascaded vs parallel fusion

        On the same data, cascaded and parallel often produce different
        match sets. K=2 cascaded tends to outperform parallel.
        Here we verify the two paths produce the right shapes.
    """),
    _py("""
        cascaded = Sequential([
            Pipe(cost=Cosine("kernel"),  assoc=Associate(Jonker(threshold=0.4))),
            Pipe(cost=CDist("position"), assoc=Associate(Jonker(threshold=10.0))),
        ])
        parallel = Pipe(
            cost=Parallel(
                children=[Cosine("kernel"), CDist("position")],
                merge=WeightedSum([1.0, 0.05]),
            ),
            assoc=Associate(Jonker(threshold=0.6)),
        )
        report("CASCADED  (Cosine → CDist)", cascaded(cs, ds, ctx))
        report("PARALLEL  (Cosine + 0.05·CDist)", parallel(cs, ds, ctx))
    """),
    _md("""
        ## What's next

        Notebook **5** introduces **state evolution** and the
        **lifecycle policy** — the parts of unitrack that keep
        tracklets alive across frames, even when they aren't matched
        on every frame.
    """),
]

# ---------------------------------------------------------------------------
# Notebook 5 — States and lifecycle
# ---------------------------------------------------------------------------

NB_05_STATES_LIFECYCLE: list[tuple[str, str]] = [
    _md("""
        # 5. State evolution and lifecycle

        Two pieces let unitrack maintain *coherent* identities across
        time, even when detections are noisy or missing for a few frames:

        1. **State** — a pair of pure functions `(Process, Observation)` plus an
           `Initializer`. Each tracklet field has its own state. The
           Process advances the field by `δt`; the Observation fuses
           a measurement when the tracklet matches a detection.
        2. **Lifecycle** — `Tentative → Active → Lost → Removed`
           transitions driven by `min_hits` and `max_age`.

        This notebook visualizes both on a synthetic clip with
        occlusion-style gaps.
    """),
    _py("""
        import torch
        import matplotlib.pyplot as plt

        import unitrack
        from unitrack.assignment import Associate, Jonker
        from unitrack.costs import Cosine
        from unitrack.data import Detections, FrameContext, TensorSpec, Tracklets
        from unitrack.lifecycle import (
            ConfirmedOnly, IncludeAll, IncludeTentative,
            NoLifecycle, StandardLifecycle, TrackletStatus,
        )
        from unitrack.pipeline import Pipe
        from unitrack.states import (
            EMADecay, EMAFuse, FromDetectionField, Identity, Replace, State,
        )
        from unitrack.states.kalman import KalmanCentroid2D

        torch.manual_seed(0)
    """),
    _md("""
        ## State catalogue

        The two halves of a State have different signatures:

        | | Signature | What it does |
        |---|---|---|
        | **Process** | `(cs, ctx) → cs'` | advance the field by `δt` |
        | **Observation** | `(cs, ds, match, ctx) → cs'` | fuse measurements for matched, apply miss-rule for unmatched |

        Built-in catalogue:

        - **Identity** (Process): no-op. Pair with `Replace` for fields like kernel embeddings or class labels.
        - **Replace** (Observation): matched tracklets adopt the new detection's value verbatim.
        - **EMADecay(field, half_life)** (Process): exponential decay toward zero.
        - **EMAFuse(field, rho)** (Observation): EMA blend of old/new values for matched.
        - **KalmanLinear / KalmanBBox / KalmanCentroid2D / KalmanCentroid3D** (Process): linear-Gaussian predict.
        - **KalmanUpdate** (Observation): Joseph-form update.
    """),
    _md("""
        ## Visualizing a Kalman state

        A 2D constant-velocity Kalman filter carries a 4D state
        `[x, y, vx, vy]` and a 4×4 covariance. The Process advances
        `x ← x + vx·δt` (likewise y) and grows the covariance by Q.
        The Observation fuses a 2D position measurement.

        We'll watch the predicted mean drift forward each frame, then
        snap toward the measurement when fused.
    """),
    _py("""
        proc = KalmanCentroid2D("centroid", q=0.1, r=0.5)

        # Construct a single-tracklet Tracklets snapshot manually.
        def _tracklets_one(mean, cov, *, age=1):
            return Tracklets(
                id=torch.tensor([1], dtype=torch.int64),
                status=torch.tensor([int(TrackletStatus.Active)], dtype=torch.int8),
                hits=torch.tensor([5], dtype=torch.int32),
                time_since_update=torch.zeros(1, dtype=torch.int32),
                age=torch.tensor([age], dtype=torch.int32),
                frame_started=torch.zeros(1, dtype=torch.int32),
                frame_last_seen=torch.zeros(1, dtype=torch.int32),
                centroid=mean.unsqueeze(0),
                centroid_cov=cov.unsqueeze(0),
                batch_size=[1],
            )

        # Initial state: at origin, moving (3, 1) px/frame, identity covariance.
        mean = torch.tensor([0.0, 0.0, 3.0, 1.0])
        cov = torch.eye(4) * 0.5
        snap = _tracklets_one(mean, cov)

        # Step the Process forward 5 frames; record the predicted positions.
        ctx = FrameContext.make(frame_idx=0, delta=1.0, fps=1.0)
        traj = [snap.centroid[0].clone()]
        for k in range(5):
            snap = proc(snap, ctx)
            traj.append(snap.centroid[0].clone())
        traj = torch.stack(traj)
        print("predicted (x, y, vx, vy) over time:")
        print(traj.round(decimals=2))
    """),
    _py("""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(traj[:, 0], traj[:, 1], "o-", label="predicted position")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("KalmanCentroid2D — pure prediction over 5 frames\\n(no measurement updates)")
        ax.grid(alpha=0.3)
        plt.show()
    """),
    _md("""
        Now let's add measurement updates. We'll match the tracklet
        against a detection that sits 0.5px off the prediction every
        frame. The Joseph-form update tightens the covariance and
        nudges the mean toward the measurement.
    """),
    _py("""
        update = proc.make_update()
        snap = _tracklets_one(mean.clone(), cov.clone())

        traj = [snap.centroid[0].clone()]
        cov_traces = [snap.centroid_cov[0].diagonal().clone()]
        meas = []
        for k in range(5):
            ctx = FrameContext.make(frame_idx=k+1, delta=1.0, fps=1.0)
            # 1. Predict
            snap = proc(snap, ctx)
            # 2. Build a fake measurement (offset by 0.5px from prediction).
            true_pos = snap.centroid[0, :2] + torch.tensor([0.5, 0.5])
            meas.append(true_pos.clone())
            ds = Detections(
                index=torch.tensor([0], dtype=torch.int64),
                centroid=true_pos.unsqueeze(0),
                batch_size=[1],
            )
            from unitrack.data import MatchOutcome
            match = MatchOutcome(
                matched_pairs=torch.tensor([[0, 0]], dtype=torch.int64),
                tracklets_residual_index=torch.zeros(0, dtype=torch.int64),
                detections_residual_index=torch.zeros(0, dtype=torch.int64),
                per_match_cost=torch.zeros(1),
                batch_size=[],
            )
            # 3. Fuse measurement
            snap = update(snap, ds, match, ctx)
            traj.append(snap.centroid[0].clone())
            cov_traces.append(snap.centroid_cov[0].diagonal().clone())
        traj = torch.stack(traj)
        cov_traces = torch.stack(cov_traces)
        meas = torch.stack(meas)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(traj[:, 0], traj[:, 1], "o-", label="fused mean")
        axes[0].scatter(meas[:, 0], meas[:, 1], marker="x", color="tab:red",
                        s=60, label="measurement", zorder=3)
        axes[0].legend(fontsize=8)
        axes[0].set_title("Kalman with measurement fusion")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].grid(alpha=0.3)
        for i, lbl in enumerate(["Var(x)", "Var(y)", "Var(vx)", "Var(vy)"]):
            axes[1].plot(cov_traces[:, i].numpy(), label=lbl, marker="o")
        axes[1].set_xlabel("frame")
        axes[1].set_ylabel("variance (diagonal)")
        axes[1].set_title("Covariance shrinks under measurement fusion")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        ## Lifecycle — Tentative → Active → Lost → Removed

        `StandardLifecycle(min_hits, max_age, allow_reid)` is unitrack's
        default state machine:

        - New tracklet enters as **Tentative**.
        - Tentative + matched on consecutive frames → **Active** when
          `hits >= min_hits`.
        - Tentative + missed → **Removed** (if past grace period).
        - Active + missed for `> max_age` frames → **Lost**.
        - Lost + missed for another `allow_reid` frames → **Removed**.

        We'll trace the lifecycle on a 12-frame clip with one tracklet
        that gets occluded for 4 frames (no detections).
    """),
    _py("""
        # Build a tracker with a 2-stage min_hits + 3-frame max_age policy.
        tracker = unitrack.Tracker(
            root=Pipe(cost=Cosine("kernel"), assoc=Associate(Jonker(threshold=0.5))),
            states={
                "kernel": State(
                    schema=TensorSpec(shape=(4,), dtype=torch.float32),
                    process=Identity("kernel"),
                    observation=Replace("kernel"),
                    init=FromDetectionField("kernel"),
                ),
            },
            lifecycle=StandardLifecycle(min_hits=2, max_age=3, allow_reid=2),
            visibility=IncludeAll(),
        )
        ms = unitrack.MultiStream(tracker)

        # 12 frames: present, present, present, present (occluded x4), present, present, present, present.
        appearance = torch.tensor([1.0, 0, 0, 0])
        present = [True, True, True, True, False, False, False, False, True, True, True, True]

        history = []  # list of (status, hits, tsu, age) per frame for tracklet 1
        for k, p in enumerate(present):
            if p:
                ds = Detections(
                    index=torch.tensor([0], dtype=torch.int64),
                    kernel=appearance.unsqueeze(0),
                    batch_size=[1],
                )
            else:
                ds = Detections(
                    index=torch.zeros(0, dtype=torch.int64),
                    kernel=torch.zeros((0, 4), dtype=torch.float32),
                    batch_size=[0],
                )
            ctx = FrameContext.make(frame_idx=k, delta=1.0, fps=1.0, stream_key=0)
            res = ms.step(stream_key=0, detections=ds, ctx=ctx)
            # Find tracklet 1 in the snapshot (might be filtered out by Removed).
            snap = res.snapshot
            mask = snap.id == 1
            if mask.any():
                history.append((
                    int(snap.status[mask].item()),
                    int(snap.hits[mask].item()),
                    int(snap.time_since_update[mask].item()),
                    int(snap.age[mask].item()),
                ))
            else:
                history.append(None)  # was removed
        for k, h in enumerate(history):
            print(f"frame {k:2d}  present={int(present[k])}  →  {h}")
    """),
    _py("""
        status_names = ["Tentative", "Active", "Lost", "Removed"]
        statuses = [(h[0] if h is not None else 3) for h in history]

        fig, ax = plt.subplots(figsize=(8, 3))
        for k, (s, p) in enumerate(zip(statuses, present)):
            color = ["#ffcc66", "#66cc66", "#aabbff", "#cccccc"][s]
            ax.barh(0, 1, left=k, color=color, edgecolor="white")
            ax.text(k + 0.5, 0, status_names[s][0], ha="center", va="center",
                    fontsize=8, color="black")
            if not p:
                ax.text(k + 0.5, -0.6, "miss", ha="center", va="center",
                        fontsize=7, color="#aa0000")
        ax.set_yticks([])
        ax.set_xticks(range(len(history) + 1))
        ax.set_xlim(0, len(history))
        ax.set_ylim(-1.0, 1.5)
        ax.set_xlabel("frame")
        ax.set_title("Lifecycle: T=Tentative, A=Active, L=Lost, R=Removed\\n(min_hits=2, max_age=3, allow_reid=2)")

        from matplotlib.patches import Patch
        legend = [
            Patch(color="#ffcc66", label="Tentative"),
            Patch(color="#66cc66", label="Active"),
            Patch(color="#aabbff", label="Lost"),
            Patch(color="#cccccc", label="Removed"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        Reading the timeline:

        - Frames 0–1: Tentative. The tracklet is being established.
        - Frame 2 onwards: Active (`hits >= 2`).
        - Frames 4–7: detection misses. `time_since_update` grows.
          When it exceeds `max_age=3` the tracklet transitions to Lost.
        - Frames 8 onwards: detection re-appears, the Lost tracklet
          re-acquires the same identity and goes back to Active.

        ## Visibility — what does the caller see?

        Three policies decide which IDs are visible to the caller of
        `Tracker.step`:

        - `ConfirmedOnly`: only Active tracklets that matched this frame.
        - `IncludeTentative`: also expose Tentative IDs.
        - `IncludeAll`: every live tracklet.

        Most production trackers use `ConfirmedOnly` to suppress
        flicker from unconfirmed detections.
    """),
    _md("""
        ## What's next

        Notebook **6** assembles everything you've seen — costs,
        gates, combinators, states, lifecycle — into a K=2 cascaded
        canonical configuration, runs it on a synthetic
        clip with known ground truth, and visualizes the resulting
        ID assignments.
    """),
]

# ---------------------------------------------------------------------------
# Notebook 6 — Cascaded and parallel fusion
# ---------------------------------------------------------------------------

NB_06_CASCADED: list[tuple[str, str]] = [
    _md("""
        # 6. End-to-end: K=2 cascaded and parallel fusion

        Time to assemble everything. We'll build a canonical
        two-stage cascaded tracker — strict appearance gating in
        stage 1 and relaxed motion-aware gating in stage 2 — run it
        on a synthetic clip with known ground truth, and contrast it
        against a parallel-fusion variant.

        > For an Optuna sweep with Mask2Former-Cityscapes detections,
        > see `examples/hpo_sweep/`.
        > This notebook is the self-contained educational version,
        > with everything happening in-process on deterministic
        > synthetic data.
    """),
    _py("""
        import torch
        import matplotlib.pyplot as plt
        import numpy as np

        import unitrack
        from unitrack.assignment import Associate, Jonker
        from unitrack.costs import CDist, Cosine, Reduce
        from unitrack.data import Detections, FrameContext, TensorSpec
        from unitrack.gates import ClassGate, MotionGate, ScoreGate
        from unitrack.lifecycle import (
            ConfirmedOnly, StandardLifecycle, StatusFilter, TrackletStatus,
        )
        from unitrack.pipeline import Filter, Gated, Parallel, Pipe, Sequential
        from unitrack.pipeline.merge import WeightedSum
        from unitrack.states import (
            FromDetectionField, Identity, Replace, State,
        )
        from unitrack.states.kalman import KalmanCentroid2D

        torch.manual_seed(42)
    """),
    _md("""
        ## Synthetic clip with known ground truth

        Three identities, eight frames. Each identity has a constant
        kernel embedding plus a 2D position drifting at constant
        velocity. Per-frame detection order is shuffled — the tracker
        has to use the kernel embedding (and motion) to recover
        identities.
    """),
    _py("""
        N_FRAMES, N_OBJS, K_DIM = 16, 3, 8

        def make_clip(seed: int = 0):
            g = torch.Generator().manual_seed(seed)
            # Orthogonal appearance embeddings so cosine cleanly separates ids.
            raw = torch.randn(K_DIM, K_DIM, generator=g)
            q, _ = torch.linalg.qr(raw)
            kernels = q[:N_OBJS]  # (N_OBJS, K_DIM), rows orthonormal.

            classes = torch.tensor([0, 0, 1])
            positions = torch.tensor([[20., 50.], [100., 30.], [180., 80.]])
            # Slow, well-separated velocities — Kalman's CV model tracks these
            # cleanly across the 16-frame window.
            velocities = torch.tensor([[0.6, 0.2], [-0.4, 0.3], [-0.3, -0.2]])

            clip, gt = [], []
            for k in range(N_FRAMES):
                order = torch.randperm(N_OBJS, generator=g)
                gt.append(order)
                kernel_obs = kernels[order] + 0.01 * torch.randn(
                    N_OBJS, K_DIM, generator=g
                )
                kernel_obs = kernel_obs / kernel_obs.norm(dim=-1, keepdim=True)
                pos_obs = positions[order] + k * velocities[order]
                clip.append(Detections(
                    index=torch.arange(N_OBJS, dtype=torch.int64),
                    kernel=kernel_obs.float(),
                    klass=classes[order],
                    score=torch.full((N_OBJS,), 0.9),
                    centroid=pos_obs.float(),
                    batch_size=[N_OBJS],
                ))
            return clip, torch.stack(gt)

        clip, gt = make_clip()
        print(f"Clip: {len(clip)} frames × {N_OBJS} detections.")
        print(f"GT identity per frame:\\n{gt.numpy()}")
    """),
    _md("""
        ## What's in the clip?

        The trackers below are graded against the ground truth, so let's
        look at it first. Un-shuffling each frame by its `gt` labels
        recovers three coherent trajectories (left). The appearance
        embeddings are orthonormal by construction, so the cosine
        similarity between identities is ≈0 off the diagonal (right) —
        appearance alone separates them, and motion is the tie-breaker
        when two of them (classes 0 and 0) get close.
    """),
    _py("""
        fig, (axp, axk) = plt.subplots(1, 2, figsize=(12, 4))
        cmap = plt.get_cmap("tab10")

        # (left) reconstruct ground-truth tracks from the shuffled clip.
        tracks = {o: [] for o in range(N_OBJS)}
        for k, dets in enumerate(clip):
            for r in range(N_OBJS):
                tracks[int(gt[k][r])].append(dets.centroid[r])
        for o in range(N_OBJS):
            pts = torch.stack(tracks[o])
            axp.plot(pts[:, 0], pts[:, 1], "-", color=cmap(o), alpha=0.4)
            axp.scatter(pts[:, 0], pts[:, 1], color=cmap(o), s=20,
                        edgecolor="black", linewidth=0.3, label=f"identity {o}")
            axp.scatter(pts[0, 0], pts[0, 1], color=cmap(o), marker="*", s=170,
                        edgecolor="black", zorder=3)
        axp.set_title("Ground-truth clip — 3 identities in 2D (★ = frame 0)")
        axp.set_xlabel("x"); axp.set_ylabel("y")
        axp.legend(fontsize=8); axp.grid(alpha=0.3)

        # (right) appearance separability at frame 0.
        kern0 = {int(gt[0][r]): clip[0].kernel[r] for r in range(N_OBJS)}
        K = torch.stack([kern0[o] for o in range(N_OBJS)])
        kn = torch.nn.functional.normalize
        sim = kn(K, dim=-1) @ kn(K, dim=-1).T
        im = axk.imshow(sim.numpy(), cmap="viridis", vmin=-1, vmax=1)
        axk.set_title("Appearance: orthonormal kernels → off-diagonal ≈ 0")
        axk.set_xlabel("identity"); axk.set_ylabel("identity")
        axk.set_xticks(range(N_OBJS)); axk.set_yticks(range(N_OBJS))
        for i in range(N_OBJS):
            for j in range(N_OBJS):
                axk.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                         color="white", fontsize=9)
        fig.colorbar(im, ax=axk, fraction=0.046)
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        ## State schema

        Five fields per tracklet — kernel (appearance), centroid
        (motion), klass, score, and a couple book-keeping fields.
    """),
    _py("""
        kalman = KalmanCentroid2D("centroid", q=0.5, r=0.5)

        STATES = {
            "kernel": State(
                schema=TensorSpec(shape=(K_DIM,), dtype=torch.float32),
                process=Identity("kernel"),
                observation=Replace("kernel"),
                init=FromDetectionField("kernel"),
            ),
            "klass": State(
                schema=TensorSpec(shape=(), dtype=torch.int64),
                process=Identity("klass"),
                observation=Replace("klass"),
                init=FromDetectionField("klass"),
            ),
            "score": State(
                schema=TensorSpec(shape=(), dtype=torch.float32),
                process=Identity("score"),
                observation=Replace("score"),
                init=FromDetectionField("score"),
            ),
            **kalman.state_entries(meas_field="centroid", init_cov_scale=10.0),
        }
    """),
    _md("""
        ## The K=2 cascaded canonical configuration

        ## The K=2 cascaded canonical configuration

        Two stages, strict then relaxed:

        - **Stage 1** — strict: `ClassGate` ∧ `ScoreGate(>0.6)` plus
          a tight cosine threshold. Only "obviously the same" pairs
          match here.
        - **Stage 2** — relaxed: `ClassGate` ∧ `MotionGate`
          (Mahalanobis χ² gate over the Kalman state) over a sum of
          cosine-on-kernel and Mahalanobis-on-centroid (here we use
          `CDist` for plot legibility on the 2-D centroid; the gate
          itself does the Kalman-aware projection).
    """),
    _py("""
        cascaded_root = Filter(
            predicate=StatusFilter(
                TrackletStatus.Tentative, TrackletStatus.Active, TrackletStatus.Lost,
            ),
            on="cs",
            then=Sequential([
                Pipe(
                    cost=Gated(
                        gate=Sequential([
                            ClassGate("klass"),
                            ScoreGate("score", threshold=0.6),
                        ]),
                        then=Cosine("kernel"),
                    ),
                    assoc=Associate(Jonker(threshold=0.3)),
                ),
                Pipe(
                    cost=Gated(
                        gate=Sequential([
                            ClassGate("klass"),
                            MotionGate("centroid", "centroid_cov", max_chi2=25.0),
                        ]),
                        then=Cosine("kernel"),
                    ),
                    assoc=Associate(Jonker(threshold=0.5)),
                ),
            ]),
        )

        cascaded_tracker = unitrack.Tracker(
            root=cascaded_root,
            states=STATES,
            lifecycle=StandardLifecycle(min_hits=1, max_age=3, allow_reid=2),
            visibility=ConfirmedOnly(),
        )
    """),
    _md("""
        ## The parallel-fusion variant

        One stage; the cost is a weighted sum of cosine-on-kernel and
        Mahalanobis distance over the Kalman state. `WeightedSum`
        merges both into parallel fusion mode.
    """),
    _py("""
        from unitrack.costs import Mahalanobis

        parallel_root = Filter(
            predicate=StatusFilter(
                TrackletStatus.Tentative, TrackletStatus.Active, TrackletStatus.Lost,
            ),
            on="cs",
            then=Pipe(
                cost=Parallel(
                    children=[
                        Cosine("kernel"),
                        Mahalanobis("centroid", "centroid_cov"),
                    ],
                    merge=WeightedSum([1.0, 0.05]),
                ),
                assoc=Associate(Jonker(threshold=2.0)),
            ),
        )

        parallel_tracker = unitrack.Tracker(
            root=parallel_root,
            states=STATES,
            lifecycle=StandardLifecycle(min_hits=1, max_age=3, allow_reid=2),
            visibility=ConfirmedOnly(),
        )
    """),
    _md("""
        ## Run both trackers on the same clip
    """),
    _py("""
        def run_clip(tracker, clip):
            ms = unitrack.MultiStream(tracker)
            results = []
            for k, dets in enumerate(clip):
                ctx = FrameContext.make(frame_idx=k, delta=1/15.0, fps=15.0, stream_key=0)
                res = ms.step(stream_key=0, detections=dets, ctx=ctx)
                results.append(res)
            return results

        cas_results = run_clip(cascaded_tracker, clip)
        par_results = run_clip(parallel_tracker, clip)

        for name, results in [("cascaded", cas_results), ("parallel", par_results)]:
            print(f"\\n{name.upper()}:")
            for k, res in enumerate(results):
                print(f"  frame {k}: snapshot={res.snapshot.batch_size[0]} live, "
                      f"confirmed ids={res.ids.tolist()}")
    """),
    _md("""
        Both configurations should produce stable identities — three
        tracklets persist across all frames, and from frame 1 onwards
        their IDs match the ground-truth identity assignment.

        ## Visualize the trajectories

        Each color = a tracker-assigned ID. We plot every detection at
        its centroid, colored by which tracklet the tracker thinks it
        belongs to. A clean visualization is one trajectory per color.
    """),
    _py("""
        def plot_trajectories(results, ax, title):
            cmap = plt.get_cmap("tab10")
            for k, res in enumerate(results):
                snap = res.snapshot
                # KalmanCentroid2D state layout: [x, y, vx, vy] — take the
                # first two dims for the plotted position.
                pos = snap.centroid[..., :2]
                ids = snap.id
                for n in range(snap.batch_size[0]):
                    ax.scatter(pos[n, 0], pos[n, 1],
                               color=cmap(int(ids[n]) % 10), s=40,
                               edgecolor="black", linewidth=0.5,
                               alpha=0.5 + 0.5 * (k / max(1, N_FRAMES - 1)))
            ax.set_title(title)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.grid(alpha=0.3)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_trajectories(cas_results, axes[0], "K=2 cascaded")
        plot_trajectories(par_results, axes[1], "Parallel fusion (Cosine + 0.05·CDist)")
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        Each trajectory should appear as a coherent fade-in line in
        one color. The marker alpha grows with frame index so you
        can tell which point came earliest.

        ## ID stability score

        A simple HOTA-flavored proxy: for each ground-truth identity,
        is its tracker-assigned ID stable across frames? We measure
        the fraction of frames where the GT identity gets its
        most-frequent tracker ID.
    """),
    _py("""
        def id_stability(results, gt) -> float:
            # For each GT identity, collect the tracker IDs it was actually
            # matched to (via res.match.matched_pairs) on each frame; the
            # fraction matching the per-GT-id mode is the stability score.
            tracker_ids_per_gt: dict[int, list[int]] = {g: [] for g in range(N_OBJS)}
            for k, res in enumerate(results):
                pairs = res.match.matched_pairs  # (P, 2) — (cs_idx, ds_idx) into res.snapshot / detection rows.
                snap_ids = res.snapshot.id
                for p in range(pairs.shape[0]):
                    cs_idx = int(pairs[p, 0].item())
                    ds_idx = int(pairs[p, 1].item())
                    tracker_id = int(snap_ids[cs_idx].item())
                    gt_id = int(gt[k][ds_idx].item())
                    tracker_ids_per_gt[gt_id].append(tracker_id)

            n_correct = n_total = 0
            for ids in tracker_ids_per_gt.values():
                if not ids:
                    continue
                most = max(set(ids), key=ids.count)
                n_correct += sum(1 for x in ids if x == most)
                n_total += len(ids)
            return n_correct / max(n_total, 1)

        print(f"Cascaded ID stability:  {id_stability(cas_results, gt):.3f}")
        print(f"Parallel ID stability:  {id_stability(par_results, gt):.3f}")
    """),
    _md("""
        On this small synthetic clip both shapes get most of the way.
        At scale, K=2 cascaded consistently outperforms parallel —
        the gap widens with more stages because parallel-merging
        conflicting matches injects errors.

        ## Where to go from here

        - For an Optuna sweep over the tracker design space (with
          Mask2Former on Cityscapes frames or a synthetic stand-in),
          look at `examples/hpo_sweep/`.
        - For multi-stream batched inference, see
          `unitrack.tracker.BatchTracker` (uses `torch.vmap`).
        - For clip-aware tracking (MinVIS, DVIS++ patterns), see
          `unitrack.tracker.ClipTracker`.
        - For end-to-end learnable tracking (gradients flowing through
          the matcher), construct your `Tracker(..., differentiable=True)`
          to swap in soft companions automatically.

        Happy tracking!
    """),
]

# ---------------------------------------------------------------------------
# Notebook 7 — Migrating from 1.x, with a real pretrained detector
# ---------------------------------------------------------------------------

NB_07_MIGRATION: list[tuple[str, str]] = [
    _md("""
        # 7. Migration & new possibilities — driven by a real detector

        This notebook does two things at once:

        1. **Migrates a real 1.x tracker to 2.0.** We take the
           appearance-embedding tracker from the
           [migration guide](../../docs/migration.md) and rebuild it in
           the 2.0 syntax, then run it on detections from an actual
           pretrained model.
        2. **Showcases what 2.0 makes possible** that 1.x could not
           express: parallel cost fusion, cascaded matching, lifecycle
           policies, and end-to-end **differentiable** matching.

        Everything runs locally on CPU. The detections come from a tiny
        Hugging Face detector (`hustvl/yolos-tiny`, ~6M params) plus a
        torchvision ResNet-18 appearance encoder — both download their
        weights on first run. If the model or network is unavailable, the
        notebook falls back to a deterministic synthetic seed so it still
        runs end to end.
    """),
    _md("""
        ## Setup

        Install the extra dependencies once (the core library needs only
        `unitrack`; this notebook adds a detector and plotting):
    """),
    _py("""
        # One-time install. Comment out after the first run.
        %pip install -q "transformers>=4.40" torchvision matplotlib pillow
    """),
    _py("""
        import urllib.request

        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import torch

        import unitrack as ut
        from unitrack.assignment import Associate, Jonker, SoftAssignment
        from unitrack.costs import BoxIoU, Cosine
        from unitrack.data import (
            Detections, FrameContext, TensorSpec, Tracklets,
        )
        from unitrack.gates import ClassGate, ScoreGate
        from unitrack.lifecycle import (
            ConfirmedOnly, IncludeAll, NoLifecycle, StandardLifecycle,
            TrackletStatus,
        )
        from unitrack.pipeline import Gated, Parallel, Pipe, Sequential
        from unitrack.pipeline.merge import WeightedSum
        from unitrack.states import (
            FromDetectionField, Identity, Replace, State,
        )

        torch.manual_seed(0)
        plt.rcParams["figure.figsize"] = (7, 4)
    """),
    _md("""
        ## A real, lightweight detector + appearance encoder

        unitrack tracks; it does not detect. In a real pipeline the
        per-frame detections come from an object detector, and the
        appearance embeddings from a ReID encoder. We use:

        - **Detector** — `hustvl/yolos-tiny` via 🤗 Transformers. Returns
          boxes (xyxy, pixels), class labels, and scores.
        - **Appearance** — torchvision `ResNet-18` with its classification
          head removed, giving a 512-d feature per cropped box. We
          L2-normalise it so a cosine cost is well-behaved.

        These are exactly the four fields the migrated tracker consumes:
        `bbox`, `category`, `score`, and `reid`.
    """),
    _py("""
        def load_detector_and_encoder():
            \"\"\"Load a tiny HF detector + a torchvision ReID encoder.\"\"\"
            import torchvision
            from transformers import (
                AutoImageProcessor, AutoModelForObjectDetection,
            )

            processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
            detector = AutoModelForObjectDetection.from_pretrained(
                "hustvl/yolos-tiny"
            ).eval()

            weights = torchvision.models.ResNet18_Weights.DEFAULT
            encoder = torchvision.models.resnet18(weights=weights)
            encoder.fc = torch.nn.Identity()  # expose the 512-d features
            encoder.eval()
            return processor, detector, encoder, weights.transforms()


        @torch.no_grad()
        def detect_and_embed(image, bundle, *, score_thr=0.7):
            \"\"\"Detect objects and attach a 512-d appearance embedding each.\"\"\"
            processor, detector, encoder, crop_tf = bundle
            inputs = processor(images=image, return_tensors="pt")
            outputs = detector(**inputs)
            target_sizes = torch.tensor([[image.height, image.width]])
            out = processor.post_process_object_detection(
                outputs, threshold=score_thr, target_sizes=target_sizes
            )[0]
            boxes, labels, scores = out["boxes"], out["labels"], out["scores"]

            crops = []
            for x0, y0, x1, y1 in boxes.tolist():
                x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)  # guard degenerate boxes
                crops.append(crop_tf(image.crop((x0, y0, x1, y1))))
            reid = (
                torch.nn.functional.normalize(encoder(torch.stack(crops)), dim=-1)
                if crops else torch.zeros(0, 512)
            )
            names = [detector.config.id2label[int(c)] for c in labels]
            return {
                "boxes": boxes.float(), "category": labels.long(),
                "score": scores.float(), "reid": reid.float(), "names": names,
            }
    """),
    _md("""
        Run the detector on one image. We use a stock multi-object photo;
        point `IMAGE_URL` (or a local `IMAGE_PATH`) at your own to try it
        on something else. If anything fails — no network, deps missing —
        we drop to a deterministic synthetic seed and keep going.
    """),
    _py("""
        IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


        def synthetic_seed(reid_dim=512):
            \"\"\"Offline stand-in: three objects, two sharing a class.\"\"\"
            g = torch.Generator().manual_seed(0)
            q, _ = torch.linalg.qr(torch.randn(reid_dim, reid_dim, generator=g))
            return {
                "boxes": torch.tensor([[10., 10, 40, 60],
                                       [120, 20, 150, 70],
                                       [220, 15, 250, 65]]),
                "category": torch.tensor([1, 1, 2]),
                "score": torch.tensor([0.95, 0.9, 0.8]),
                "reid": q[:3],
                "names": ["cat", "cat", "remote"],
            }


        image = None
        try:
            from PIL import Image
            bundle = load_detector_and_encoder()
            image = Image.open(urllib.request.urlopen(IMAGE_URL)).convert("RGB")
            seed = detect_and_embed(image, bundle)
            if seed["boxes"].shape[0] == 0:
                raise RuntimeError("detector found no objects above threshold")
            SOURCE = f"hugging face yolos-tiny ({len(seed['names'])} detections)"
        except Exception as exc:  # noqa: BLE001 — any failure -> deterministic seed
            print(f"[fallback] real model unavailable "
                  f"({type(exc).__name__}: {exc}); using a synthetic seed.")
            seed = synthetic_seed()
            SOURCE = "synthetic fallback"

        REID_DIM = seed["reid"].shape[1]
        print(f"source:  {SOURCE}")
        print(f"objects: {seed['names']}")
        print(f"scores:  {seed['score'].round(decimals=2).tolist()}")
    """),
    _py("""
        # Show the detections on the image (or list them in the fallback case).
        if image is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.imshow(image)
            for (x0, y0, x1, y1), name, sc in zip(
                seed["boxes"].tolist(), seed["names"], seed["score"].tolist()
            ):
                ax.add_patch(mpatches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    fill=False, edgecolor="lime", linewidth=2))
                ax.text(x0, y0 - 4, f"{name} {sc:.2f}", color="lime",
                        fontsize=9,
                        bbox=dict(facecolor="black", alpha=0.5, pad=1))
            ax.set_axis_off()
            ax.set_title(f"Detections from {SOURCE}")
            plt.show()
        else:
            for name, box in zip(seed["names"], seed["boxes"].tolist()):
                print(f"  {name:8s} bbox={[round(v, 1) for v in box]}")
    """),
    _md("""
        ## From one image to a clip

        Tracking needs a *sequence*. We synthesize one from the single set
        of real detections: each frame pans the camera by a fixed vector,
        shuffles the detection order (so the tracker can't cheat on row
        alignment), and jitters the appearance embeddings slightly. The
        motion is deterministic, so every tracker below sees the identical
        clip — differences in their output come from the tracker, not the
        data.
    """),
    _py("""
        def make_clip(seed, *, n_frames=8, pan=(15.0, 4.0), drop=()):
            M = seed["boxes"].shape[0]
            pan_v = torch.tensor([pan[0], pan[1], pan[0], pan[1]])
            g = torch.Generator().manual_seed(1)
            clip = []
            for k in range(n_frames):
                if k in drop:  # a missed-detection / occlusion frame
                    clip.append(Detections(
                        index=torch.zeros(0, dtype=torch.int64),
                        reid=torch.zeros(0, REID_DIM), bbox=torch.zeros(0, 4),
                        centroid=torch.zeros(0, 2),
                        category=torch.zeros(0, dtype=torch.int64),
                        score=torch.zeros(0), batch_size=[0]))
                    continue
                boxes = seed["boxes"] + k * pan_v
                centroid = torch.stack(
                    [(boxes[:, 0] + boxes[:, 2]) / 2,
                     (boxes[:, 1] + boxes[:, 3]) / 2], dim=-1)
                reid = seed["reid"] + 0.01 * torch.randn(
                    M, REID_DIM, generator=g)
                reid = torch.nn.functional.normalize(reid, dim=-1)
                order = torch.randperm(M, generator=g)
                clip.append(Detections(
                    index=torch.arange(M, dtype=torch.int64),
                    reid=reid[order], bbox=boxes[order], centroid=centroid[order],
                    category=seed["category"][order], score=seed["score"][order],
                    batch_size=[M]))
            return clip

        clip = make_clip(seed)
        print(f"clip: {len(clip)} frames x {seed['boxes'].shape[0]} detections")
    """),
    _md("""
        ## Are the objects separable by appearance?

        The migrated tracker matches on the `reid` embedding, so before
        running it, it's worth asking whether the embeddings actually
        tell the objects apart. The cosine similarity below is 1 on the
        diagonal; small off-diagonal values mean the cue is informative.
        Two objects of the same class (e.g. two cats) will show a higher
        off-diagonal value — exactly the case where motion has to help.
    """),
    _py("""
        sim = seed["reid"] @ seed["reid"].T   # reid rows are L2-normalized
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        im = ax.imshow(sim.numpy(), cmap="viridis", vmin=-1, vmax=1)
        ax.set_title("Appearance space — reid cosine similarity")
        n = len(seed["names"])
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(seed["names"], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(seed["names"], fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.show()
    """),
    _md("""
        ## The 1.x tracker we're migrating

        In unitrack 1.x the appearance tracker was built like this (see the
        [migration guide](../../docs/migration.md) for the full before /
        after and the symbol-by-symbol mapping):

        ```python
        # --- unitrack 1.x ---
        cost = ut.costs.Cosine(field="reid")
        cost = ut.costs.GateCost("category").wrap(cost)        # gate fused into cost
        tracker = ut.SimpleTracker(
            tracker=ut.MultiStageTracker(
                fields=[_build_field("reid"), _build_field("score"), ...],
                stages=[ut.stages.Gate(gate=SelectAndFilter("score", min_score),
                        then=[ut.stages.Association(
                            cost=cost,
                            assignment=ut.assignment.Jonker(threshold=0.9))])],
            ),
            memory=ut.TrackletMemory(states={
                "reid": ut.states.Value(torch.float, shape=(512,)), ...}),
        )
        ```

        Field-selection modules at the front, gating fused into the cost,
        states living on the memory, and a `SimpleTracker` bundling
        tracker + memory.
    """),
    _md("""
        ## Step 1 — the same tracker, in 2.0

        The 2.0 version reads as a small tree. Gates are first-class
        (`ClassGate`, `ScoreGate`) and combine with `Sequential`; `Gated`
        applies them to a `Pipe` that turns a `Cosine` cost into a match.
        States move onto the `Tracker` as `(Process, Observation,
        Initializer)` triples — a plain appearance cache is
        `Identity` + `Replace` + `FromDetectionField`.
    """),
    _py("""
        def feature_state(name, shape, dtype):
            \"\"\"A pure cache: no motion model, replace on match, seed from det.\"\"\"
            return State(
                schema=TensorSpec(shape=shape, dtype=dtype),
                process=Identity(name), observation=Replace(name),
                init=FromDetectionField(name))

        STATES = {
            "reid": feature_state("reid", (REID_DIM,), torch.float32),
            "bbox": feature_state("bbox", (4,), torch.float32),
            "centroid": feature_state("centroid", (2,), torch.float32),
            "category": feature_state("category", (), torch.int64),
            "score": feature_state("score", (), torch.float32),
        }

        def make_tracker(root, *, lifecycle=None, visibility=None):
            return ut.Tracker(
                root=root, states=dict(STATES),
                lifecycle=lifecycle or NoLifecycle(),
                visibility=visibility or IncludeAll())

        def run_clip(tracker, clip):
            ms = ut.MultiStream(tracker)
            return [ms.step(0, d, FrameContext.make(k, fps=15.0, stream_key=0))
                    for k, d in enumerate(clip)]

        def distinct_ids(results):
            ids = set()
            for r in results:
                ids.update(r.snapshot.id.tolist())
            return sorted(ids)
    """),
    _py("""
        appearance_tracker = make_tracker(
            Gated(
                gate=Sequential([
                    ClassGate("category"),          # cat can't match remote
                    ScoreGate("score", threshold=0.1),
                ]),
                then=Pipe(cost=Cosine("reid"),
                          assoc=Associate(Jonker(threshold=0.3))),
            )
        )
        app_results = run_clip(appearance_tracker, clip)
        print(f"appearance tracker -> stable ids: {distinct_ids(app_results)}")
    """),
    _py("""
        def plot_tracks(results, title, ax):
            cmap = plt.get_cmap("tab10")
            n_frames = max(1, len(results) - 1)
            for k, r in enumerate(results):
                snap = r.snapshot
                for n in range(snap.batch_size[0]):
                    c = snap.centroid[n]
                    ax.scatter(c[0].item(), c[1].item(),
                               color=cmap(int(snap.id[n]) % 10), s=45,
                               edgecolor="black", linewidth=0.4,
                               alpha=0.35 + 0.65 * k / n_frames)
            ax.set_title(title)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3)
            ax.invert_yaxis()  # image coordinates: y grows downward

        fig, ax = plt.subplots()
        plot_tracks(app_results,
                    "Migrated appearance tracker (color = track id)", ax)
        plt.show()
    """),
    _md("""
        One coherent trajectory per color: the cosine cost re-associates
        every shuffled detection to the right identity, exactly as the 1.x
        tracker did — but the 2.0 construction is the springboard for
        everything below.

        ## Why one cue isn't enough

        The migrated tracker matches on appearance alone. A pure
        **motion** tracker (IoU between boxes, à la SORT) is the classic
        alternative — but it breaks under fast camera motion: when the pan
        between frames exceeds the box size, consecutive boxes don't
        overlap, IoU is zero everywhere, and every object spawns a fresh ID
        each frame. We size a pan past the box width to force exactly that.
    """),
    _py("""
        box_w = (seed["boxes"][:, 2] - seed["boxes"][:, 0]).median().item()
        clip_fast = make_clip(seed, pan=(1.5 * box_w, 0.0))  # pan > box width

        iou_tracker = make_tracker(
            Pipe(cost=BoxIoU("bbox"), assoc=Associate(Jonker(threshold=0.7))))

        n_obj = seed["boxes"].shape[0]
        iou_ids = distinct_ids(run_clip(iou_tracker, clip_fast))
        app_ids = distinct_ids(run_clip(appearance_tracker, clip_fast))
        print(f"objects in scene:                 {n_obj}")
        print(f"IoU-only ids under a fast pan:    {len(iou_ids)}  (id explosion)")
        print(f"appearance ids under a fast pan:  {len(app_ids)}  (stable)")
    """),
    _md("""
        Appearance survives the pan; IoU collapses. But appearance alone is
        fragile the other way — two similar-looking objects that cross
        paths can swap IDs, where IoU would have held them apart. Real
        trackers want **both** cues. In 1.x, combining them meant a bespoke
        cost subclass or an awkward multi-stage hack. 2.0 makes it a
        one-liner — two ways.

        ## New in 2.0 #1 — parallel cost fusion

        `Parallel([...], merge=...)` runs several cost producers on the
        same snapshot and merges their matrices. `WeightedSum` lets you dial
        the appearance/motion balance. The whole fused cost is still a
        single `CostProducer`, so it drops straight into a `Pipe`.
    """),
    _py("""
        fused_tracker = make_tracker(
            Pipe(
                cost=Parallel(
                    children=[Cosine("reid"), BoxIoU("bbox")],
                    merge=WeightedSum([1.0, 0.5]),   # appearance + 0.5 * motion
                ),
                assoc=Associate(Jonker(threshold=1.0)),
            )
        )
        fused_results = run_clip(fused_tracker, clip_fast)
        print(f"parallel fusion (appearance + IoU) -> ids: "
              f"{distinct_ids(fused_results)}")
        print("Stable under the same fast pan that broke IoU-only — "
              "appearance carries it, IoU sharpens it when boxes do overlap.")
    """),
    _md("""
        ## New in 2.0 #2 — cascaded matching (ByteTrack-style)

        `Sequential([stage1, stage2, ...])` over match-producing stages
        chains *residuals*: stage 1 matches what it can, and only the
        leftover tracklets and detections flow into stage 2. A strict
        appearance pass followed by an IoU fallback is the canonical
        two-stage cascade — and it's just a list.
    """),
    _py("""
        cascaded_tracker = make_tracker(
            Sequential([
                # Stage 1: confident appearance matches only.
                Pipe(cost=Cosine("reid"),
                     assoc=Associate(Jonker(threshold=0.2))),
                # Stage 2: IoU fallback on whoever is left.
                Pipe(cost=BoxIoU("bbox"),
                     assoc=Associate(Jonker(threshold=0.7))),
            ])
        )
        cascaded_results = run_clip(cascaded_tracker, clip)
        print(f"cascaded (appearance -> IoU) -> ids: "
              f"{distinct_ids(cascaded_results)}")
    """),
    _md("""
        ## New in 2.0 #3 — lifecycle & gating

        The 1.x embedding tracker had no notion of track birth or death —
        every detection was a track, immediately. 2.0 adds a first-class
        **lifecycle**: `StandardLifecycle` runs the
        `Tentative → Active → Lost → Removed` state machine, and a
        **visibility** policy (`ConfirmedOnly`) hides flickering unconfirmed
        tracks from the caller. Below, three frames of detections go missing
        (an occlusion); a confirmed track goes `Lost`, then re-acquires its
        original ID when it reappears — within the `allow_reid` window.
    """),
    _py("""
        clip_gap = make_clip(seed, n_frames=10, pan=(4.0, 1.0), drop={4, 5, 6})
        lifecycle_tracker = make_tracker(
            Gated(gate=ClassGate("category"),
                  then=Pipe(cost=Cosine("reid"),
                            assoc=Associate(Jonker(threshold=0.3)))),
            lifecycle=StandardLifecycle(min_hits=2, max_age=2, allow_reid=4),
            visibility=ConfirmedOnly(),
        )
        ms = ut.MultiStream(lifecycle_tracker)
        names = ["Tentative", "Active", "Lost", "Removed"]
        rows = []
        for k in range(len(clip_gap)):
            res = ms.step(0, clip_gap[k], FrameContext.make(k, fps=15.0))
            snap = res.snapshot
            counts = [int((snap.status == s).sum()) for s in range(4)]
            rows.append(counts)
            present = clip_gap[k].batch_size[0] > 0
            tally = ", ".join(f"{names[s]}={counts[s]}" for s in range(4)
                              if counts[s])
            print(f"frame {k:2d}  {'det ' if present else 'MISS'}  {tally}")
    """),
    _py("""
        rows_t = torch.tensor(rows)
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ["#ffcc66", "#66cc66", "#aabbff", "#cccccc"]
        bottom = torch.zeros(len(rows))
        for s in range(4):
            ax.bar(range(len(rows)), rows_t[:, s].numpy(), bottom=bottom.numpy(),
                   color=colors[s], label=names[s], edgecolor="white")
            bottom = bottom + rows_t[:, s]
        for k in range(len(clip_gap)):
            if clip_gap[k].batch_size[0] == 0:
                ax.text(k, -0.4, "miss", ha="center", color="#aa0000", fontsize=7)
        ax.set_xlabel("frame"); ax.set_ylabel("tracklets")
        ax.set_title("Lifecycle through a 3-frame occlusion "
                     "(min_hits=2, max_age=2, allow_reid=4)")
        ax.legend(fontsize=8, ncol=4, loc="upper center")
        plt.tight_layout(); plt.show()
    """),
    _md("""
        ## New in 2.0 #4 — differentiable matching

        The headline capability: with a **soft** assignment, the matching
        is differentiable, so a tracking loss can backpropagate into the
        embeddings (and through them, the detector backbone). 1.x had no
        path to this — the Hungarian solve is a hard, non-differentiable
        argmax.

        `SoftAssignment` solves an entropy-regularised optimal-transport
        problem (Sinkhorn) and exposes the transport plan on
        `MatchOutcome.soft_plan`. We build a single soft matching stage,
        run it, and backprop a "match the right pairs" loss to the
        detection embeddings.
    """),
    _py("""
        def tracklets_from_reid(reid):
            n = reid.shape[0]
            return Tracklets(
                id=torch.arange(n, dtype=torch.int64),
                status=torch.full((n,), int(TrackletStatus.Active),
                                  dtype=torch.int8),
                hits=torch.ones(n, dtype=torch.int32),
                time_since_update=torch.zeros(n, dtype=torch.int32),
                age=torch.ones(n, dtype=torch.int32),
                frame_started=torch.zeros(n, dtype=torch.int32),
                frame_last_seen=torch.zeros(n, dtype=torch.int32),
                reid=reid, batch_size=[n])

        cs = tracklets_from_reid(seed["reid"].clone())
        torch.manual_seed(7)
        det_emb = (seed["reid"] + 0.05 * torch.randn_like(seed["reid"]))
        det_emb = det_emb.requires_grad_(True)         # the learnable input
        ds = Detections(index=torch.arange(det_emb.shape[0], dtype=torch.int64),
                        reid=det_emb, batch_size=[det_emb.shape[0]])

        soft_stage = Pipe(cost=Cosine("reid"),
                          assoc=Associate(SoftAssignment(epsilon=0.1)))
        match = soft_stage(cs, ds, FrameContext.make(0))
        plan = match.soft_plan                          # (N, M), differentiable

        loss = -plan.diagonal().sum()                   # reward correct matches
        loss.backward()
        print(f"soft transport plan shape: {tuple(plan.shape)}")
        print(f"plan total mass: {plan.detach().sum().item():.2f}  "
              f"(uniform OT marginals; mass concentrates on the diagonal)")
        print(f"gradient norm on detection embeddings: "
              f"{det_emb.grad.norm().item():.4f}  (non-zero -> it learns)")
    """),
    _py("""
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        im = ax.imshow(plan.detach().numpy(), cmap="magma", vmin=0)
        ax.set_title("Sinkhorn soft-assignment plan\\n(differentiable)")
        ax.set_xlabel("detection"); ax.set_ylabel("tracklet")
        for i in range(plan.shape[0]):
            for j in range(plan.shape[1]):
                ax.text(j, i, f"{plan[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
        plt.colorbar(im, ax=ax, label="transport mass")
        plt.tight_layout(); plt.show()
    """),
    _md("""
        The plan is sharp on the diagonal and the gradient is non-zero, so
        the embeddings receive a learning signal *through the matcher*. To
        get the same behavior inside a full tracker, build it with a single
        flag — `ut.Tracker(..., differentiable=True)` — and unitrack swaps
        every hard node (`Associate`, `Replace`, `StandardLifecycle`, …) for
        its soft companion automatically. Combined with `ClipTracker`, that
        is enough to train end-to-end across a clip.

        ## Where to go next

        - **[Migration guide](../../docs/migration.md)** — the full
          1.x → 2.0 symbol map and a line-by-line port of this tracker.
        - **Notebooks 1–6** — the library from the ground up: data model,
          cost & gate zoos, the pipeline tree, states & lifecycle.
        - **`unitrack.tracker.BatchTracker`** — vmap-batched multi-stream
          inference. **`ClipTracker`** — clip-based methods (MinVIS,
          DVIS++).
        - **Recipes** (`docs/recipes/`) — SORT and the overlap-IoU tracker
          as compact, self-contained builders.
    """),
]


# ---------------------------------------------------------------------------
# Build everything
# ---------------------------------------------------------------------------

NOTEBOOKS = {
    "01_quickstart.ipynb": NB_01_QUICKSTART,
    "02_data_model.ipynb": NB_02_DATA_MODEL,
    "03_costs_and_gates.ipynb": NB_03_COSTS_GATES,
    "04_pipeline_tree.ipynb": NB_04_PIPELINE,
    "05_states_and_lifecycle.ipynb": NB_05_STATES_LIFECYCLE,
    "06_cascaded_and_parallel.ipynb": NB_06_CASCADED,
    "07_migration.ipynb": NB_07_MIGRATION,
}


def main() -> None:
    print(f"Writing {len(NOTEBOOKS)} notebooks to {HERE}/")
    for filename, cells in NOTEBOOKS.items():
        write_notebook(HERE / filename, cells)
    print("\nDone.")


if __name__ == "__main__":
    main()
