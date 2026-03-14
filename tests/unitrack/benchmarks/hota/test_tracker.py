import pytest
import torch
from unitrack import MultiStream
from unitrack.benchmarks.hota.tracker import (
    CHI2_GATE,
    TRACKER_REGISTRY,
    build_cascade_tracker,
    build_cosine_tracker,
    build_kalman_tracker,
    build_learned_tracker,
    build_mask_tracker,
    ids_per_detection,
)
from unitrack.data import Detections, FrameContext


def _dets(masks, cats):
    m = masks.shape[0]
    return Detections(
        index=torch.arange(m),
        mask=masks,
        category=torch.tensor(cats, dtype=torch.int64),
        batch_size=[m],
    )


def _mask(h, w, box):
    t = torch.zeros((h, w), dtype=torch.bool)
    y0, x0, y1, x1 = box
    t[y0:y1, x0:x1] = True
    return t


def _ctx(frame_idx):
    return FrameContext.make(frame_idx, fps=15.0, stream_key=0)


def test_consistent_instance_keeps_id_and_spawn_gets_new_id():
    h = w = 8
    ms = MultiStream(build_mask_tracker(height=h, width=w, cost_threshold=0.5))
    # frame 0: one instance A (class 11)
    a = _mask(h, w, (0, 0, 4, 4))
    r0 = ms.step(0, _dets(a[None], [11]), FrameContext.make(0, fps=15.0, stream_key=0))
    id0 = ids_per_detection(r0, n_dets=1)
    assert id0.tolist()[0] >= 0
    # frame 1: A again (overlapping) + a NEW instance B (class 13)
    b = _mask(h, w, (4, 4, 8, 8))
    r1 = ms.step(
        0,
        _dets(torch.stack([a, b]), [11, 13]),
        FrameContext.make(1, fps=15.0, stream_key=0),
    )
    id1 = ids_per_detection(r1, n_dets=2)
    assert id1[0].item() == id0[0].item()  # A keeps its id
    assert id1[1].item() != id0[0].item()  # B is a new track
    assert (id1 >= 0).all()


def test_ids_per_detection_handles_permuted_order():
    # The matched detection is emitted AFTER the new one, so a naive
    # "snapshot.id[-n:]" heuristic would mis-assign; index mapping must hold.
    h = w = 8
    ms = MultiStream(build_mask_tracker(height=h, width=w, cost_threshold=0.5))
    a = _mask(h, w, (0, 0, 4, 4))  # instance A, class 11
    r0 = ms.step(0, _dets(a[None], [11]), FrameContext.make(0, fps=15.0, stream_key=0))
    id0 = ids_per_detection(r0, n_dets=1)[0].item()
    # frame 1: emit [B (new, class 13, non-overlapping), A (matched, class 11)]
    b = _mask(h, w, (4, 4, 8, 8))
    r1 = ms.step(
        0,
        _dets(torch.stack([b, a]), [13, 11]),
        FrameContext.make(1, fps=15.0, stream_key=0),
    )
    out = ids_per_detection(r1, n_dets=2)
    assert out[1].item() == id0  # the second (matched) detection is A's id
    assert out[0].item() != id0  # the first (new) detection is a fresh id
    assert (out >= 0).all()


def test_cost_threshold_controls_matching():
    h = w = 10
    ms = MultiStream(build_mask_tracker(height=h, width=w, cost_threshold=0.4))
    a = _mask(h, w, (0, 0, 10, 5))  # left half
    r0 = ms.step(0, _dets(a[None], [11]), FrameContext.make(0, fps=15.0, stream_key=0))
    id0 = ids_per_detection(r0, 1)[0].item()
    # ~25% IoU overlap -> cost ~0.75 > 0.4 -> NOT matched (new id)
    a2 = _mask(h, w, (0, 3, 10, 8))
    r1 = ms.step(0, _dets(a2[None], [11]), FrameContext.make(1, fps=15.0, stream_key=0))
    id1 = ids_per_detection(r1, 1)[0].item()
    assert id1 != id0  # below-threshold overlap breaks the track


# --- appearance / cascade / motion trackers -------------------------------


def _embed_dets(*, embeddings, cats, scores=None, masks=None, centroids=None):
    m = embeddings.shape[0]
    fields = {
        "embedding": embeddings,
        "category": torch.tensor(cats, dtype=torch.int64),
    }
    if scores is not None:
        fields["score"] = torch.tensor(scores, dtype=torch.float32)
    if masks is not None:
        fields["mask"] = masks
    if centroids is not None:
        fields["centroid"] = centroids
    return Detections(index=torch.arange(m), batch_size=[m], **fields)


def _unit(idx, dim=256):
    v = torch.zeros(dim, dtype=torch.float32)
    v[idx] = 1.0
    return v


def test_cosine_tracker_keeps_stable_id_over_clip():
    dim = 256
    ms = MultiStream(build_cosine_tracker(height=8, width=8, embed_dim=dim))
    emb = _unit(3, dim)[None]
    last = None
    for f in range(3):
        # Tiny perturbation each frame; cosine still ~1 with the track.
        e = emb.clone()
        e[0, 7] = 0.01 * f
        r = ms.step(0, _embed_dets(embeddings=e, cats=[11]), _ctx(f))
        cur = ids_per_detection(r, n_dets=1)[0].item()
        assert cur >= 0
        if last is not None:
            assert cur == last
        last = cur


def test_cosine_tracker_distinct_appearance_is_new_track():
    dim = 256
    ms = MultiStream(build_cosine_tracker(height=8, width=8, embed_dim=dim))
    r0 = ms.step(0, _embed_dets(embeddings=_unit(0, dim)[None], cats=[11]), _ctx(0))
    id0 = ids_per_detection(r0, n_dets=1)[0].item()
    # Orthogonal embedding -> cosine cost 1.0 > threshold -> a fresh id.
    r1 = ms.step(0, _embed_dets(embeddings=_unit(1, dim)[None], cats=[11]), _ctx(1))
    id1 = ids_per_detection(r1, n_dets=1)[0].item()
    assert id1 != id0


def test_cascade_high_score_matched_by_appearance():
    # A high-score instance whose mask moves a lot but whose appearance is stable
    # is held by the cosine (appearance) stage, not mask-IoU.
    dim = 256
    h = w = 8
    ms = MultiStream(build_cascade_tracker(height=h, width=w, hi=0.5, embed_dim=dim))
    emb = _unit(5, dim)[None]
    m0 = _mask(h, w, (0, 0, 4, 4))[None]
    r0 = ms.step(
        0,
        _embed_dets(embeddings=emb, cats=[11], scores=[0.9], masks=m0),
        _ctx(0),
    )
    id0 = ids_per_detection(r0, n_dets=1)[0].item()
    # Frame 1: same appearance, NON-overlapping mask (mask-IoU would fail).
    m1 = _mask(h, w, (4, 4, 8, 8))[None]
    r1 = ms.step(
        0,
        _embed_dets(embeddings=emb.clone(), cats=[11], scores=[0.9], masks=m1),
        _ctx(1),
    )
    id1 = ids_per_detection(r1, n_dets=1)[0].item()
    assert id1 == id0


def test_cascade_low_score_matched_by_mask_iou():
    # A low-score instance (below hi) is routed to the mask-IoU stage and held
    # by overlap even though its embedding drifts to orthogonal.
    dim = 256
    h = w = 8
    ms = MultiStream(build_cascade_tracker(height=h, width=w, hi=0.5, embed_dim=dim))
    m = _mask(h, w, (0, 0, 6, 6))[None]
    r0 = ms.step(
        0,
        _embed_dets(embeddings=_unit(0, dim)[None], cats=[11], scores=[0.2], masks=m),
        _ctx(0),
    )
    id0 = ids_per_detection(r0, n_dets=1)[0].item()
    # Overlapping mask, orthogonal embedding, still low score -> mask-IoU holds it.
    r1 = ms.step(
        0,
        _embed_dets(
            embeddings=_unit(1, dim)[None], cats=[11], scores=[0.2], masks=m.clone()
        ),
        _ctx(1),
    )
    id1 = ids_per_detection(r1, n_dets=1)[0].item()
    assert id1 == id0


def test_kalman_tracker_tracks_gentle_drift():
    # A gently drifting instance (+0.3 px/frame) stays one track: each
    # measurement sits well inside the motion gate around the predicted
    # centroid while the constant-velocity state acquires the drift.
    ms = MultiStream(build_kalman_tracker(height=64, width=64))
    last = None
    for f in range(4):
        c = torch.tensor([[10.0 + 0.3 * f, 10.0 + 0.3 * f]], dtype=torch.float32)
        r = ms.step(0, _embed_dets_centroid(c, [11]), _ctx(f))
        cur = ids_per_detection(r, n_dets=1)[0].item()
        assert cur >= 0
        if last is not None:
            assert cur == last
        last = cur


def test_kalman_motion_gate_rejects_teleport():
    ms = MultiStream(build_kalman_tracker(height=64, width=64))
    c0 = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
    r0 = ms.step(0, _embed_dets_centroid(c0, [11]), _ctx(0))
    id0 = ids_per_detection(r0, n_dets=1)[0].item()
    # A cross-image teleport is far outside the chi-squared gate -> a fresh id.
    c1 = torch.tensor([[60.0, 60.0]], dtype=torch.float32)
    r1 = ms.step(0, _embed_dets_centroid(c1, [11]), _ctx(1))
    id1 = ids_per_detection(r1, n_dets=1)[0].item()
    assert id1 != id0


def test_kalman_motion_gate_isolated_from_jonker_threshold():
    # Isolate the MotionGate: with a LOOSE Jonker threshold (which would happily
    # admit the teleport on cost alone) the only thing that can reject a teleport
    # is the tight MotionGate. The positive control (loose gate too) proves the
    # cost path admits the teleport, so the rejection below is the gate's doing.
    loose = 1e6

    # Positive control: loose gate + loose jonker -> teleport KEEPS its id.
    ms_loose = MultiStream(
        build_kalman_tracker(
            height=64, width=64, max_chi2=loose, jonker_threshold=loose
        )
    )
    c0 = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
    r0 = ms_loose.step(0, _embed_dets_centroid(c0, [11]), _ctx(0))
    id0 = ids_per_detection(r0, n_dets=1)[0].item()
    c1 = torch.tensor([[60.0, 60.0]], dtype=torch.float32)
    r1 = ms_loose.step(0, _embed_dets_centroid(c1, [11]), _ctx(1))
    assert ids_per_detection(r1, n_dets=1)[0].item() == id0  # admitted by cost

    # Test: loose jonker + TIGHT gate -> the same teleport gets a fresh id, so
    # only the MotionGate can explain the rejection.
    ms_gate = MultiStream(
        build_kalman_tracker(
            height=64, width=64, max_chi2=CHI2_GATE, jonker_threshold=loose
        )
    )
    g0 = ms_gate.step(0, _embed_dets_centroid(c0, [11]), _ctx(0))
    gid0 = ids_per_detection(g0, n_dets=1)[0].item()
    g1 = ms_gate.step(0, _embed_dets_centroid(c1, [11]), _ctx(1))
    assert ids_per_detection(g1, n_dets=1)[0].item() != gid0  # rejected by gate


def test_cosine_class_gate_separates_same_appearance_distinct_classes():
    # Two same-frame instances of DIFFERENT categories carrying near-identical
    # embeddings, constructed so APPEARANCE ALONE prefers the cross-class match:
    # in frame 1 each detection's embedding is nearer the OTHER class's track. The
    # diagonal (same-class) match is therefore the cosine-costlier one, so only
    # ClassGate masking the cross-class pairs can keep the ids separated.
    dim = 256
    ms = MultiStream(build_cosine_tracker(height=8, width=8, embed_dim=dim))
    e = _unit(7, dim)
    d = 1e-3 * _unit(8, dim)
    e_a, e_b = e + d, e - d
    # frame 0: det_A (cls 11) -> track0 ; det_B (cls 13) -> track1
    r0 = ms.step(
        0, _embed_dets(embeddings=torch.stack([e_a, e_b]), cats=[11, 13]), _ctx(0)
    )
    ids0 = ids_per_detection(r0, n_dets=2)
    assert ids0[0].item() != ids0[1].item()  # distinct classes -> distinct ids
    # frame 1: det0 (cls 11) carries e_b (looks like track1); det1 (cls 13)
    # carries e_a (looks like track0). Appearance favors the cross-class swap.
    r1 = ms.step(
        0, _embed_dets(embeddings=torch.stack([e_b, e_a]), cats=[11, 13]), _ctx(1)
    )
    ids1 = ids_per_detection(r1, n_dets=2)
    assert ids1[0].item() == ids0[0].item()  # class-11 instance keeps its id
    assert ids1[1].item() == ids0[1].item()  # class-13 instance keeps its id
    assert ids1[0].item() != ids1[1].item()  # never merged across classes


def test_cascade_class_gate_separates_same_appearance_distinct_classes():
    # Same cross-class isolation for the cascade's appearance stage: both
    # detections are high-score (>= hi) so they stay in the cosine stage, and the
    # frame-1 embeddings are swapped so appearance prefers the cross-class match.
    dim = 256
    h = w = 8
    ms = MultiStream(build_cascade_tracker(height=h, width=w, hi=0.5, embed_dim=dim))
    e = _unit(7, dim)
    d = 1e-3 * _unit(8, dim)
    e_a, e_b = e + d, e - d
    m0 = torch.stack([_mask(h, w, (0, 0, 4, 4)), _mask(h, w, (4, 4, 8, 8))])
    r0 = ms.step(
        0,
        _embed_dets(
            embeddings=torch.stack([e_a, e_b]),
            cats=[11, 13],
            scores=[0.9, 0.9],
            masks=m0,
        ),
        _ctx(0),
    )
    ids0 = ids_per_detection(r0, n_dets=2)
    assert ids0[0].item() != ids0[1].item()
    r1 = ms.step(
        0,
        _embed_dets(
            embeddings=torch.stack([e_b, e_a]),
            cats=[11, 13],
            scores=[0.9, 0.9],
            masks=m0.clone(),
        ),
        _ctx(1),
    )
    ids1 = ids_per_detection(r1, n_dets=2)
    assert ids1[0].item() == ids0[0].item()
    assert ids1[1].item() == ids0[1].item()
    assert ids1[0].item() != ids1[1].item()


def _embed_dets_centroid(centroids, cats):
    m = centroids.shape[0]
    return Detections(
        index=torch.arange(m),
        centroid=centroids,
        category=torch.tensor(cats, dtype=torch.int64),
        batch_size=[m],
    )


def test_registry_exposes_all_factories():
    assert set(TRACKER_REGISTRY) >= {
        "maskiou",
        "cosine",
        "cascade",
        "kalman",
        "learned",
    }
    for key, factory in TRACKER_REGISTRY.items():
        # ``learned`` is checkpoint-dependent (its build succeeds or raises
        # depending on whether the committed checkpoint is present); its
        # contract is covered by the dedicated tests below rather than here.
        if key == "learned":
            continue
        ms = factory(16, 16)
        assert isinstance(ms, MultiStream), key


def test_learned_tracker_missing_checkpoint_raises_clearly(tmp_path):
    missing = tmp_path / "no_such_filter.safetensors"
    with pytest.raises(FileNotFoundError) as exc:
        build_learned_tracker(height=16, width=16, checkpoint=missing)
    # The error must point the user at the training script.
    assert "train_learned" in str(exc.value)


def test_learned_tracker_builds_and_tracks_from_checkpoint(tmp_path):
    # With a (synthetic) checkpoint present, the factory loads the modules and
    # runs end-to-end. A near-identity propagator keeps one stable id, exercising
    # the LearnedProcess/LearnedObservation wiring.
    from unitrack.benchmarks.hota.learned_modules import Fuser, Propagator
    from unitrack.benchmarks.hota.train_learned import save_checkpoint

    dim = 256
    prop = Propagator(dim)
    fuse = Fuser(dim)
    # Zero the propagator residual so it propagates the (renormalized) input.
    with torch.no_grad():
        prop.net[-1].weight.zero_()
        prop.net[-1].bias.zero_()
    ckpt = tmp_path / "learned_filter.safetensors"
    save_checkpoint(prop, fuse, ckpt)

    tracker = build_learned_tracker(height=8, width=8, embed_dim=dim, checkpoint=ckpt)
    ms = MultiStream(tracker)
    emb = torch.nn.functional.normalize(torch.randn(1, dim), dim=-1)
    last = None
    for f in range(3):
        dets = Detections(
            index=torch.arange(1),
            embedding=emb.clone(),
            category=torch.tensor([11], dtype=torch.int64),
            batch_size=[1],
        )
        r = ms.step(0, dets, _ctx(f))
        cur = ids_per_detection(r, n_dets=1)[0].item()
        assert cur >= 0
        if last is not None:
            assert cur == last
        last = cur
