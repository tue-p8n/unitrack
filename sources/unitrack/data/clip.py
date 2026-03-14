"""Clip-aware records: sequences of per-frame detections and tracklets."""

from __future__ import annotations

import dataclasses

import torch

from .detections import Detections
from .frame import FrameContext
from .match import MatchOutcome
from .tracklets import Tracklets

__all__ = [
    "ClipDetections",
    "ClipFrameContext",
    "ClipMatchOutcome",
    "ClipTracklets",
    "StackedClipMatch",
]


@dataclasses.dataclass(frozen=True, slots=True)
class StackedClipMatch:
    """
    Stacked view of a :class:`ClipMatchOutcome` along its matched-pair axis.

    Per-frame :class:`~unitrack.data.MatchOutcome` residual-index and soft-plan tensors
    have variable per-frame shapes (they depend on per-frame ``N`` and
    ``M``), so the stacked form carries only the matched-pair triple
    plus a per-row ``frame_idx`` annotation. Use
    :meth:`ClipMatchOutcome.frame_ranges` to slice back into per-frame
    views.

    Attributes
    ----------
    matched_pairs : torch.Tensor
        ``int64`` shape ``(K_total, 2)`` matched-pair indices.
    per_match_cost : torch.Tensor
        Float shape ``(K_total,)`` per-pair realised costs.
    frame_idx : torch.Tensor
        ``int64`` shape ``(K_total,)`` source-frame index for each row.

    """

    matched_pairs: torch.Tensor
    per_match_cost: torch.Tensor
    frame_idx: torch.Tensor

    @property
    def K_total(self) -> int:  # noqa: N802
        """Total number of matched pairs across all frames."""
        return int(self.matched_pairs.shape[0])


@dataclasses.dataclass(frozen=True, slots=True)
class ClipDetections:
    """
    K frames' worth of :class:`~unitrack.data.Detections`.

    Per-frame detection counts vary in MOT, so the canonical
    representation is a Python list of per-frame :class:`~unitrack.data.Detections`.
    :meth:`frame_lengths` and :meth:`frame_ranges` expose the per-frame
    index ranges; :meth:`stacked` produces a flat tensordict with a
    single leading batch axis equal to ``sum(frame_lengths)`` and a
    parallel ``frame_idx`` column for callers that want the clip as one
    tensor.

    Attributes
    ----------
    frames : list of Detections
        Per-frame detection records, in clip order.

    """

    frames: list[Detections]

    @property
    def K(self) -> int:  # noqa: N802
        """Number of frames in the clip."""
        return len(self.frames)

    @property
    def frame_lengths(self) -> torch.Tensor:
        """``(K,)`` int64 tensor of per-frame detection counts."""
        if not self.frames:
            return torch.zeros(0, dtype=torch.int64)
        device = self.frames[0].index.device
        return torch.tensor(
            [f.batch_size[0] for f in self.frames],
            dtype=torch.int64,
            device=device,
        )

    @property
    def frame_ranges(self) -> torch.Tensor:
        """``(K, 2)`` int64 tensor of ``[start, stop)`` ranges into :meth:`stacked`."""
        lens = self.frame_lengths
        stops = torch.cumsum(lens, dim=0)
        starts = stops - lens
        return torch.stack([starts, stops], dim=1)

    def stacked(self) -> Detections:
        """
        Return one :class:`~unitrack.data.Detections` flattening all K frames.

        Returns
        -------
        Detections
            Concatenated detections with an extra ``frame_idx`` field
            marking each row's source frame. Variable per-frame counts
            are preserved by concatenation rather than padding.

        Raises
        ------
        ValueError
            If any per-frame :class:`~unitrack.data.Detections` already carries a user
            field named ``frame_idx``; the stacked column would silently
            overwrite it and corrupt downstream cascade indexing.

        """
        for k, f in enumerate(self.frames):
            if "frame_idx" in f:
                msg = (
                    f"ClipDetections.stacked: frame[{k}] already carries a "
                    "'frame_idx' field; the stacked column would overwrite it. "
                    "Rename the user field before calling stacked()."
                )
                raise ValueError(msg)
        if not self.frames:
            return Detections.empty().set(  # type: ignore[invalid-return-type]
                "frame_idx", torch.zeros(0, dtype=torch.int64)
            )
        # tensordict overloads ``torch.cat`` for TensorDict-typed arguments.
        cat = torch.cat(self.frames, dim=0)  # type: ignore[no-matching-overload]
        frame_idx = torch.cat(
            [
                torch.full(
                    (f.batch_size[0],),
                    k,
                    dtype=torch.int64,
                    device=f.index.device,
                )
                for k, f in enumerate(self.frames)
            ]
        )
        return cat.set("frame_idx", frame_idx)  # type: ignore[invalid-return-type]


@dataclasses.dataclass(frozen=True, slots=True)
class ClipTracklets:
    """
    K aligned :class:`~unitrack.data.Tracklets` snapshots, one per frame.

    Row ``n`` at frame ``k`` is the same identity as row ``n`` at frame
    ``k + 1``. The list-of-frames form is canonical; :meth:`stacked`
    produces a flat tensordict view marked with ``frame_idx`` for
    kernels that want a single tensor input.

    Attributes
    ----------
    frames : list of Tracklets
        Per-frame snapshots, in clip order, all sharing the same row
        order by identity.

    """

    frames: list[Tracklets]

    @property
    def K(self) -> int:  # noqa: N802
        """Number of frames in the clip."""
        return len(self.frames)

    @property
    def frame_lengths(self) -> torch.Tensor:
        """``(K,)`` int64 tensor of per-frame row counts."""
        if not self.frames:
            return torch.zeros(0, dtype=torch.int64)
        device = self.frames[0].id.device
        return torch.tensor(
            [f.batch_size[0] for f in self.frames],
            dtype=torch.int64,
            device=device,
        )

    @property
    def frame_ranges(self) -> torch.Tensor:
        """``(K, 2)`` int64 tensor of ``[start, stop)`` ranges into :meth:`stacked`."""
        lens = self.frame_lengths
        stops = torch.cumsum(lens, dim=0)
        return torch.stack([stops - lens, stops], dim=1)

    def stacked(self) -> Tracklets:
        """
        Return one :class:`~unitrack.data.Tracklets` flattening all K frames.

        Returns
        -------
        Tracklets
            Concatenated snapshots with an extra ``frame_idx`` field
            marking each row's source frame.

        Raises
        ------
        ValueError
            If any per-frame :class:`~unitrack.data.Tracklets` already carries a user
            field named ``frame_idx``; the stacked column would silently
            overwrite it.

        """
        for k, f in enumerate(self.frames):
            if "frame_idx" in f:
                msg = (
                    f"ClipTracklets.stacked: frame[{k}] already carries a "
                    "'frame_idx' field; the stacked column would overwrite it. "
                    "Rename the user field before calling stacked()."
                )
                raise ValueError(msg)
        if not self.frames:
            return Tracklets.empty().set(  # type: ignore[invalid-return-type]
                "frame_idx", torch.zeros(0, dtype=torch.int64)
            )
        # tensordict overloads ``torch.cat`` for TensorDict-typed arguments.
        cat = torch.cat(self.frames, dim=0)  # type: ignore[no-matching-overload]
        frame_idx = torch.cat(
            [
                torch.full(
                    (f.batch_size[0],),
                    k,
                    dtype=torch.int64,
                    device=f.id.device,
                )
                for k, f in enumerate(self.frames)
            ]
        )
        return cat.set("frame_idx", frame_idx)  # type: ignore[invalid-return-type]


@dataclasses.dataclass(frozen=True, slots=True)
class ClipFrameContext:
    """
    Per-frame contexts plus clip-level metadata.

    Attributes
    ----------
    frame_contexts : list of FrameContext
        Per-frame contexts in clip order.
    clip_idx : int
        Clip-level identifier, useful when batching multiple clips.

    """

    frame_contexts: list[FrameContext]
    clip_idx: int = 0

    @property
    def K(self) -> int:  # noqa: N802
        """Number of frames in the clip."""
        return len(self.frame_contexts)

    @property
    def frame_lengths(self) -> torch.Tensor:
        """``(K,)`` int64 tensor — each frame contributes exactly one context."""
        if not self.frame_contexts:
            return torch.zeros(0, dtype=torch.int64)
        device = self.frame_contexts[0].frame_idx.device
        return torch.ones(len(self.frame_contexts), dtype=torch.int64, device=device)

    @property
    def frame_ranges(self) -> torch.Tensor:
        """``(K, 2)`` int64 tensor of ``[start, stop)`` ranges into :meth:`stacked`."""
        lens = self.frame_lengths
        stops = torch.cumsum(lens, dim=0)
        return torch.stack([stops - lens, stops], dim=1)

    def stacked(self) -> FrameContext:
        """
        Return a single :class:`FrameContext` with a leading ``(K,)`` batch axis.

        Returns
        -------
        FrameContext
            Stacked context across all ``K`` frames.

        Raises
        ------
        ValueError
            If :attr:`frame_contexts` is empty.

        """
        if not self.frame_contexts:
            msg = "stacked() on empty ClipFrameContext"
            raise ValueError(msg)
        return torch.stack(self.frame_contexts, dim=0)  # type: ignore[no-matching-overload]

    @classmethod
    def make(  # noqa: PLR0913
        cls,
        *,
        start_frame: int,
        K: int,  # noqa: N803
        fps: float = 1.0,
        stream_key: int = 0,
        clip_idx: int = 0,
        device: torch.types.Device | None = None,
    ) -> ClipFrameContext:
        """
        Build a :class:`~unitrack.data.ClipFrameContext` from Python scalars.

        Parameters
        ----------
        start_frame
            Frame index of the first frame in the clip.
        K
            Number of frames.
        fps
            Frame rate; sets ``delta = 1.0 / fps`` for every frame.
        stream_key
            Stream identifier shared by all frames.
        clip_idx
            Clip-level identifier.
        device
            Device for the packed scalar tensors.

        Returns
        -------
        ClipFrameContext
            The packed clip context.

        """
        contexts = [
            FrameContext.make(
                frame_idx=start_frame + k,
                delta=1.0 / fps,
                fps=fps,
                stream_key=stream_key,
                device=device,
            )
            for k in range(K)
        ]
        return cls(frame_contexts=contexts, clip_idx=clip_idx)


@dataclasses.dataclass(frozen=True, slots=True)
class ClipMatchOutcome:
    """
    Per-frame :class:`~unitrack.data.MatchOutcome` records for a clip.

    Attributes
    ----------
    frames : list of MatchOutcome
        Per-frame match outcomes in clip order.

    """

    frames: list[MatchOutcome]

    @property
    def K(self) -> int:  # noqa: N802
        """Number of frames in the clip."""
        return len(self.frames)

    @property
    def frame_lengths(self) -> torch.Tensor:
        """``(K,)`` int64 tensor of per-frame matched-pair counts."""
        if not self.frames:
            return torch.zeros(0, dtype=torch.int64)
        device = self.frames[0].matched_pairs.device
        return torch.tensor(
            [f.matched_pairs.shape[0] for f in self.frames],
            dtype=torch.int64,
            device=device,
        )

    @property
    def frame_ranges(self) -> torch.Tensor:
        """``(K, 2)`` int64 ``[start, stop)`` ranges over the matched-pair axis."""
        lens = self.frame_lengths
        stops = torch.cumsum(lens, dim=0)
        return torch.stack([stops - lens, stops], dim=1)

    def stacked(self) -> StackedClipMatch:
        """
        Return a :class:`StackedClipMatch` view across all K frames.

        Per-frame :class:`~unitrack.data.MatchOutcome` rows are concatenated along
        their matched-pair axis; the residual-index arrays and soft-plan
        tensors cannot share a uniform shape across frames (variable
        ``N`` and ``M``), so the stacked form carries only the
        matched-pair triple plus a per-row ``frame_idx`` annotation. Use
        :meth:`frame_ranges` to slice back into per-frame views.

        Returns
        -------
        StackedClipMatch
            Concatenated matched-pair view.

        """
        if not self.frames:
            return StackedClipMatch(
                matched_pairs=torch.zeros((0, 2), dtype=torch.int64),
                per_match_cost=torch.zeros(0, dtype=torch.float32),
                frame_idx=torch.zeros(0, dtype=torch.int64),
            )
        pairs = torch.cat([f.matched_pairs for f in self.frames], dim=0)
        costs = torch.cat([f.per_match_cost for f in self.frames], dim=0)
        frame_idx = torch.cat(
            [
                torch.full(
                    (f.matched_pairs.shape[0],),
                    k,
                    dtype=torch.int64,
                    device=f.matched_pairs.device,
                )
                for k, f in enumerate(self.frames)
            ]
        )
        return StackedClipMatch(
            matched_pairs=pairs,
            per_match_cost=costs,
            frame_idx=frame_idx,
        )
