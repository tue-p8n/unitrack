r"""This module defines the `TrackletMemory` class, which is a container module for tracking predictions/paths, commonly referred to as "tracklets" in the literature.

A tracklet represents a generic collection of states, such as objects in a scene. The states of the tracklets are implemented in subclasses of the `Field` class, which can be used to compute a distance matrix between tracklets at different frames.
"""

import typing as T
from enum import StrEnum

import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor, nn

from .consts import KEY_ACTIVE, KEY_DELTA, KEY_FRAME, KEY_ID, KEY_INDEX, KEY_START
from .debug import check_debug_enabled
from .states import DEFAULT_STATE_SLOTS, State
from .states import Value as ValueState

__all__ = ["TrackletMemory"]


class TrackletMemoryWriteReturnType(StrEnum):
    """Enum for the return type of the TrackletMemory write method."""

    NONE = "none"
    ID = "id"
    STATE = "state"


class TrackletMemory(nn.Module):
    """A memory module that tracks the states of tracklets over time and provides
    a list of IDs to assign the the new detections at the current frame when
    the memory is written to.

    Properties
    ----------
    states
        A ``torch.nn.ModuleDict`` mapping of states that are tracked for each
        tracklet.
    frame
        The frame number of the last observation.
    count
        The number of frames that have been observed.
    fps
        The frames per second of the video (fixed).
    max_id
        The maximum ID that can be assigned to a tracklet (fixed).
    max_lost
        The maximum amount of frames a tracklet can be lost before it is removed.
    auto_reset
        Whether to automatically reset the memory when the current frame is larger
        than the stored frame (fixed).
    """

    states: nn.ModuleDict
    frame: Tensor
    write_count: Tensor
    tracklet_count: Tensor
    index_range: Tensor
    index_mask: Tensor

    fps: T.Final[float]
    max_id: T.Final[int]
    max_lost: T.Final[int]
    auto_reset: T.Final[bool]

    def __init__(
        self,
        states: T.Mapping[str, State],
        max_id: int = 2**15,
        max_lost: int = 1,
        auto_reset: bool = True,
        fps: float = 15,
        slots: int = DEFAULT_STATE_SLOTS,
    ):
        super().__init__()

        assert max_id > 0, max_id
        assert len(states) > 0, len(states)

        self.fps = float(fps)
        self.max_id = int(max_id)
        self.max_lost = int(max_lost)
        self.auto_reset = auto_reset  # reset the memory when the current frame is larger than the stored frame

        self.states = nn.ModuleDict()
        self.states[KEY_FRAME] = ValueState(torch.int32, slots=slots)
        self.states[KEY_START] = ValueState(torch.int32, slots=slots)
        self.states[KEY_ID] = ValueState(torch.long, slots=slots)
        self.states[KEY_ACTIVE] = ValueState(torch.bool, slots=slots)
        for key, state in states.items():
            assert key not in self.states, f"State {key!r} already exists!"
            state.slots = slots
            self.states[key] = state

        # NOTE: we use buffers for compatability with StatefulTracker, which uses
        # vmap to parallelize the memory across video streams.
        self.register_buffer(
            "index_range",
            torch.arange(slots, dtype=torch.long, requires_grad=False),
            persistent=False,
        )
        self.register_buffer(
            "index_mask",
            torch.zeros((slots,), dtype=torch.bool, requires_grad=False),
            persistent=False,
        )
        self.register_buffer(
            "frame",
            torch.tensor(-1, dtype=torch.int, requires_grad=False),
            persistent=False,
        )
        self.register_buffer(
            "write_count",
            torch.tensor(0, dtype=torch.int, requires_grad=False),
            persistent=False,
        )
        self.register_buffer(
            "tracklet_count",
            torch.tensor(0, dtype=torch.int, requires_grad=False),
            persistent=False,
        )

    def __len__(self) -> int:
        """Return the number of times that a new observation has been comitted to the
        memory.

        Note that this does not necessarily correspond to the number of frames that
        have passed in the upstream application, as the memory write operation may
        not have been called at each frame by the user.
        """
        return int(self.write_count.item())

    @torch.no_grad()
    def write(
        self,
        ctx: TensorDictBase,
        obs: TensorDictBase,
        new: TensorDictBase,
    ) -> Tensor:
        """Apply the updated observations and new detections to the tracklets states.

        Both the updated observations and new detections have a key `KEY_INDEX` that
        corresponds to a `torch.long` tensor of indices. This value is used to
        determine the assignment of detections to tracklets at the current frame, i.e.
        the timestep for which the memory is being written to. All positive indices
        must be unique.
        Users can use these indices to recover the sequence in which detections were
        made, such that the IDs returned by this method can be assigned to the
        detections in the same order that they were found in the model.


        Parameters
        ----------
        ctx
            The observation context. See :meth:`read` for more information.
        obs
            The evolved state of the tracklets returned by :meth:`read`. The entry
            at `KEY_INDEX` is positive when it is assigned to a detection, or negative
            when it has not been assigned. Note that when the tracklet is not assigned,
            this does not imply that it will also become inactive (`KEY_ACTIVE`). This
            is up to the user to explicitly set (e.g. a use case could be to allow
            re-identification of tracklets that were lost for a few frames).
        new
            The new detections at the current frame, which have not been assigned to
            any of the tracklets passed as the `obs` argument. Must have an entry
            with key `KEY_INDEX` similar to the `obs` argument, but without negative
            values.

        Returns
        -------
        Tensor[N]
            The complete set of IDs of the context (which is also updated in-place).
            The length $N$ is equal to the amount of positive indices at the
            key `KEY_INDEX` in the `obs` and `new` arguments.

        """
        return T.cast(
            Tensor,
            self._write(ctx, obs, new, return_type=TrackletMemoryWriteReturnType.ID),
        )

    @torch.no_grad()
    def write_with_observe(
        self,
        ctx: TensorDictBase,
        obs: TensorDictBase,
        new: TensorDictBase,
    ) -> TensorDictBase:
        """Variant of :meth:`write` that also returns the observed state of the tracklets."""
        return T.cast(
            TensorDictBase,
            self._write(ctx, obs, new, return_type=TrackletMemoryWriteReturnType.STATE),
        )

    def _write(
        self,
        ctx: TensorDictBase,
        obs: TensorDictBase,
        new: TensorDictBase,
        *,
        return_type: TrackletMemoryWriteReturnType = TrackletMemoryWriteReturnType.ID,
    ) -> Tensor | TensorDictBase | None:
        assert KEY_INDEX in obs.keys(), f"Key {KEY_INDEX} not found in obs keys."  # noqa: SIM118
        assert KEY_INDEX in new.keys(), f"Key {KEY_INDEX} not found in new keys."  # noqa: SIM118

        # Evolve the memory states
        frame = int(ctx.get(KEY_FRAME).item())
        self.frame.fill_(frame)
        self.write_count += 1

        obs_track_idxs, obs_ids = self._update_states(obs, frame=frame)
        new_track_idxs, new_ids = self._extend_states(new, frame=frame)

        # Apply selection and reordering based on the indices provided in obs and new
        idx_obs = obs.get(KEY_INDEX)
        obs_mask = idx_obs >= 0
        idx_obs_valid = idx_obs[obs_mask]
        idx_obs_amt = idx_obs_valid.numel()
        if idx_obs_valid.unique().numel() != idx_obs_amt:
            msg = (
                "Positive indices of observations are not unique! "
                f"Got: {idx_obs.tolist()}"
            )
            raise ValueError(msg)

        idx_new = new.get(KEY_INDEX)
        idx_new_amt = idx_new.numel()
        if idx_new_amt > 0 and idx_new.min() < 0:
            msg = f"New detections contain negative indices!Got: {idx_new.tolist()}"
            raise ValueError(msg)
        if idx_new.unique().numel() != idx_new.numel():
            msg = f"Indices of new detections are not unique! Got: {idx_new.tolist()}"
            raise ValueError(msg)

        if check_debug_enabled():
            pass

        idx_all = torch.cat((idx_obs_valid, idx_new), dim=0)
        idx_all_amt = idx_all.numel()
        if idx_all.unique().numel() != idx_all_amt:
            msg = (
                "Combined indices of observations and new detections are not unique! "
                f"Got: {idx_all.tolist()}"
            )
            raise ValueError(msg)
        if idx_all_amt > 0 and idx_all.min() != 0:
            msg = (
                f"Indices must start at 0! Got: {idx_all.min()} for {idx_all.tolist()}"
            )
            raise ValueError(msg)
        if idx_all_amt > 0 and idx_all.max() != idx_all_amt - 1:
            msg = (
                "Indices must be contiguous! Got: "
                f"{idx_all.min()} to {idx_all.max()} for {idx_all.tolist()}"
            )
            raise ValueError(msg)

        # A list of IDs to assign to detections in the next
        # frame.
        result_idx = torch.full_like(idx_all, -1)
        result_ids = torch.zeros_like(idx_all)
        if idx_obs_amt > 0:
            result_idx[idx_obs_valid] = obs_track_idxs[obs_mask]
            result_ids[idx_obs_valid] = obs_ids[obs_mask]
        if idx_new_amt > 0:
            result_idx[idx_new] = new_track_idxs
            result_ids[idx_new] = new_ids

        assert torch.all(result_idx >= 0), result_idx
        assert torch.all(result_ids > 0), result_ids

        # Return value
        match return_type:
            case TrackletMemoryWriteReturnType.ID:
                return result_ids
            case TrackletMemoryWriteReturnType.STATE:
                return TensorDict(
                    {
                        id: state.observe(result_idx)
                        for id, state in self.states.items()
                    },
                    batch_size=idx_all.shape[0],
                    device=self.frame.device,
                )
            case return_none:
                assert return_none == TrackletMemoryWriteReturnType.NONE
                return None

    def _update_states(self, obs: TensorDictBase, frame: int) -> tuple[Tensor, Tensor]:
        """Update the states with the evolved observations.

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tuple of (track indices, track IDs) of the updated tracklets.

        """
        num = obs.batch_size[0]
        if num <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.frame.device)
            return empty, empty.clone()

        detect_idxs = obs.get(KEY_INDEX)  # Detection indices assigned to tracklets
        track_idxs = self.index_range[self.index_mask]  # Index in the memory

        # Iterate over the states and update them with the values of the evolved
        # observations.
        # This updates the existing tracklets.
        for id, state in self.states.items():
            if id == KEY_FRAME:
                # Update the frame to the current frame for all observations that
                # are active, otherwise keep the previous frame.
                upd = torch.where(~(detect_idxs >= 0), obs.get(KEY_FRAME), frame)
            else:
                upd = T.cast(Tensor, obs.get(id))
            try:
                state.update(track_idxs, upd)
            except ValueError as e:
                msg = f"Could not update state {id!r}"
                raise ValueError(msg) from e

        return track_idxs, obs.get(KEY_ID)

    def _extend_states(self, new: TensorDictBase, frame: int) -> tuple[Tensor, Tensor]:
        """Exctend the states with new detections, creating new tracklets.

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tuple of (track indices, track IDs) of the newly created tracklets.

        """
        num = new.batch_size[0]
        if num <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.frame.device)
            return empty, empty.clone()

        ids_start = T.cast(int, self.tracklet_count.item()) + 1
        ids_end = ids_start + num
        if ids_end > self.max_id:
            msg = (
                f"Attempted to assign new IDs in range {ids_start} to {ids_end} "
                f"which has a values larger than the maximum allowed ID {self.max_id}!"
            )
            raise RuntimeError(msg)
        ids = torch.arange(
            ids_start, ids_start + num, dtype=torch.long, device=new.device
        )
        size = torch.Size((num,))
        keys = set(T.cast(T.Iterable[str], new.keys()))

        # Find free indices of slots we can occupy
        ext_index = self.index_range[~self.index_mask][:num]
        self.index_mask[ext_index] = True

        # Iterate over the states and extend them with the new detections.
        # This creates new tracklets.
        for id, state in self.states.items():
            if id == KEY_ACTIVE:
                ext = torch.ones(
                    size,
                    dtype=torch.bool,
                    device=self.frame.device,
                )
            elif id in (KEY_FRAME, KEY_START):
                ext = torch.full(
                    size,
                    fill_value=frame,
                    dtype=torch.int,
                    device=self.frame.device,
                )
            elif id == KEY_ID:
                ext = ids
            elif id in keys:
                ext = T.cast(Tensor, new.get(id))
            else:
                msg = (
                    f"State {id!r} is not represented in the new detections! "
                    f"Got new keys: {keys}"
                )
                raise KeyError(msg)
            try:
                state.extend(ext_index, ext)
            except ValueError as e:
                msg = f"Could not extend state {id!r}"
                raise ValueError(msg) from e

        # Update the tracklet count to the last (maximum) ID of the newly made tracklets
        self.tracklet_count.fill_(ids.max())

        return ext_index, ids

    @torch.no_grad()
    def read(self, frame: int) -> tuple[TensorDictBase, TensorDictBase]:
        """Observe the current state of tracklets. Evolves the states of the tracklets
        to the current frame.

        Parameters
        ----------
        frame
            The current frame index, used to automatically reset the memory before
            reading when ``auto_reset`` is enabled.

        Returns
        -------
            A ``TensorDict`` object will all observed states.

        """
        if frame < 0:
            msg = f"Cannot read from memory at negative frame index! Got: {frame!r}"
            raise IndexError(msg)

        if self.frame >= frame:
            if self.auto_reset:
                # Reset because the current frame is larger than the stored frame.
                if check_debug_enabled():
                    pass
                self.reset()
            else:
                msg = f"Frame index {frame} is less than or equal to saved frame {self.frame=}!"
                raise IndexError(msg)

        if check_debug_enabled():
            pass

        # Calculate the time-delta from the amoutn of frames passed and the FPS.
        delta = torch.abs(frame - self.frame) / self.fps

        ctx = TensorDict.from_dict(
            {
                KEY_DELTA: delta,
                KEY_FRAME: frame,
            },
            batch_size=[],
            device=self.frame.device,
        )

        obs_index = self.index_range[self.index_mask]
        obs = TensorDict(
            {id: state.evolve(obs_index, delta) for id, state in self.states.items()},
            batch_size=obs_index.shape[0],
            device=self.frame.device,
        )
        obs[KEY_INDEX] = torch.full(
            obs.batch_size[:1],
            -1,
            dtype=torch.long,
            device=self.frame.device,
            requires_grad=False,
        )

        # Compute which tracklets have been lost for too long
        time_lost = frame - obs.get(KEY_FRAME) - 1
        lost_mask = time_lost > self.max_lost

        # Reset the states of the lost tracklets
        for state in self.states.values():
            state.reset(obs_index[lost_mask])

        # Continue with the active tracklets only
        obs = obs._get_sub_tensordict(~lost_mask)
        lost_index = obs_index[lost_mask]
        self.index_mask[lost_index] = False

        return ctx, obs

    @torch.no_grad()
    def reset(self) -> None:
        """Reset the states of this ``Tracklets`` module."""
        if check_debug_enabled():
            pass
        self.frame.fill_(-1)
        self.write_count.fill_(0)
        self.tracklet_count.fill_(0)
        self.index_mask.fill_(False)

        for state in self.states.values():
            state.reset(self.index_range)  # type: ignore

    @T.override
    def forward(
        self, ctx: TensorDictBase, obs: TensorDictBase, new: TensorDictBase
    ) -> Tensor:
        """Forward pass of the memory module that wraps the :meth:`write` method.

        Users are encouraged to use the :meth:`write` method directly, as it is more
        explicit.
        """
        return self.write(ctx, obs, new)
