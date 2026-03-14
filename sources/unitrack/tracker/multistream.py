"""Per-key tracker wrapper and fork-policy primitives."""

from __future__ import annotations

import dataclasses
import typing

from unitrack.data import Detections, FrameContext, Tracklets

from .memory import TrackletMemory
from .tracker import StepResult, Tracker


@typing.runtime_checkable
class ForkPolicy(typing.Protocol):
    """How a :class:`~unitrack.tracker.MultiStream` reacts to a stream key."""

    def observe(self, key: int | str) -> None:
        """Update policy state for the given stream ``key``."""
        ...

    def forget(self, key: int | str) -> None:
        """
        Drop any state held for ``key``.

        Called by :meth:`MultiStream.end_stream`. Implementations that
        hold no per-key state may make this a no-op.
        """
        ...


@dataclasses.dataclass(slots=True)
class AutoForkOnNewKey:
    """
    Lazily fork a fresh memory on first sight of a key.

    Any sequence of keys, including interleaved access, is accepted.
    """

    def observe(self, key: int | str) -> None:
        """Accept any ``key`` without restriction."""
        del key

    def forget(self, key: int | str) -> None:
        """No-op: this policy holds no per-key state."""
        del key


@dataclasses.dataclass(slots=True)
class OrderedNoInterleaving:
    """
    Strict one-stream-at-a-time policy.

    Only one stream key may be live at a time. Revisiting an earlier
    key after a different one took over raises :exc:`ValueError` so
    callers notice they should be using :class:`AutoForkOnNewKey` or
    :class:`BatchTracker` instead. A key can be revisited cleanly after
    :meth:`MultiStream.end_stream`; :meth:`forget` drops the key from
    the seen set.
    """

    _seen: set[int | str] = dataclasses.field(default_factory=set)
    _current: int | str | None = None

    def observe(self, key: int | str) -> None:
        """
        Accept ``key`` only if it is current or never been seen.

        Raises
        ------
        ValueError
            If ``key`` was previously seen and a different key has
            since taken over.

        """
        if self._current == key:
            return
        if key in self._seen:
            msg = f"key {key!r} re-encountered after interleaving"
            raise ValueError(msg)
        self._seen.add(key)
        self._current = key

    def forget(self, key: int | str) -> None:
        """Drop ``key`` so a fresh stream can re-open after ``end_stream``."""
        self._seen.discard(key)
        if self._current == key:
            self._current = None


class MultiStream:
    """
    Multi-stream wrapper around a :class:`~unitrack.tracker.Tracker`.

    Holds one :class:`TrackletMemory` per stream key. The :class:`ForkPolicy`
    decides whether a previously-seen key is allowed to be revisited.

    Parameters
    ----------
    tracker
        Underlying :class:`~unitrack.tracker.Tracker`.
    fork_policy
        Policy for handling stream keys; defaults to
        :class:`AutoForkOnNewKey`.

    """

    def __init__(
        self,
        tracker: Tracker,
        fork_policy: ForkPolicy | None = None,
    ) -> None:
        """See class docstring for parameter semantics."""
        self.tracker = tracker
        if fork_policy is None:
            fork_policy = AutoForkOnNewKey()
        self.fork_policy = fork_policy
        self._template = tracker.empty_snapshot()
        self._streams: dict[int | str, TrackletMemory] = {}

    def _memory_for(self, key: int | str) -> TrackletMemory:
        if key not in self._streams:
            self._streams[key] = TrackletMemory(self._template)
        return self._streams[key]

    def begin_stream(self, key: int | str) -> None:
        """Eagerly create a memory slot for ``key``."""
        self._memory_for(key)

    def end_stream(self, key: int | str) -> None:
        """Remove the memory slot for ``key`` and notify the fork policy."""
        self._streams.pop(key, None)
        self.fork_policy.forget(key)

    def reset(self, key: int | str | None = None) -> None:
        """
        Reset one stream or all streams.

        Parameters
        ----------
        key
            Stream key to reset, or ``None`` to reset every registered
            stream.

        Raises
        ------
        KeyError
            If ``key`` is supplied but has no memory slot.

        """
        if key is None:
            for mem in self._streams.values():
                mem.reset()
            return
        if key not in self._streams:
            msg = f"MultiStream.reset: no stream registered for key {key!r}"
            raise KeyError(msg)
        self._streams[key].reset()

    def has_stream(self, key: int | str) -> bool:
        """Return ``True`` iff a memory slot already exists for ``key``."""
        return key in self._streams

    def snapshot_of(self, key: int | str) -> Tracklets:
        """
        Return the current snapshot for ``key`` without allocating state.

        Parameters
        ----------
        key
            Stream key.

        Returns
        -------
        ~unitrack.data.Tracklets
            Live snapshot for ``key``.

        Raises
        ------
        KeyError
            If ``key`` has no memory slot. A read-only accessor must
            not silently allocate state; use :meth:`begin_stream` (or
            :meth:`step`, which auto-allocates) to create the slot.

        """
        if key not in self._streams:
            msg = (
                f"MultiStream.snapshot_of: no stream registered for key {key!r}. "
                "Call begin_stream(key) or step(key, ...) first."
            )
            raise KeyError(msg)
        return self._streams[key].snapshot

    def step(
        self,
        stream_key: int | str,
        detections: Detections,
        ctx: FrameContext,
    ) -> StepResult:
        """
        Advance ``stream_key`` by one frame.

        Parameters
        ----------
        stream_key
            Stream identifier.
        detections
            Frame detections.
        ctx
            Frame context.

        Returns
        -------
        StepResult
            The tracker's :class:`StepResult` for this frame.

        """
        self.fork_policy.observe(stream_key)
        mem = self._memory_for(stream_key)
        res = self.tracker.step(mem.snapshot, detections, ctx, mem.next_id)
        mem.load(res.snapshot, res.next_id)
        return res


__all__ = [
    "AutoForkOnNewKey",
    "ForkPolicy",
    "MultiStream",
    "OrderedNoInterleaving",
]
