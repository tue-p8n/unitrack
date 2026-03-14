"""Per-stream tracker state: current snapshot and the next-ID counter."""

from __future__ import annotations

from unitrack.data import Tracklets

__all__ = ["TrackletMemory"]


class TrackletMemory:
    """
    Hold one stream's :class:`~unitrack.data.Tracklets` and its next-ID counter.

    Carries no knowledge of the :class:`~unitrack.tracker.Tracker`, its policies, or its
    schema. The snapshot supplied to :meth:`__init__` serves as the
    schema template for :meth:`reset`.

    Attributes
    ----------
    snapshot : ~unitrack.data.Tracklets
        Live snapshot for this stream.
    next_id : int
        First ID available for new tracklets.

    """

    def __init__(self, empty: Tracklets) -> None:
        """
        Initialise from a zero-row :class:`~unitrack.data.Tracklets` template.

        Parameters
        ----------
        empty
            Zero-row schema template used both as the initial snapshot
            and as the :meth:`reset` template.

        """
        self._empty = empty
        self.snapshot = empty
        self.next_id = 1

    def reset(self) -> None:
        """Restore the initial empty snapshot and reset ``next_id`` to ``1``."""
        self.snapshot = self._empty
        self.next_id = 1

    def load(self, snapshot: Tracklets, next_id: int) -> None:
        """Replace ``snapshot`` and ``next_id`` with external values."""
        self.snapshot = snapshot
        self.next_id = int(next_id)

    def fork(self) -> TrackletMemory:
        """
        Return an independent copy of this memory.

        The fork carries the current ``snapshot`` and ``next_id``
        so callers can branch from the live state (multi-stream / vmap).
        The new memory has its own snapshot reference but shares the
        empty schema template for :meth:`reset`.

        Returns
        -------
        TrackletMemory
            Fresh instance pointing at the same schema template.

        """
        cloned = TrackletMemory(self._empty)
        cloned.snapshot = self.snapshot
        cloned.next_id = self.next_id
        return cloned
