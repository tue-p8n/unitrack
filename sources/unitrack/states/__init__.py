"""A state is an entry of ``Tracklets`` that tracks the state of a field."""

from __future__ import annotations

from .base_state import DEFAULT_STATE_SLOTS, State
from .value import Value

__all__ = ["DEFAULT_STATE_SLOTS", "State", "Value"]