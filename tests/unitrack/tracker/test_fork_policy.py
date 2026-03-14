# tests/unitrack/tracker/test_fork_policy.py
from __future__ import annotations

import pytest
from unitrack.tracker import (
    AutoForkOnNewKey,
    ForkPolicy,
    OrderedNoInterleaving,
)


def test_auto_fork_default_allows_any_keys():
    p = AutoForkOnNewKey()
    p.observe(0)
    p.observe(1)
    p.observe(0)


def test_ordered_no_interleaving_rejects_revisit():
    p = OrderedNoInterleaving()
    p.observe(0)
    p.observe(1)
    with pytest.raises(ValueError, match="interleav"):
        p.observe(0)


def test_protocol_runtime_checkable():
    assert isinstance(AutoForkOnNewKey(), ForkPolicy)
    assert isinstance(OrderedNoInterleaving(), ForkPolicy)
