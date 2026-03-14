# tests/unitrack/data/test_tensor_spec.py
from __future__ import annotations

import torch
from unitrack.data import TensorSpec


def test_construct_from_shape_and_dtype():
    s = TensorSpec(shape=(3,), dtype=torch.float32)
    assert s.shape == (3,)
    assert s.dtype is torch.float32


def test_zero_dim_shape_allowed():
    s = TensorSpec(shape=(), dtype=torch.int64)
    assert s.shape == ()
    assert s.dtype is torch.int64


def test_make_zero_buffer_matches_spec():
    s = TensorSpec(shape=(2, 3), dtype=torch.float32)
    out = s.empty(slots=4)
    assert out.shape == (4, 2, 3)
    assert out.dtype is torch.float32
    assert torch.all(out == 0)


def test_equality():
    a = TensorSpec(shape=(3,), dtype=torch.float32)
    b = TensorSpec(shape=(3,), dtype=torch.float32)
    c = TensorSpec(shape=(4,), dtype=torch.float32)
    assert a == b
    assert a != c
