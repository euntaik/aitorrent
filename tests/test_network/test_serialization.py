import pytest
import torch

from aitorrent.network.serialization import (
    deserialize_tensor,
    serialize_tensor,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_roundtrip(dtype):
    original = torch.randn(2, 4, 128, dtype=dtype)
    data, dtype_str, shape = serialize_tensor(original)
    restored = deserialize_tensor(data, dtype_str, shape)
    torch.testing.assert_close(original.float(), restored.float(), rtol=1e-3, atol=1e-5)


def test_shape_preserved():
    original = torch.randn(1, 1, 4096, dtype=torch.float16)
    data, dtype_str, shape = serialize_tensor(original)
    restored = deserialize_tensor(data, dtype_str, shape)
    assert list(restored.shape) == [1, 1, 4096]


def test_bfloat16_converts_to_float16():
    original = torch.randn(2, 4, dtype=torch.bfloat16)
    data, dtype_str, shape = serialize_tensor(original)
    assert dtype_str == "float16"
    restored = deserialize_tensor(data, dtype_str, shape)
    assert restored.dtype == torch.float16
