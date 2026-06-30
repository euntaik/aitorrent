from __future__ import annotations

import struct

import msgpack
import torch

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}


def serialize_tensor(tensor: torch.Tensor) -> tuple[bytes, str, list[int]]:
    t = tensor.contiguous().cpu()
    dtype_str = DTYPE_REVERSE.get(t.dtype, "float32")
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float16)
        dtype_str = "float16"
    shape = list(t.shape)
    raw = t.numpy().tobytes()
    return raw, dtype_str, shape


def deserialize_tensor(
    data: bytes, dtype_str: str, shape: list[int], device: str = "cpu"
) -> torch.Tensor:
    dtype = DTYPE_MAP.get(dtype_str, torch.float32)
    np_dtype = "float16" if dtype == torch.float16 else "float32"
    import numpy as np
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    tensor = torch.from_numpy(arr.copy()).to(dtype)
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def pack_message(msg: dict) -> bytes:
    return msgpack.packb(msg, use_bin_type=True)


def unpack_message(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False)
