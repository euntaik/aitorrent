"""Shared test fixtures — tiny models and peer helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from aitorrent.model.loader import TransformerShard
from aitorrent.model.manifest import ModelManifest


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** -0.5

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return self.o(attn @ v)


class TinyTransformerLayer(nn.Module):
    """Minimal transformer layer for testing."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = TinyAttention(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, **kwargs):
        h = hidden_states + self.self_attn(self.norm1(hidden_states))
        h = h + self.mlp(self.norm2(h))
        return (h,)  # match HF tuple output style


HIDDEN_SIZE = 64
VOCAB_SIZE = 256
NUM_LAYERS = 4


def build_tiny_model(
    num_layers: int = NUM_LAYERS,
    hidden_size: int = HIDDEN_SIZE,
    vocab_size: int = VOCAB_SIZE,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Embedding, nn.ModuleList, nn.LayerNorm, nn.Linear]:
    """Build components of a tiny transformer for testing."""
    torch.manual_seed(42)
    embed = nn.Embedding(vocab_size, hidden_size).to(dtype)
    layers = nn.ModuleList([TinyTransformerLayer(hidden_size).to(dtype) for _ in range(num_layers)])
    norm = nn.LayerNorm(hidden_size).to(dtype)
    head = nn.Linear(hidden_size, vocab_size, bias=False).to(dtype)
    return embed, layers, norm, head


def make_shard(
    start_layer: int,
    end_layer: int,
    embed: nn.Embedding | None,
    layers: nn.ModuleList,
    norm: nn.LayerNorm | None,
    head: nn.Linear | None,
    dtype: torch.dtype = torch.float32,
) -> TransformerShard:
    shard_layers = nn.ModuleList([layers[i] for i in range(start_layer, end_layer)])
    return TransformerShard(
        model_id="tiny-test",
        start_layer=start_layer,
        end_layer=end_layer,
        layers=shard_layers,
        embed=embed,
        head=head,
        norm=norm,
        device="cpu",
        dtype=dtype,
    )


@pytest.fixture
def tiny_manifest():
    return ModelManifest(
        model_id="tiny-test",
        architecture="test",
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=1,
        num_kv_heads=1,
        vocab_size=VOCAB_SIZE,
        max_seq_length=128,
        dtype="float32",
    )


@pytest.fixture
def tiny_model_parts():
    return build_tiny_model()
