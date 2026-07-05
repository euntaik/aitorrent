"""
Integration test: distributed KV cache across 2 peers.

Uses a random-init tiny Llama (real HF layers, rotary embeddings,
DynamicCache) split across a local stage and a gRPC-served remote stage.
"""

from __future__ import annotations

import pytest
import torch

from aitorrent.credit.ledger import CreditLedger
from aitorrent.credit.pricing import CreditPricer
from aitorrent.inference.pipeline import InferencePipeline, PipelineStage
from aitorrent.model.manifest import ModelManifest
from aitorrent.network.peer import PeerInfo
from aitorrent.network.transport import GrpcServer, InferenceServicer, PeerConnection
from tests.conftest import build_tiny_llama, make_llama_shard, NUM_LAYERS, VOCAB_SIZE

MANIFEST = ModelManifest(
    model_id="tiny-llama",
    architecture="llama",
    num_layers=NUM_LAYERS,
    hidden_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    vocab_size=VOCAB_SIZE,
    max_seq_length=128,
    dtype="float32",
)

PROMPT = [1, 42, 100, 200, 7]
GEN_TOKENS = 8


def _baseline_greedy(model, prompt: list[int], n_new: int) -> list[int]:
    """Single-node greedy generation, full re-feed (ground truth)."""
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(n_new):
            logits = model(torch.tensor([ids])).logits
            ids.append(logits[0, -1].argmax().item())
    return ids


async def _build_two_peer_pipeline(model, tmp_path, port: int):
    shard_a = make_llama_shard(model, 0, 2, includes_embed=True)
    shard_b = make_llama_shard(model, 2, 4, includes_head=True)

    servicer = InferenceServicer()
    servicer.register_shard("tiny-llama", shard_b)
    server = GrpcServer(servicer, port=port)
    await server.start()

    conn = PeerConnection(peer_id="peer_b", address=f"localhost:{port}")
    await conn.connect()

    stages = [
        PipelineStage(
            peer_info=PeerInfo(
                peer_id="peer_a", address="localhost:0",
                start_layer=0, end_layer=2,
            ),
            connection=None, local_shard=shard_a,
        ),
        PipelineStage(
            peer_info=PeerInfo(
                peer_id="peer_b", address=f"localhost:{port}",
                start_layer=2, end_layer=4,
            ),
            connection=conn, local_shard=None,
        ),
    ]
    pipeline = InferencePipeline(
        node=None, manifest=MANIFEST, stages=stages,
        ledger=CreditLedger("peer_a", tmp_path / "credits.db"),
        pricer=CreditPricer(),
    )
    return pipeline, servicer, server, conn


@pytest.mark.asyncio
async def test_cached_generation_matches_baseline(tmp_path):
    """Greedy generation with KV cache == single-node full re-feed."""
    model = build_tiny_llama()
    pipeline, servicer, server, conn = await _build_two_peer_pipeline(
        model, tmp_path, port=19890,
    )
    try:
        generated = await pipeline.generate(
            torch.tensor(PROMPT), max_new_tokens=GEN_TOKENS,
            temperature=0.0, use_cache=True,
        )
        baseline = _baseline_greedy(model, PROMPT, GEN_TOKENS)
        assert generated == baseline

        # The remote peer actually built a session cache
        assert servicer._cache_manager.active_sessions >= 1
    finally:
        await conn.close()
        await server.stop()


@pytest.mark.asyncio
async def test_cache_on_off_equivalence(tmp_path):
    """use_cache=True and use_cache=False produce identical greedy output."""
    model = build_tiny_llama()
    pipeline, servicer, server, conn = await _build_two_peer_pipeline(
        model, tmp_path, port=19891,
    )
    try:
        with_cache = await pipeline.generate(
            torch.tensor(PROMPT), max_new_tokens=GEN_TOKENS,
            temperature=0.0, use_cache=True,
        )
        without_cache = await pipeline.generate(
            torch.tensor(PROMPT), max_new_tokens=GEN_TOKENS,
            temperature=0.0, use_cache=False,
        )
        assert with_cache == without_cache
    finally:
        await conn.close()
        await server.stop()


@pytest.mark.asyncio
async def test_decode_sends_single_token(tmp_path):
    """After prefill, each decode request carries exactly one token."""
    model = build_tiny_llama()
    pipeline, servicer, server, conn = await _build_two_peer_pipeline(
        model, tmp_path, port=19892,
    )
    seen_shapes = []
    original = conn.forward_pass

    async def spy(session_id, model_id, hidden_states, use_cache=True, past_length=0):
        seen_shapes.append(tuple(hidden_states.shape))
        return await original(session_id, model_id, hidden_states, use_cache, past_length)

    conn.forward_pass = spy
    try:
        await pipeline.generate(
            torch.tensor(PROMPT), max_new_tokens=4,
            temperature=0.0, use_cache=True,
        )
        # Prefill carries the prompt, every decode step carries 1 token
        assert seen_shapes[0][1] == len(PROMPT)
        assert all(s[1] == 1 for s in seen_shapes[1:])
        assert len(seen_shapes) == 1 + 3
    finally:
        await conn.close()
        await server.stop()


@pytest.mark.asyncio
async def test_recovers_from_mid_generation_cache_eviction(tmp_path):
    """If the remote peer loses its session cache mid-generation, the
    pipeline replays the full context and output stays correct."""
    model = build_tiny_llama()
    pipeline, servicer, server, conn = await _build_two_peer_pipeline(
        model, tmp_path, port=19893,
    )
    try:
        generated = list(PROMPT)
        step = 0
        async for token_id in pipeline.generate_stream(
            torch.tensor(PROMPT), max_new_tokens=GEN_TOKENS,
            temperature=0.0, use_cache=True,
        ):
            generated.append(token_id)
            step += 1
            if step == 3:
                # Simulate the remote peer losing all state
                for sid in list(servicer._cache_manager._entries):
                    servicer._cache_manager.evict(sid)

        baseline = _baseline_greedy(model, PROMPT, GEN_TOKENS)
        assert generated == baseline
    finally:
        await conn.close()
        await server.stop()
