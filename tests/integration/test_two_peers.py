"""
Integration test: 2-peer distributed inference over gRPC.

Builds a tiny 4-layer transformer, splits it in half,
runs each half on a separate gRPC server, and verifies that
the distributed result matches the single-node baseline.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from aitorrent.config import AITorrentConfig
from aitorrent.credit.ledger import CreditLedger
from aitorrent.credit.pricing import CreditPricer
from aitorrent.inference.pipeline import InferencePipeline, PipelineStage
from aitorrent.inference.session import InferenceSession
from aitorrent.network.peer import PeerInfo
from aitorrent.network.transport import GrpcServer, InferenceServicer, PeerConnection
from tests.conftest import build_tiny_model, make_shard, HIDDEN_SIZE, NUM_LAYERS, VOCAB_SIZE


def _single_node_forward(input_ids: torch.Tensor):
    """Baseline: run all layers on one machine."""
    embed, layers, norm, head = build_tiny_model()
    with torch.no_grad():
        h = embed(input_ids)
        for layer in layers:
            h = layer(h)[0]
        h = norm(h)
        logits = head(h)
    return logits


@pytest.mark.asyncio
async def test_two_peer_forward_matches_baseline(tmp_path):
    """Distributed 2-peer inference produces the same logits as single-node."""

    # --- build the same tiny model for both paths ---
    embed, layers, norm, head = build_tiny_model()

    # Shard A: layers 0-1 + embedding
    shard_a = make_shard(0, 2, embed, layers, norm=None, head=None)
    # Shard B: layers 2-3 + norm + head
    shard_b = make_shard(2, 4, embed=None, layers=layers, norm=norm, head=head)

    # --- start peer B as a gRPC server ---
    servicer_b = InferenceServicer()
    servicer_b.register_shard("tiny-test", shard_b)
    server_b = GrpcServer(servicer_b, port=19877)
    await server_b.start()

    try:
        # --- peer A connects to peer B ---
        conn_b = PeerConnection(peer_id="peer_b", address="localhost:19877")
        await conn_b.connect()

        # health check
        assert await conn_b.health_check() is True

        # --- run distributed forward pass ---
        input_ids = torch.tensor([[1, 50, 100, 200]], dtype=torch.long)

        with torch.no_grad():
            # Stage 1: local on peer A
            h = shard_a.embed(input_ids)
            h, _ = shard_a.forward(h)

            # Stage 2: remote on peer B
            h_remote, tokens = await conn_b.forward_pass(
                session_id="test-session",
                model_id="tiny-test",
                hidden_states=h,
                use_cache=False,
            )

        # --- baseline ---
        baseline_logits = _single_node_forward(input_ids)

        # --- compare ---
        torch.testing.assert_close(
            h_remote.float(), baseline_logits.float(), rtol=1e-3, atol=1e-4
        )

    finally:
        await conn_b.close()
        await server_b.stop()


@pytest.mark.asyncio
async def test_two_peer_pipeline_generates_tokens(tmp_path):
    """Full pipeline: generate tokens via the InferencePipeline orchestrator."""

    embed, layers, norm, head = build_tiny_model()
    shard_a = make_shard(0, 2, embed, layers, norm=None, head=None)
    shard_b = make_shard(2, 4, embed=None, layers=layers, norm=norm, head=head)

    servicer_b = InferenceServicer()
    servicer_b.register_shard("tiny-test", shard_b)
    server_b = GrpcServer(servicer_b, port=19878)
    await server_b.start()

    try:
        conn_b = PeerConnection(peer_id="peer_b", address="localhost:19878")
        await conn_b.connect()

        from aitorrent.model.manifest import ModelManifest
        manifest = ModelManifest(
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

        ledger = CreditLedger("peer_a", tmp_path / "credits.db")
        pricer = CreditPricer()

        stages = [
            PipelineStage(
                peer_info=PeerInfo(peer_id="peer_a", address="localhost:0",
                                   start_layer=0, end_layer=2),
                connection=None,
                local_shard=shard_a,
            ),
            PipelineStage(
                peer_info=PeerInfo(peer_id="peer_b", address="localhost:19878",
                                   start_layer=2, end_layer=4),
                connection=conn_b,
                local_shard=None,
            ),
        ]

        pipeline = InferencePipeline(
            node=None,  # not used in this test
            manifest=manifest,
            stages=stages,
            ledger=ledger,
            pricer=pricer,
        )

        input_ids = torch.tensor([1, 50, 100], dtype=torch.long)
        generated = await pipeline.generate(
            input_ids, max_new_tokens=5, temperature=0.0
        )

        # input (3) + generated (5) = 8 tokens
        assert len(generated) == 8
        assert all(0 <= t < VOCAB_SIZE for t in generated)

    finally:
        await conn_b.close()
        await server_b.stop()


@pytest.mark.asyncio
async def test_credits_are_settled(tmp_path):
    """After inference, the requester's credit balance should decrease."""

    embed, layers, norm, head = build_tiny_model()
    shard_b = make_shard(0, 4, embed, layers, norm=norm, head=head)

    servicer_b = InferenceServicer()
    servicer_b.register_shard("tiny-test", shard_b)
    server_b = GrpcServer(servicer_b, port=19879)
    await server_b.start()

    try:
        conn_b = PeerConnection(peer_id="peer_b", address="localhost:19879")
        await conn_b.connect()

        ledger = CreditLedger("peer_a", tmp_path / "credits.db", bootstrap_credits=1000)
        initial_balance = ledger.balance

        # single remote forward
        h = torch.randn(1, 3, HIDDEN_SIZE, dtype=torch.float32)
        _, tokens = await conn_b.forward_pass(
            session_id="credit-test",
            model_id="tiny-test",
            hidden_states=h,
            use_cache=False,
        )

        pricer = CreditPricer()
        cost = pricer.price_tokens(tokens, num_layers=4)
        ledger.debit("peer_b", cost, "test-inference")

        assert ledger.balance < initial_balance
        assert ledger.balance == initial_balance - cost

    finally:
        await conn_b.close()
        await server_b.stop()
