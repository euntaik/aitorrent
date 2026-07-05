"""
Integration test: 3-peer distributed inference with signed credits.

Splits a 4-layer model across 3 peers (A:layer0, B:layers1-2, C:layer3),
verifies correctness against baseline, token generation, and signed credit settlement.
"""

from __future__ import annotations

import pytest
import torch

from aitorrent.config import AITorrentConfig
from aitorrent.credit.crypto import PeerIdentity
from aitorrent.credit.ledger import CreditLedger
from aitorrent.credit.pricing import CreditPricer
from aitorrent.inference.pipeline import InferencePipeline, PipelineStage
from aitorrent.model.manifest import ModelManifest
from aitorrent.network.peer import PeerInfo
from aitorrent.network.transport import GrpcServer, InferenceServicer, PeerConnection
from tests.conftest import build_tiny_model, make_shard, HIDDEN_SIZE, NUM_LAYERS, VOCAB_SIZE

MANIFEST = ModelManifest(
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


def _baseline_logits(input_ids: torch.Tensor) -> torch.Tensor:
    embed, layers, norm, head = build_tiny_model()
    with torch.no_grad():
        h = embed(input_ids)
        for layer in layers:
            h = layer(h)[0]
        h = norm(h)
        return head(h)


@pytest.fixture
def model_parts():
    return build_tiny_model()


@pytest.mark.asyncio
async def test_three_peer_forward_matches_baseline(model_parts):
    """3-peer chain produces same logits as single-node."""
    embed, layers, norm, head = model_parts

    shard_a = make_shard(0, 1, embed, layers, norm=None, head=None)
    shard_b = make_shard(1, 3, embed=None, layers=layers, norm=None, head=None)
    shard_c = make_shard(3, 4, embed=None, layers=layers, norm=norm, head=head)

    servicer_b = InferenceServicer()
    servicer_b.register_shard("tiny-test", shard_b)
    server_b = GrpcServer(servicer_b, port=19880)

    servicer_c = InferenceServicer()
    servicer_c.register_shard("tiny-test", shard_c)
    server_c = GrpcServer(servicer_c, port=19881)

    await server_b.start()
    await server_c.start()

    try:
        conn_b = PeerConnection(peer_id="peer_b", address="localhost:19880")
        conn_c = PeerConnection(peer_id="peer_c", address="localhost:19881")
        await conn_b.connect()
        await conn_c.connect()

        input_ids = torch.tensor([[1, 50, 100, 200]], dtype=torch.long)

        with torch.no_grad():
            h = shard_a.embed(input_ids)
            h = shard_a.forward(h)

            h, _ = await conn_b.forward_pass(
                session_id="test", model_id="tiny-test",
                hidden_states=h, use_cache=False,
            )

            h, _ = await conn_c.forward_pass(
                session_id="test", model_id="tiny-test",
                hidden_states=h, use_cache=False,
            )

        baseline = _baseline_logits(input_ids)
        torch.testing.assert_close(h.float(), baseline.float(), rtol=1e-3, atol=1e-4)

    finally:
        await conn_b.close()
        await conn_c.close()
        await server_b.stop()
        await server_c.stop()


@pytest.mark.asyncio
async def test_three_peer_pipeline_generates_tokens(model_parts, tmp_path):
    """Pipeline with 3 stages generates correct number of tokens."""
    embed, layers, norm, head = model_parts

    shard_a = make_shard(0, 1, embed, layers, norm=None, head=None)
    shard_b = make_shard(1, 3, embed=None, layers=layers, norm=None, head=None)
    shard_c = make_shard(3, 4, embed=None, layers=layers, norm=norm, head=head)

    servicer_b = InferenceServicer()
    servicer_b.register_shard("tiny-test", shard_b)
    server_b = GrpcServer(servicer_b, port=19882)

    servicer_c = InferenceServicer()
    servicer_c.register_shard("tiny-test", shard_c)
    server_c = GrpcServer(servicer_c, port=19883)

    await server_b.start()
    await server_c.start()

    try:
        conn_b = PeerConnection(peer_id="peer_b", address="localhost:19882")
        conn_c = PeerConnection(peer_id="peer_c", address="localhost:19883")
        await conn_b.connect()
        await conn_c.connect()

        ledger = CreditLedger("peer_a", tmp_path / "credits.db")
        pricer = CreditPricer()

        stages = [
            PipelineStage(
                peer_info=PeerInfo(
                    peer_id="peer_a", address="localhost:0",
                    start_layer=0, end_layer=1,
                ),
                connection=None, local_shard=shard_a,
            ),
            PipelineStage(
                peer_info=PeerInfo(
                    peer_id="peer_b", address="localhost:19882",
                    start_layer=1, end_layer=3,
                ),
                connection=conn_b, local_shard=None,
            ),
            PipelineStage(
                peer_info=PeerInfo(
                    peer_id="peer_c", address="localhost:19883",
                    start_layer=3, end_layer=4,
                ),
                connection=conn_c, local_shard=None,
            ),
        ]

        pipeline = InferencePipeline(
            node=None, manifest=MANIFEST,
            stages=stages, ledger=ledger, pricer=pricer,
        )

        input_ids = torch.tensor([1, 50, 100], dtype=torch.long)
        # Custom tiny layers don't implement KV caching
        generated = await pipeline.generate(
            input_ids, max_new_tokens=5, temperature=0.0, use_cache=False,
        )

        assert len(generated) == 8
        assert all(0 <= t < VOCAB_SIZE for t in generated)

    finally:
        await conn_b.close()
        await conn_c.close()
        await server_b.stop()
        await server_c.stop()


@pytest.mark.asyncio
async def test_signed_credit_settlement_three_peers(model_parts, tmp_path):
    """Credits are debited with valid Ed25519 signatures across 3 peers."""
    embed, layers, norm, head = model_parts

    shard_a = make_shard(0, 1, embed, layers, norm=None, head=None)
    shard_b = make_shard(1, 3, embed=None, layers=layers, norm=None, head=None)
    shard_c = make_shard(3, 4, embed=None, layers=layers, norm=norm, head=head)

    servicer_b = InferenceServicer()
    servicer_b.register_shard("tiny-test", shard_b)
    server_b = GrpcServer(servicer_b, port=19884)

    servicer_c = InferenceServicer()
    servicer_c.register_shard("tiny-test", shard_c)
    server_c = GrpcServer(servicer_c, port=19885)

    await server_b.start()
    await server_c.start()

    try:
        conn_b = PeerConnection(peer_id="peer_b", address="localhost:19884")
        conn_c = PeerConnection(peer_id="peer_c", address="localhost:19885")
        await conn_b.connect()
        await conn_c.connect()

        identity_a = PeerIdentity.generate(tmp_path / "a.pem")
        identity_b = PeerIdentity.generate(tmp_path / "b.pem")
        identity_c = PeerIdentity.generate(tmp_path / "c.pem")

        ledger_a = CreditLedger(
            "peer_a", tmp_path / "a.db", identity=identity_a,
        )
        ledger_b = CreditLedger(
            "peer_b", tmp_path / "b.db", identity=identity_b,
        )
        ledger_c = CreditLedger(
            "peer_c", tmp_path / "c.db", identity=identity_c,
        )

        pricer = CreditPricer()

        stages = [
            PipelineStage(
                peer_info=PeerInfo(
                    peer_id="peer_a", address="localhost:0",
                    start_layer=0, end_layer=1,
                ),
                connection=None, local_shard=shard_a,
            ),
            PipelineStage(
                peer_info=PeerInfo(
                    peer_id="peer_b", address="localhost:19884",
                    start_layer=1, end_layer=3,
                ),
                connection=conn_b, local_shard=None,
            ),
            PipelineStage(
                peer_info=PeerInfo(
                    peer_id="peer_c", address="localhost:19885",
                    start_layer=3, end_layer=4,
                ),
                connection=conn_c, local_shard=None,
            ),
        ]

        pipeline = InferencePipeline(
            node=None, manifest=MANIFEST,
            stages=stages, ledger=ledger_a, pricer=pricer,
        )

        input_ids = torch.tensor([1, 50, 100], dtype=torch.long)
        generated = await pipeline.generate(
            input_ids, max_new_tokens=3, temperature=0.0, use_cache=False,
        )
        assert len(generated) == 6

        # Verify credits were debited with signatures
        assert ledger_a.balance < 1000.0
        txs = ledger_a.recent_transactions()
        assert len(txs) > 0
        for tx in txs:
            assert tx.signature != b""
            assert tx.from_pubkey != b""

        # Verify peer B can accept the signed credits (sort by nonce for ordering)
        b_txs = sorted(
            [t for t in txs if t.to_peer == "peer_b"], key=lambda t: t.nonce,
        )
        for tx in b_txs:
            ledger_b.credit(tx)
        assert ledger_b.balance > 1000.0

        # Verify peer C can accept the signed credits
        c_txs = sorted(
            [t for t in txs if t.to_peer == "peer_c"], key=lambda t: t.nonce,
        )
        for tx in c_txs:
            ledger_c.credit(tx)
        assert ledger_c.balance > 1000.0

    finally:
        await conn_b.close()
        await conn_c.close()
        await server_b.stop()
        await server_c.stop()
