from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import grpc
import grpc.aio
import torch

from aitorrent.credit.ledger import CreditLedger
from aitorrent.credit.pricing import CreditPricer
from aitorrent.inference.failover import FailoverManager
from aitorrent.inference.kv_cache import KVCacheManager
from aitorrent.inference.session import InferenceSession
from aitorrent.model.loader import TransformerShard
from aitorrent.model.manifest import ModelManifest
from aitorrent.network.peer import PeerInfo, PeerNode
from aitorrent.network.transport import PeerConnection

logger = logging.getLogger(__name__)

MAX_STEP_RESTARTS = 3


class CacheInvalidatedError(RuntimeError):
    """A stage's KV cache is out of sync; the step must be replayed with
    the full sequence under a fresh session."""


@dataclass
class PipelineStage:
    peer_info: PeerInfo
    connection: PeerConnection | None  # None if local
    local_shard: TransformerShard | None  # Set if this stage runs locally


class InferencePipeline:
    """Orchestrates distributed inference across peers in a pipeline."""

    def __init__(
        self,
        node: PeerNode,
        manifest: ModelManifest,
        stages: list[PipelineStage],
        ledger: CreditLedger,
        pricer: CreditPricer,
        failover: FailoverManager | None = None,
    ):
        self._node = node
        self._manifest = manifest
        self._stages = stages
        self._ledger = ledger
        self._pricer = pricer
        self._failover = failover
        self._local_caches = KVCacheManager()

    async def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> list[int]:
        generated_ids = (
            input_ids.tolist() if input_ids.dim() == 1 else input_ids[0].tolist()
        )
        async for token_id in self.generate_stream(
            input_ids, max_new_tokens, temperature, top_p, use_cache
        ):
            generated_ids.append(token_id)
        return generated_ids

    async def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_cache: bool = True,
    ):
        session = InferenceSession(model_id=self._manifest.model_id)
        device = self._get_device()
        ids_tensor = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        ids_tensor = ids_tensor.to(device)

        # Prefill sends the whole prompt; with cache each decode step then
        # sends only the newly generated token.
        step_input = ids_tensor
        past_length = 0

        try:
            for step in range(max_new_tokens):
                if not use_cache:
                    step_input = ids_tensor
                    past_length = 0

                hidden, session, step_input, past_length = await self._forward_step(
                    session, ids_tensor, step_input, past_length, use_cache
                )

                logits = hidden[:, -1, :]
                next_token = self._sample(logits, temperature, top_p)
                token_id = next_token.item()
                yield token_id

                next_col = next_token.unsqueeze(0).to(device)  # shape [1, 1]
                past_length += step_input.shape[1]
                ids_tensor = torch.cat([ids_tensor, next_col], dim=1)
                step_input = next_col
        finally:
            self._clear_local_caches(session)

    async def _forward_step(
        self,
        session: InferenceSession,
        ids_tensor: torch.Tensor,
        step_input: torch.Tensor,
        past_length: int,
        use_cache: bool,
    ) -> tuple[torch.Tensor, InferenceSession, torch.Tensor, int]:
        """Run one forward pass, replaying with full context if any stage
        lost its KV cache (eviction, peer restart, or failover)."""
        for attempt in range(MAX_STEP_RESTARTS):
            try:
                hidden = await self._full_forward(
                    session, step_input, past_length, use_cache
                )
                return hidden, session, step_input, past_length
            except CacheInvalidatedError as e:
                if attempt == MAX_STEP_RESTARTS - 1:
                    raise
                logger.warning(
                    "KV cache invalidated (%s); replaying full context "
                    "under a fresh session", e,
                )
                self._clear_local_caches(session)
                session = InferenceSession(model_id=self._manifest.model_id)
                step_input = ids_tensor
                past_length = 0
        raise RuntimeError("unreachable")

    async def _full_forward(
        self,
        session: InferenceSession,
        step_input: torch.Tensor,
        past_length: int,
        use_cache: bool,
    ) -> torch.Tensor:
        hidden = None

        for idx, stage in enumerate(self._stages):
            if stage.local_shard is not None:
                hidden = await self._local_forward(
                    idx, stage.local_shard, session,
                    step_input if hidden is None else hidden,
                    past_length, use_cache,
                )
            elif stage.connection is not None:
                if hidden is None:
                    raise RuntimeError("First stage must be local (embedding required)")
                hidden = await self._remote_forward(
                    stage, session, hidden, past_length, use_cache
                )
            else:
                raise RuntimeError(
                    f"Stage for peer {stage.peer_info.peer_id} has no shard or connection"
                )

        return hidden

    async def _local_forward(
        self,
        stage_idx: int,
        shard: TransformerShard,
        session: InferenceSession,
        input_tensor: torch.Tensor,
        past_length: int,
        use_cache: bool,
    ) -> torch.Tensor:
        kv_cache = None
        cache_key = f"{session.session_id}:{stage_idx}"
        if use_cache:
            kv_cache = self._local_caches.get(cache_key)
            if shard.cache_length(kv_cache) != past_length:
                self._local_caches.evict(cache_key)
                raise CacheInvalidatedError(
                    f"local stage {stage_idx} cache out of sync"
                )
            if kv_cache is None:
                from transformers.cache_utils import DynamicCache
                kv_cache = DynamicCache()

        with torch.no_grad():
            if shard.embed is not None and input_tensor.dtype == torch.long:
                hidden = shard.embed(input_tensor)
            else:
                hidden = input_tensor

            hidden = shard.forward(hidden, past_length=past_length, kv_cache=kv_cache)

            if shard.norm is not None:
                hidden = shard.norm(hidden)
            if shard.head is not None:
                hidden = shard.head(hidden)

        if use_cache:
            self._local_caches.put(cache_key, kv_cache)

        return hidden

    async def _remote_forward(
        self,
        stage: PipelineStage,
        session: InferenceSession,
        hidden: torch.Tensor,
        past_length: int,
        use_cache: bool,
    ) -> torch.Tensor:
        try:
            output, tokens = await stage.connection.forward_pass(
                session_id=session.session_id,
                model_id=self._manifest.model_id,
                hidden_states=hidden,
                use_cache=use_cache,
                past_length=past_length,
            )
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                # Server-side cache out of sync; replay, no failover needed
                raise CacheInvalidatedError(
                    f"peer {stage.peer_info.peer_id}: {e.details()}"
                ) from e
            await self._handle_stage_failure(stage, e)
            raise CacheInvalidatedError(
                f"peer {stage.peer_info.peer_id} replaced after failure"
            ) from e
        except Exception as e:
            await self._handle_stage_failure(stage, e)
            raise CacheInvalidatedError(
                f"peer {stage.peer_info.peer_id} replaced after failure"
            ) from e

        num_layers = stage.peer_info.end_layer - stage.peer_info.start_layer
        credits = self._pricer.price_tokens(tokens, num_layers=num_layers)
        self._ledger.debit(
            stage.connection.peer_id, credits, f"inference:{session.session_id}"
        )
        session.record_tokens(tokens, credits)

        return output

    async def _handle_stage_failure(
        self, stage: PipelineStage, error: Exception
    ) -> None:
        """Try to swap the failed stage's peer for a backup. Raises if no
        replacement is available; otherwise mutates the stage in place and
        returns (caller replays the step, rebuilding caches on the new peer)."""
        logger.warning(
            "Remote forward to %s failed: %s", stage.peer_info.peer_id, error
        )
        if self._failover is None:
            raise RuntimeError(
                f"Remote forward to {stage.peer_info.peer_id} failed: {error}"
            ) from error

        self._failover.report_failure(stage.peer_info.peer_id)
        result = await self._failover.find_replacement(
            stage.peer_info, self._manifest.model_id
        )
        if not result.success:
            raise RuntimeError(
                f"Peer {stage.peer_info.peer_id} failed and no replacement "
                f"is available: {error}"
            ) from error

        old_peer = stage.peer_info.peer_id
        stage.connection = result.replacement
        stage.peer_info = result.replacement_peer
        logger.info("Stage failover: %s -> %s", old_peer, result.replacement_peer.peer_id)

    def _clear_local_caches(self, session: InferenceSession) -> None:
        for idx in range(len(self._stages)):
            self._local_caches.evict(f"{session.session_id}:{idx}")

    def _sample(
        self, logits: torch.Tensor, temperature: float, top_p: float
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        idx = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, idx).squeeze(-1)

    def _get_device(self) -> str:
        for stage in self._stages:
            if stage.local_shard is not None:
                return stage.local_shard.device
        return "cpu"
