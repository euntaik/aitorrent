from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import torch

from aitorrent.credit.ledger import CreditLedger
from aitorrent.credit.pricing import CreditPricer
from aitorrent.inference.failover import FailoverManager, MAX_RETRIES
from aitorrent.inference.session import InferenceSession
from aitorrent.model.loader import TransformerShard
from aitorrent.model.manifest import ModelManifest
from aitorrent.network.peer import PeerInfo, PeerNode
from aitorrent.network.transport import PeerConnection

logger = logging.getLogger(__name__)


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

    async def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[int]:
        session = InferenceSession(model_id=self._manifest.model_id)
        generated_ids = input_ids.tolist() if input_ids.dim() == 1 else input_ids[0].tolist()

        device = self._get_device()
        ids_tensor = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        ids_tensor = ids_tensor.to(device)

        for step in range(max_new_tokens):
            hidden = await self._full_forward(session, ids_tensor, step == 0)

            logits = hidden[:, -1, :]
            next_token = self._sample(logits, temperature, top_p)
            token_id = next_token.item()
            generated_ids.append(token_id)

            # No distributed KV cache yet: re-feed the full sequence each step
            # so new tokens attend to the whole context.
            ids_tensor = torch.cat(
                [ids_tensor, next_token.unsqueeze(0).to(device)], dim=1
            )

        return generated_ids

    async def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        session = InferenceSession(model_id=self._manifest.model_id)
        device = self._get_device()
        ids_tensor = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        ids_tensor = ids_tensor.to(device)

        for step in range(max_new_tokens):
            hidden = await self._full_forward(session, ids_tensor, step == 0)
            logits = hidden[:, -1, :]
            next_token = self._sample(logits, temperature, top_p)
            token_id = next_token.item()
            yield token_id
            ids_tensor = torch.cat(
                [ids_tensor, next_token.unsqueeze(0).to(device)], dim=1
            )

    async def _full_forward(
        self,
        session: InferenceSession,
        ids_tensor: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        hidden = None

        for stage in self._stages:
            if stage.local_shard is not None:
                hidden = await self._local_forward(
                    stage.local_shard, session, ids_tensor if hidden is None else hidden
                )
            elif stage.connection is not None:
                if hidden is None:
                    raise RuntimeError("First stage must be local (embedding required)")
                hidden = await self._remote_forward(
                    stage, stage.connection, session, hidden
                )
            else:
                raise RuntimeError(f"Stage for peer {stage.peer_info.peer_id} has no shard or connection")

        return hidden

    async def _local_forward(
        self,
        shard: TransformerShard,
        session: InferenceSession,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            if shard.embed is not None and input_tensor.dtype == torch.long:
                hidden = shard.embed(input_tensor)
            else:
                hidden = input_tensor

            hidden, _ = shard.forward(hidden, kv_cache=None)

            if shard.norm is not None:
                hidden = shard.norm(hidden)
            if shard.head is not None:
                hidden = shard.head(hidden)

        return hidden

    async def _remote_forward(
        self,
        stage: PipelineStage,
        connection: PeerConnection,
        session: InferenceSession,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        last_error = None
        current_conn = connection
        current_peer = stage.peer_info

        for attempt in range(MAX_RETRIES):
            try:
                output, tokens = await current_conn.forward_pass(
                    session_id=session.session_id,
                    model_id=self._manifest.model_id,
                    hidden_states=hidden,
                    use_cache=True,
                )

                num_layers = current_peer.end_layer - current_peer.start_layer
                credits = self._pricer.price_tokens(tokens, num_layers=num_layers)
                self._ledger.debit(current_conn.peer_id, credits, f"inference:{session.session_id}")
                session.record_tokens(tokens, credits)
                return output

            except Exception as e:
                last_error = e
                logger.warning(
                    "Remote forward to %s failed (attempt %d): %s",
                    current_peer.peer_id, attempt + 1, e,
                )

                if self._failover is None:
                    break

                self._failover.report_failure(current_peer.peer_id)
                result = await self._failover.find_replacement(
                    current_peer, self._manifest.model_id,
                )
                if not result.success:
                    break
                current_conn = result.replacement
                current_peer = result.replacement_peer
                logger.info("Retrying with replacement peer %s", current_peer.peer_id)

        raise RuntimeError(
            f"Remote forward failed after {MAX_RETRIES} attempts: {last_error}"
        )

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
