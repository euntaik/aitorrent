from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import grpc
import grpc.aio
import torch

from aitorrent.inference.kv_cache import KVCacheManager
from aitorrent.model.loader import TransformerShard
from aitorrent.network.serialization import (
    deserialize_tensor,
    pack_message,
    serialize_tensor,
    unpack_message,
)

logger = logging.getLogger(__name__)

METHOD_FORWARD = "/aitorrent.InferenceService/ForwardPass"
METHOD_HEALTH = "/aitorrent.InferenceService/HealthCheck"

_identity = lambda x: x


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class InferenceServicer:
    """Handles incoming ForwardPass requests from other peers."""

    def __init__(self, kv_cache_ttl_sec: int = 300):
        self._shards: dict[str, TransformerShard] = {}
        self._cache_manager = KVCacheManager(ttl_sec=kv_cache_ttl_sec)

    def register_shard(self, model_id: str, shard: TransformerShard) -> None:
        self._shards[model_id] = shard
        logger.info(
            "Registered shard for %s: layers %d-%d",
            model_id, shard.start_layer, shard.end_layer,
        )

    async def handle_forward(self, request_bytes: bytes, context) -> bytes:
        req = unpack_message(request_bytes)
        model_id = req["model_id"]
        shard = self._shards.get(model_id)
        if shard is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"No shard for {model_id}")

        hidden = deserialize_tensor(req["input_tensor"], req["dtype"], req["shape"], shard.device)
        session_id = req["session_id"]
        use_cache = req.get("use_cache", False)
        past_length = req.get("past_length", 0)

        kv_cache = None
        if use_cache:
            cache_key = f"{session_id}:{model_id}"
            kv_cache = self._cache_manager.get(cache_key)
            server_past = shard.cache_length(kv_cache)
            if server_past != past_length:
                # Cache out of sync (evicted, restarted, or replayed request).
                # The client recovers by re-sending the full sequence under a
                # fresh session.
                self._cache_manager.evict(cache_key)
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"KV cache mismatch: server has {server_past} tokens, "
                    f"request expects {past_length}",
                )
            if kv_cache is None:
                from transformers.cache_utils import DynamicCache
                kv_cache = DynamicCache()

        with torch.no_grad():
            if shard.embed is not None and hidden.dtype == torch.long:
                hidden = shard.embed(hidden)
            output = shard.forward(hidden, past_length=past_length, kv_cache=kv_cache)
            if shard.norm is not None:
                output = shard.norm(output)
            if shard.head is not None:
                output = shard.head(output)

        if use_cache:
            self._cache_manager.put(f"{session_id}:{model_id}", kv_cache)

        tokens = req["shape"][1] if len(req["shape"]) > 1 else 1
        out_data, out_dtype, out_shape = serialize_tensor(output)
        return pack_message({
            "output_tensor": out_data,
            "dtype": out_dtype,
            "shape": out_shape,
            "tokens_processed": tokens,
        })

    async def handle_health(self, request_bytes: bytes, context) -> bytes:
        return pack_message({
            "healthy": True,
            "active_sessions": self._cache_manager.active_sessions,
            "models": list(self._shards.keys()),
        })

    def clear_session(self, session_id: str) -> None:
        for model_id in self._shards:
            self._cache_manager.evict(f"{session_id}:{model_id}")


class _Handler(grpc.GenericRpcHandler):
    def __init__(self, servicer: InferenceServicer):
        self._servicer = servicer
        self._methods = {
            METHOD_FORWARD: grpc.unary_unary_rpc_method_handler(
                servicer.handle_forward,
                request_deserializer=_identity,
                response_serializer=_identity,
            ),
            METHOD_HEALTH: grpc.unary_unary_rpc_method_handler(
                servicer.handle_health,
                request_deserializer=_identity,
                response_serializer=_identity,
            ),
        }

    def service(self, handler_call_details):
        return self._methods.get(handler_call_details.method)


class GrpcServer:
    def __init__(self, servicer: InferenceServicer, port: int = 9877):
        self._servicer = servicer
        self._port = port
        self._server: grpc.aio.Server | None = None

    async def start(self) -> None:
        self._server = grpc.aio.server()
        self._server.add_generic_rpc_handlers([_Handler(self._servicer)])
        self._server.add_insecure_port(f"[::]:{self._port}")
        await self._server.start()
        logger.info("gRPC server started on port %d", self._port)

    async def stop(self) -> None:
        if self._server:
            await self._server.stop(grace=2)
            logger.info("gRPC server stopped")

    async def wait(self) -> None:
        if self._server:
            await self._server.wait_for_termination()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class PeerConnection:
    peer_id: str
    address: str
    channel: grpc.aio.Channel | None = None

    async def connect(self) -> None:
        opts = [
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ]
        self.channel = grpc.aio.insecure_channel(self.address, options=opts)
        logger.info("Connected to peer %s at %s", self.peer_id, self.address)

    async def close(self) -> None:
        if self.channel:
            await self.channel.close()
            self.channel = None

    async def forward_pass(
        self,
        session_id: str,
        model_id: str,
        hidden_states: torch.Tensor,
        use_cache: bool = True,
        past_length: int = 0,
    ) -> tuple[torch.Tensor, int]:
        if not self.channel:
            raise RuntimeError("Not connected")

        input_data, dtype_str, shape = serialize_tensor(hidden_states)
        request = pack_message({
            "session_id": session_id,
            "model_id": model_id,
            "input_tensor": input_data,
            "dtype": dtype_str,
            "shape": shape,
            "use_cache": use_cache,
            "past_length": past_length,
        })

        call = self.channel.unary_unary(
            METHOD_FORWARD,
            request_serializer=_identity,
            response_deserializer=_identity,
        )
        response_bytes = await call(request)

        resp = unpack_message(response_bytes)
        output = deserialize_tensor(
            resp["output_tensor"],
            resp["dtype"],
            resp["shape"],
            device=hidden_states.device.type,
        )
        return output, resp["tokens_processed"]

    async def health_check(self) -> bool:
        if not self.channel:
            return False
        try:
            call = self.channel.unary_unary(
                METHOD_HEALTH,
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            response_bytes = await call(pack_message({}), timeout=5)
            resp = unpack_message(response_bytes)
            return resp.get("healthy", False)
        except Exception:
            return False
