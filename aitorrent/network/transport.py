from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import grpc
import grpc.aio
import torch

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

    def __init__(self):
        self._shards: dict[str, TransformerShard] = {}
        self._kv_caches: dict[str, dict] = {}

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
        kv_cache = self._kv_caches.get(session_id) if use_cache else None

        with torch.no_grad():
            if shard.embed is not None and hidden.dtype == torch.long:
                hidden = shard.embed(hidden)
            output, new_cache = shard.forward(hidden, kv_cache=kv_cache)
            if shard.norm is not None:
                output = shard.norm(output)
            if shard.head is not None:
                output = shard.head(output)

        if use_cache and new_cache is not None:
            self._kv_caches[session_id] = new_cache

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
            "active_sessions": len(self._kv_caches),
            "models": list(self._shards.keys()),
        })

    def clear_session(self, session_id: str) -> None:
        self._kv_caches.pop(session_id, None)


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
