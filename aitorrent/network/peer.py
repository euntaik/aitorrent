from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aitorrent.config import AITorrentConfig
from aitorrent.model.loader import ShardLoader, TransformerShard
from aitorrent.model.manifest import ModelManifest
from aitorrent.model.profiler import HardwareProfile, HardwareProfiler
from aitorrent.network.transport import (
    GrpcServer,
    InferenceServicer,
    PeerConnection,
)

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    peer_id: str
    address: str
    profile: HardwareProfile | None = None
    start_layer: int = 0
    end_layer: int = 0
    includes_embed: bool = False
    includes_head: bool = False


class PeerNode:
    """Represents this node in the AITorrent network."""

    def __init__(self, config: AITorrentConfig):
        self.config = config
        self.peer_id = str(uuid.uuid4())[:12]
        self.profile: HardwareProfile | None = None
        self.servicer = InferenceServicer()
        self.server = GrpcServer(self.servicer, config.network.grpc_port)
        self.connections: dict[str, PeerConnection] = {}
        self._running = False

    async def start(self) -> None:
        profiler = HardwareProfiler()
        self.profile = profiler.profile()
        logger.info(
            "Node %s starting — GPU: %s, VRAM: %dMB, RAM: %dMB, %.2f TFLOPS",
            self.peer_id,
            self.profile.gpu_name or "none",
            self.profile.gpu_vram_free_mb,
            self.profile.ram_free_mb,
            self.profile.compute_tflops,
        )
        await self.server.start()
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for conn in self.connections.values():
            await conn.close()
        await self.server.stop()

    def load_shard(self, model_id: str, shard: TransformerShard) -> None:
        self.servicer.register_shard(model_id, shard)

    async def connect_to_peer(self, peer_info: PeerInfo) -> PeerConnection:
        conn = PeerConnection(
            peer_id=peer_info.peer_id,
            address=peer_info.address,
        )
        await conn.connect()
        self.connections[peer_info.peer_id] = conn
        return conn

    def get_connection(self, peer_id: str) -> PeerConnection | None:
        return self.connections.get(peer_id)

    @property
    def address(self) -> str:
        return f"{self.config.network.listen_host}:{self.config.network.grpc_port}"
