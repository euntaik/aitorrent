from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from aitorrent.network.transport import PeerConnection

logger = logging.getLogger(__name__)


@dataclass
class PeerHealth:
    peer_id: str
    healthy: bool
    latency_ms: float
    last_check: float
    consecutive_failures: int = 0


class HealthMonitor:
    def __init__(self, check_interval: int = 10, timeout_sec: int = 30):
        self._interval = check_interval
        self._timeout = timeout_sec
        self._health: dict[str, PeerHealth] = {}
        self._running = False

    async def start(self, connections: dict[str, PeerConnection]) -> None:
        self._running = True
        while self._running:
            for peer_id, conn in list(connections.items()):
                await self._check_peer(peer_id, conn)
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False

    async def _check_peer(self, peer_id: str, conn: PeerConnection) -> None:
        start = time.perf_counter()
        healthy = await conn.health_check()
        latency = (time.perf_counter() - start) * 1000

        prev = self._health.get(peer_id)
        failures = 0 if healthy else (prev.consecutive_failures + 1 if prev else 1)

        self._health[peer_id] = PeerHealth(
            peer_id=peer_id,
            healthy=healthy,
            latency_ms=latency,
            last_check=time.time(),
            consecutive_failures=failures,
        )

        if not healthy:
            logger.warning("Peer %s health check failed (%d consecutive)", peer_id, failures)

    def is_healthy(self, peer_id: str) -> bool:
        h = self._health.get(peer_id)
        return h is not None and h.healthy

    def get_latency(self, peer_id: str) -> float:
        h = self._health.get(peer_id)
        return h.latency_ms if h else float("inf")
