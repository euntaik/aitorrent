"""Peer failure detection and automatic failover during inference."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from aitorrent.network.peer import PeerInfo
from aitorrent.network.transport import PeerConnection

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
HEALTH_TIMEOUT_SEC = 3.0


@dataclass
class FailoverResult:
    success: bool
    replacement: PeerConnection | None = None
    replacement_peer: PeerInfo | None = None
    error: str | None = None


class FailoverManager:
    """Manages backup peers and handles failover during inference."""

    def __init__(self):
        self._backup_peers: dict[str, list[PeerInfo]] = {}
        self._blacklist: set[str] = set()
        self._failure_counts: dict[str, int] = {}

    def register_backups(self, model_id: str, peers: list[PeerInfo]) -> None:
        self._backup_peers[model_id] = list(peers)

    def report_failure(self, peer_id: str) -> None:
        count = self._failure_counts.get(peer_id, 0) + 1
        self._failure_counts[peer_id] = count
        if count >= MAX_RETRIES:
            self._blacklist.add(peer_id)
            logger.warning("Peer %s blacklisted after %d failures", peer_id, count)

    def is_blacklisted(self, peer_id: str) -> bool:
        return peer_id in self._blacklist

    async def find_replacement(
        self,
        failed_peer: PeerInfo,
        model_id: str,
    ) -> FailoverResult:
        logger.warning(
            "Peer %s failed (layers %d-%d), searching for replacement...",
            failed_peer.peer_id, failed_peer.start_layer, failed_peer.end_layer,
        )

        backups = self._backup_peers.get(model_id, [])
        for backup in backups:
            if backup.peer_id == failed_peer.peer_id:
                continue
            if self.is_blacklisted(backup.peer_id):
                continue
            if not self._covers_layers(backup, failed_peer):
                continue

            try:
                conn = PeerConnection(peer_id=backup.peer_id, address=backup.address)
                await conn.connect()
                healthy = await conn.health_check()
                if healthy:
                    logger.info(
                        "Failover: %s -> %s for layers %d-%d",
                        failed_peer.peer_id, backup.peer_id,
                        failed_peer.start_layer, failed_peer.end_layer,
                    )
                    return FailoverResult(
                        success=True, replacement=conn, replacement_peer=backup,
                    )
                await conn.close()
            except Exception as e:
                logger.debug("Backup %s unreachable: %s", backup.peer_id, e)

        self.report_failure(failed_peer.peer_id)
        return FailoverResult(success=False, error="No replacement peer available")

    def _covers_layers(self, candidate: PeerInfo, needed: PeerInfo) -> bool:
        return (
            candidate.start_layer <= needed.start_layer
            and candidate.end_layer >= needed.end_layer
        )


async def with_retry(
    coro_factory,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = 0.5,
):
    """Retry an async operation with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            wait = backoff_base * (2 ** attempt)
            logger.warning("Attempt %d failed: %s. Retrying in %.1fs", attempt + 1, e, wait)
            await asyncio.sleep(wait)
    raise last_error
