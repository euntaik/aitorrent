from __future__ import annotations

import logging
from dataclasses import dataclass

from aitorrent.network.peer import PeerInfo
from aitorrent.network.transport import PeerConnection

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


@dataclass
class FailoverResult:
    success: bool
    replacement: PeerConnection | None = None
    error: str | None = None


async def attempt_failover(
    failed_peer: PeerInfo,
    backup_peers: list[PeerInfo],
) -> FailoverResult:
    logger.warning("Peer %s failed, attempting failover...", failed_peer.peer_id)

    for backup in backup_peers:
        if backup.peer_id == failed_peer.peer_id:
            continue
        if not (backup.start_layer <= failed_peer.start_layer
                and backup.end_layer >= failed_peer.end_layer):
            continue

        try:
            conn = PeerConnection(peer_id=backup.peer_id, address=backup.address)
            await conn.connect()
            healthy = await conn.health_check()
            if healthy:
                logger.info("Failover to peer %s successful", backup.peer_id)
                return FailoverResult(success=True, replacement=conn)
            await conn.close()
        except Exception as e:
            logger.debug("Backup peer %s also failed: %s", backup.peer_id, e)

    return FailoverResult(success=False, error="No suitable backup peer found")
