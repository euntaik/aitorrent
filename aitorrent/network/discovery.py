"""Peer discovery — MVP: static config-based. Phase 2: hivemind DHT."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aitorrent.network.peer import PeerInfo
from aitorrent.network.transport import PeerConnection

logger = logging.getLogger(__name__)


class StaticDiscovery:
    """Config-based peer discovery for MVP.

    Peers are specified as a list of addresses in the config.
    Phase 2 replaces this with hivemind Kademlia DHT.
    """

    def __init__(self):
        self._peers: dict[str, PeerInfo] = {}

    def add_peer(self, peer: PeerInfo) -> None:
        self._peers[peer.peer_id] = peer
        logger.info("Discovered peer %s at %s", peer.peer_id, peer.address)

    def remove_peer(self, peer_id: str) -> None:
        self._peers.pop(peer_id, None)

    def find_peers_for_model(self, model_id: str) -> list[PeerInfo]:
        return [p for p in self._peers.values() if model_id in (p.profile and [] or [])]

    def all_peers(self) -> list[PeerInfo]:
        return list(self._peers.values())

    def get_peer(self, peer_id: str) -> PeerInfo | None:
        return self._peers.get(peer_id)

    async def probe_peer(self, address: str) -> PeerInfo | None:
        """Connect to an address and get peer info via health check."""
        conn = PeerConnection(peer_id="unknown", address=address)
        try:
            await conn.connect()
            healthy = await conn.health_check()
            if healthy:
                peer = PeerInfo(peer_id=f"peer_{address}", address=address)
                self.add_peer(peer)
                return peer
        except Exception as e:
            logger.debug("Failed to probe %s: %s", address, e)
        finally:
            await conn.close()
        return None
