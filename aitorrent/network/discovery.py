"""Peer discovery — StaticDiscovery for config-based, DHTDiscovery for network-based."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from aitorrent.network.peer import PeerInfo
from aitorrent.network.transport import PeerConnection

logger = logging.getLogger(__name__)

DHT_BROADCAST_PORT = 9876
DHT_ANNOUNCE_INTERVAL = 30
DHT_PEER_TTL = 90


class StaticDiscovery:
    """Config-based peer discovery for simple setups."""

    def __init__(self):
        self._peers: dict[str, PeerInfo] = {}

    def add_peer(self, peer: PeerInfo) -> None:
        self._peers[peer.peer_id] = peer
        logger.info("Discovered peer %s at %s", peer.peer_id, peer.address)

    def remove_peer(self, peer_id: str) -> None:
        self._peers.pop(peer_id, None)

    def find_peers_for_model(self, model_id: str) -> list[PeerInfo]:
        return list(self._peers.values())

    def all_peers(self) -> list[PeerInfo]:
        return list(self._peers.values())

    def get_peer(self, peer_id: str) -> PeerInfo | None:
        return self._peers.get(peer_id)

    async def probe_peer(self, address: str) -> PeerInfo | None:
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


@dataclass
class _PeerRecord:
    peer_info: PeerInfo
    models: list[str]
    last_seen: float


class DHTDiscovery:
    """UDP broadcast-based peer discovery for LAN environments.

    Each peer periodically broadcasts an announce packet on the LAN.
    Other peers listen and maintain a table of known peers with TTL expiry.
    For WAN, a rendezvous server address can be provided (future).
    """

    def __init__(
        self,
        local_peer: PeerInfo,
        broadcast_port: int = DHT_BROADCAST_PORT,
        announce_interval: float = DHT_ANNOUNCE_INTERVAL,
        peer_ttl: float = DHT_PEER_TTL,
        rendezvous_address: str | None = None,
    ):
        self._local = local_peer
        self._port = broadcast_port
        self._interval = announce_interval
        self._ttl = peer_ttl
        self._rendezvous = rendezvous_address
        self._peers: dict[str, _PeerRecord] = {}
        self._models: list[str] = []
        self._transport: asyncio.DatagramTransport | None = None
        self._announce_task: asyncio.Task | None = None
        self._running = False

    def register_model(self, model_id: str) -> None:
        if model_id not in self._models:
            self._models.append(model_id)

    async def start(self) -> None:
        self._running = True
        loop = asyncio.get_running_loop()

        protocol = _BroadcastProtocol(self)
        try:
            self._transport, _ = await loop.create_datagram_endpoint(
                lambda: protocol,
                local_addr=("0.0.0.0", self._port),
                allow_broadcast=True,
            )
        except OSError as e:
            logger.warning("Could not bind broadcast port %d: %s", self._port, e)
            self._transport, _ = await loop.create_datagram_endpoint(
                lambda: protocol,
                local_addr=("0.0.0.0", 0),
                allow_broadcast=True,
            )

        self._announce_task = asyncio.create_task(self._announce_loop())
        logger.info("DHT discovery started on port %d", self._port)

    async def stop(self) -> None:
        self._running = False
        if self._announce_task:
            self._announce_task.cancel()
            try:
                await self._announce_task
            except asyncio.CancelledError:
                pass
        if self._transport:
            self._transport.close()

    async def _announce_loop(self) -> None:
        while self._running:
            self._announce()
            self._expire_peers()
            await asyncio.sleep(self._interval)

    def _announce(self) -> None:
        if not self._transport:
            return
        msg = json.dumps({
            "type": "announce",
            "peer_id": self._local.peer_id,
            "address": self._local.address,
            "models": self._models,
            "pubkey": self._local.pubkey.hex() if self._local.pubkey else "",
        }).encode()
        try:
            self._transport.sendto(msg, ("<broadcast>", self._port))
        except OSError as e:
            logger.debug("Broadcast failed: %s", e)

    def handle_announce(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            msg = json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if msg.get("type") != "announce":
            return
        peer_id = msg.get("peer_id", "")
        if peer_id == self._local.peer_id:
            return

        address = msg.get("address", f"{addr[0]}:{self._port + 1}")
        pubkey_hex = msg.get("pubkey", "")
        pubkey = bytes.fromhex(pubkey_hex) if pubkey_hex else b""
        models = msg.get("models", [])

        peer = PeerInfo(peer_id=peer_id, address=address, pubkey=pubkey)
        record = self._peers.get(peer_id)
        if record is None:
            logger.info("Discovered new peer %s at %s", peer_id[:12], address)
        self._peers[peer_id] = _PeerRecord(
            peer_info=peer, models=models, last_seen=time.time(),
        )

    def _expire_peers(self) -> None:
        now = time.time()
        expired = [
            pid for pid, rec in self._peers.items()
            if now - rec.last_seen > self._ttl
        ]
        for pid in expired:
            logger.info("Peer %s expired (no announce for %.0fs)", pid[:12], self._ttl)
            del self._peers[pid]

    def find_peers_for_model(self, model_id: str) -> list[PeerInfo]:
        self._expire_peers()
        return [
            rec.peer_info for rec in self._peers.values()
            if model_id in rec.models
        ]

    def all_peers(self) -> list[PeerInfo]:
        self._expire_peers()
        return [rec.peer_info for rec in self._peers.values()]

    def get_peer(self, peer_id: str) -> PeerInfo | None:
        rec = self._peers.get(peer_id)
        return rec.peer_info if rec else None

    async def probe_peer(self, address: str) -> PeerInfo | None:
        conn = PeerConnection(peer_id="unknown", address=address)
        try:
            await conn.connect()
            healthy = await conn.health_check()
            if healthy:
                peer = PeerInfo(peer_id=f"peer_{address}", address=address)
                self._peers[peer.peer_id] = _PeerRecord(
                    peer_info=peer, models=[], last_seen=time.time(),
                )
                return peer
        except Exception as e:
            logger.debug("Failed to probe %s: %s", address, e)
        finally:
            await conn.close()
        return None


class _BroadcastProtocol(asyncio.DatagramProtocol):
    def __init__(self, discovery: DHTDiscovery):
        self._discovery = discovery

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._discovery.handle_announce(data, addr)

    def error_received(self, exc: Exception) -> None:
        logger.debug("Broadcast protocol error: %s", exc)
