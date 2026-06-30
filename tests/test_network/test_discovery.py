"""Tests for DHT peer discovery."""

from __future__ import annotations

import json
import time

import pytest

from aitorrent.network.discovery import DHTDiscovery, StaticDiscovery, DHT_PEER_TTL
from aitorrent.network.peer import PeerInfo


def _peer(pid: str, addr: str = "localhost:9877") -> PeerInfo:
    return PeerInfo(peer_id=pid, address=addr)


class TestStaticDiscovery:
    def test_add_and_get(self):
        d = StaticDiscovery()
        p = _peer("p1")
        d.add_peer(p)
        assert d.get_peer("p1") is p
        assert len(d.all_peers()) == 1

    def test_remove(self):
        d = StaticDiscovery()
        d.add_peer(_peer("p1"))
        d.remove_peer("p1")
        assert d.get_peer("p1") is None


class TestDHTDiscovery:
    def _make_dht(self, pid: str = "local") -> DHTDiscovery:
        return DHTDiscovery(local_peer=_peer(pid, "localhost:9877"))

    def test_handle_announce_adds_peer(self):
        dht = self._make_dht()
        msg = json.dumps({
            "type": "announce",
            "peer_id": "remote_1",
            "address": "192.168.1.5:9877",
            "models": ["llama-70b"],
            "pubkey": "",
        }).encode()
        dht.handle_announce(msg, ("192.168.1.5", 9876))
        assert dht.get_peer("remote_1") is not None
        assert dht.get_peer("remote_1").address == "192.168.1.5:9877"

    def test_ignores_self_announce(self):
        dht = self._make_dht("local")
        msg = json.dumps({
            "type": "announce",
            "peer_id": "local",
            "address": "localhost:9877",
            "models": [],
            "pubkey": "",
        }).encode()
        dht.handle_announce(msg, ("127.0.0.1", 9876))
        assert len(dht.all_peers()) == 0

    def test_find_peers_for_model(self):
        dht = self._make_dht()
        for i, models in enumerate([["llama-70b"], ["mistral-7b"], ["llama-70b", "mistral-7b"]]):
            msg = json.dumps({
                "type": "announce",
                "peer_id": f"peer_{i}",
                "address": f"10.0.0.{i}:9877",
                "models": models,
                "pubkey": "",
            }).encode()
            dht.handle_announce(msg, ("10.0.0.1", 9876))

        llama_peers = dht.find_peers_for_model("llama-70b")
        assert len(llama_peers) == 2

    def test_expire_peers(self):
        dht = self._make_dht()
        msg = json.dumps({
            "type": "announce",
            "peer_id": "old_peer",
            "address": "10.0.0.1:9877",
            "models": [],
            "pubkey": "",
        }).encode()
        dht.handle_announce(msg, ("10.0.0.1", 9876))
        assert len(dht.all_peers()) == 1

        dht._peers["old_peer"].last_seen = time.time() - DHT_PEER_TTL - 1
        assert len(dht.all_peers()) == 0

    def test_register_model(self):
        dht = self._make_dht()
        dht.register_model("llama-70b")
        dht.register_model("llama-70b")  # duplicate
        assert dht._models == ["llama-70b"]

    def test_ignores_invalid_json(self):
        dht = self._make_dht()
        dht.handle_announce(b"not json", ("1.2.3.4", 9876))
        assert len(dht.all_peers()) == 0

    def test_ignores_wrong_type(self):
        dht = self._make_dht()
        msg = json.dumps({"type": "query", "peer_id": "x"}).encode()
        dht.handle_announce(msg, ("1.2.3.4", 9876))
        assert len(dht.all_peers()) == 0

    def test_pubkey_hex_roundtrip(self):
        dht = self._make_dht()
        key = b"\x01\x02\x03\x04" * 8
        msg = json.dumps({
            "type": "announce",
            "peer_id": "keyed_peer",
            "address": "10.0.0.1:9877",
            "models": [],
            "pubkey": key.hex(),
        }).encode()
        dht.handle_announce(msg, ("10.0.0.1", 9876))
        peer = dht.get_peer("keyed_peer")
        assert peer.pubkey == key

    @pytest.mark.asyncio
    async def test_start_stop(self):
        dht = self._make_dht()
        await dht.start()
        assert dht._running
        await dht.stop()
        assert not dht._running
