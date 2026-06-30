"""Tests for failover manager and pipeline retry logic."""

from __future__ import annotations

import asyncio
import pytest
import torch

from aitorrent.inference.failover import FailoverManager, FailoverResult, MAX_RETRIES
from aitorrent.network.peer import PeerInfo


@pytest.fixture
def failover():
    return FailoverManager()


def _peer(peer_id: str, start: int, end: int, address: str = "localhost:9999") -> PeerInfo:
    return PeerInfo(
        peer_id=peer_id, address=address,
        start_layer=start, end_layer=end,
    )


class TestFailoverManager:
    def test_report_failure_blacklists_after_max(self, failover):
        for _ in range(MAX_RETRIES - 1):
            failover.report_failure("peer_a")
            assert not failover.is_blacklisted("peer_a")
        failover.report_failure("peer_a")
        assert failover.is_blacklisted("peer_a")

    def test_register_and_find_backups(self, failover):
        backup = _peer("backup_1", 0, 10)
        failover.register_backups("model_x", [backup])
        assert failover._backup_peers["model_x"] == [backup]

    @pytest.mark.asyncio
    async def test_find_replacement_no_backups(self, failover):
        failed = _peer("peer_a", 0, 10)
        result = await failover.find_replacement(failed, "no_model")
        assert not result.success
        assert "No replacement" in result.error

    @pytest.mark.asyncio
    async def test_find_replacement_skips_blacklisted(self, failover):
        backup = _peer("backup_1", 0, 10)
        failover.register_backups("model_x", [backup])
        for _ in range(MAX_RETRIES):
            failover.report_failure("backup_1")

        failed = _peer("peer_a", 0, 10)
        result = await failover.find_replacement(failed, "model_x")
        assert not result.success

    def test_covers_layers_exact(self, failover):
        candidate = _peer("c", 0, 10)
        needed = _peer("n", 0, 10)
        assert failover._covers_layers(candidate, needed)

    def test_covers_layers_superset(self, failover):
        candidate = _peer("c", 0, 20)
        needed = _peer("n", 5, 15)
        assert failover._covers_layers(candidate, needed)

    def test_covers_layers_insufficient(self, failover):
        candidate = _peer("c", 5, 10)
        needed = _peer("n", 0, 10)
        assert not failover._covers_layers(candidate, needed)
