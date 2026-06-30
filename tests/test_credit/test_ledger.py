import tempfile
from pathlib import Path

import pytest

from aitorrent.credit.ledger import CreditLedger, CreditTransaction


@pytest.fixture
def ledger(tmp_path):
    return CreditLedger("peer_a", tmp_path / "test.db", bootstrap_credits=1000)


def test_initial_balance(ledger):
    assert ledger.balance == 1000.0


def test_debit(ledger):
    tx = ledger.debit("peer_b", 50.0, "test")
    assert ledger.balance == 950.0
    assert tx.from_peer == "peer_a"
    assert tx.to_peer == "peer_b"
    assert tx.amount == 50.0


def test_credit(ledger):
    tx = CreditTransaction(
        tx_id="test_tx",
        from_peer="peer_b",
        to_peer="peer_a",
        amount=25.0,
        reason="serving",
        timestamp=0,
        nonce=1,
    )
    ledger.credit(tx)
    assert ledger.balance == 1025.0


def test_stale_nonce_rejected(ledger):
    tx1 = CreditTransaction(
        tx_id="tx1", from_peer="peer_b", to_peer="peer_a",
        amount=10.0, reason="", timestamp=0, nonce=1,
    )
    ledger.credit(tx1)

    tx2 = CreditTransaction(
        tx_id="tx2", from_peer="peer_b", to_peer="peer_a",
        amount=10.0, reason="", timestamp=0, nonce=1,
    )
    with pytest.raises(ValueError, match="Stale nonce"):
        ledger.credit(tx2)


def test_balance_with(ledger):
    ledger.debit("peer_b", 100.0)
    assert ledger.balance_with("peer_b") == -100.0


def test_recent_transactions(ledger):
    ledger.debit("peer_b", 10.0)
    ledger.debit("peer_c", 20.0)
    txns = ledger.recent_transactions(10)
    assert len(txns) == 2
