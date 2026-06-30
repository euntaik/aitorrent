import pytest

from aitorrent.credit.crypto import (
    PeerIdentity,
    transaction_message,
    verify_with_pubkey,
)
from aitorrent.credit.ledger import CreditLedger, CreditTransaction


@pytest.fixture
def identity_a(tmp_path):
    return PeerIdentity.generate(tmp_path / "a.pem")


@pytest.fixture
def identity_b(tmp_path):
    return PeerIdentity.generate(tmp_path / "b.pem")


def test_identity_generate_and_load(tmp_path):
    ident = PeerIdentity.generate(tmp_path / "key.pem")
    loaded = PeerIdentity.load(tmp_path / "key.pem")
    assert ident.peer_id == loaded.peer_id


def test_identity_load_or_create(tmp_path):
    path = tmp_path / "id.pem"
    a = PeerIdentity.load_or_create(path)
    b = PeerIdentity.load_or_create(path)
    assert a.peer_id == b.peer_id


def test_sign_and_verify(identity_a):
    msg = b"hello world"
    sig = identity_a.sign(msg)
    assert identity_a.verify(msg, sig)


def test_verify_wrong_message(identity_a):
    sig = identity_a.sign(b"hello")
    assert not identity_a.verify(b"world", sig)


def test_verify_with_pubkey(identity_a):
    msg = b"test message"
    sig = identity_a.sign(msg)
    pub = identity_a.public_key_bytes()
    assert verify_with_pubkey(pub, msg, sig)


def test_signed_debit(tmp_path, identity_a):
    ledger = CreditLedger(
        identity_a.peer_id, tmp_path / "credits.db",
        identity=identity_a,
    )
    tx = ledger.debit("peer_b", 50.0, "test")
    assert tx.signature != b""
    assert tx.from_pubkey != b""

    msg = transaction_message(
        tx.tx_id, tx.from_peer, tx.to_peer,
        tx.amount, tx.reason, tx.timestamp, tx.nonce,
    )
    assert verify_with_pubkey(tx.from_pubkey, msg, tx.signature)


def test_signed_credit_accepted(tmp_path, identity_a, identity_b):
    ledger_a = CreditLedger(
        identity_a.peer_id, tmp_path / "a.db", identity=identity_a,
    )
    ledger_b = CreditLedger(
        identity_b.peer_id, tmp_path / "b.db", identity=identity_b,
    )

    tx = ledger_a.debit(identity_b.peer_id, 30.0, "serving")
    ledger_b.credit(tx)
    assert ledger_b.balance == 1030.0


def test_tampered_signature_rejected(tmp_path, identity_a, identity_b):
    ledger_a = CreditLedger(
        identity_a.peer_id, tmp_path / "a.db", identity=identity_a,
    )
    ledger_b = CreditLedger(
        identity_b.peer_id, tmp_path / "b.db", identity=identity_b,
    )

    tx = ledger_a.debit(identity_b.peer_id, 30.0, "serving")
    tx.amount = 9999.0  # tamper
    with pytest.raises(ValueError, match="Invalid transaction signature"):
        ledger_b.credit(tx)
