from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aitorrent.credit.crypto import (
    PeerIdentity,
    transaction_message,
    verify_with_pubkey,
)

logger = logging.getLogger(__name__)


@dataclass
class CreditTransaction:
    tx_id: str
    from_peer: str
    to_peer: str
    amount: float
    reason: str
    timestamp: float
    nonce: int
    signature: bytes = b""
    from_pubkey: bytes = b""


class CreditLedger:
    def __init__(
        self,
        peer_id: str,
        db_path: Path,
        bootstrap_credits: int = 1000,
        identity: PeerIdentity | None = None,
    ):
        self._peer_id = peer_id
        self._db_path = db_path
        self._bootstrap = bootstrap_credits
        self._identity = identity
        self._nonces: dict[str, int] = {}
        self._peer_pubkeys: dict[str, bytes] = {}
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                from_peer TEXT NOT NULL,
                to_peer TEXT NOT NULL,
                amount REAL NOT NULL,
                reason TEXT,
                timestamp REAL NOT NULL,
                nonce INTEGER NOT NULL,
                signature BLOB,
                from_pubkey BLOB
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS balances (
                peer_id TEXT PRIMARY KEY,
                balance REAL NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS peer_keys (
                peer_id TEXT PRIMARY KEY,
                pubkey BLOB NOT NULL
            )
        """)
        row = conn.execute(
            "SELECT balance FROM balances WHERE peer_id = ?", (self._peer_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO balances (peer_id, balance) VALUES (?, ?)",
                (self._peer_id, self._bootstrap),
            )
        # Load known peer public keys
        for row in conn.execute("SELECT peer_id, pubkey FROM peer_keys").fetchall():
            self._peer_pubkeys[row[0]] = row[1]
        # Load nonce high-water marks
        for row in conn.execute(
            "SELECT from_peer, MAX(nonce) FROM transactions WHERE to_peer = ? GROUP BY from_peer",
            (self._peer_id,),
        ).fetchall():
            self._nonces[row[0]] = row[1]
        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def register_peer_key(self, peer_id: str, pubkey: bytes) -> None:
        self._peer_pubkeys[peer_id] = pubkey
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO peer_keys (peer_id, pubkey) VALUES (?, ?)",
            (peer_id, pubkey),
        )
        conn.commit()
        conn.close()

    @property
    def balance(self) -> float:
        conn = self._connect()
        row = conn.execute(
            "SELECT balance FROM balances WHERE peer_id = ?", (self._peer_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else 0.0

    def balance_with(self, peer_id: str) -> float:
        conn = self._connect()
        earned = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE from_peer = ? AND to_peer = ?",
            (peer_id, self._peer_id),
        ).fetchone()[0]
        spent = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE from_peer = ? AND to_peer = ?",
            (self._peer_id, peer_id),
        ).fetchone()[0]
        conn.close()
        return earned - spent

    def debit(self, to_peer: str, amount: float, reason: str = "") -> CreditTransaction:
        nonce = self._nonces.get(to_peer, 0) + 1
        self._nonces[to_peer] = nonce
        ts = time.time()
        tx_id = uuid.uuid4().hex[:16]

        signature = b""
        pubkey = b""
        if self._identity:
            msg = transaction_message(tx_id, self._peer_id, to_peer, amount, reason, ts, nonce)
            signature = self._identity.sign(msg)
            pubkey = self._identity.public_key_bytes()

        tx = CreditTransaction(
            tx_id=tx_id,
            from_peer=self._peer_id,
            to_peer=to_peer,
            amount=amount,
            reason=reason,
            timestamp=ts,
            nonce=nonce,
            signature=signature,
            from_pubkey=pubkey,
        )
        self._record(tx)
        self._update_balance(self._peer_id, -amount)
        logger.debug("Debited %.2f credits to %s: %s", amount, to_peer, reason)
        return tx

    def credit(self, tx: CreditTransaction) -> None:
        if tx.to_peer != self._peer_id:
            raise ValueError("Transaction not addressed to this peer")
        last_nonce = self._nonces.get(tx.from_peer, 0)
        if tx.nonce <= last_nonce:
            raise ValueError(f"Stale nonce {tx.nonce} <= {last_nonce}")

        if tx.signature and tx.from_pubkey:
            msg = transaction_message(
                tx.tx_id, tx.from_peer, tx.to_peer,
                tx.amount, tx.reason, tx.timestamp, tx.nonce,
            )
            if not verify_with_pubkey(tx.from_pubkey, msg, tx.signature):
                raise ValueError("Invalid transaction signature")
            self.register_peer_key(tx.from_peer, tx.from_pubkey)

        self._nonces[tx.from_peer] = tx.nonce
        self._record(tx)
        self._update_balance(self._peer_id, tx.amount)
        logger.debug("Credited %.2f from %s", tx.amount, tx.from_peer)

    def _record(self, tx: CreditTransaction) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT OR IGNORE INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tx.tx_id, tx.from_peer, tx.to_peer, tx.amount, tx.reason,
             tx.timestamp, tx.nonce, tx.signature, tx.from_pubkey),
        )
        conn.commit()
        conn.close()

    def _update_balance(self, peer_id: str, delta: float) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO balances (peer_id, balance) VALUES (?, ?) "
            "ON CONFLICT(peer_id) DO UPDATE SET balance = balance + ?",
            (peer_id, delta, delta),
        )
        conn.commit()
        conn.close()

    def total_earned(self) -> float:
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE to_peer = ?",
            (self._peer_id,),
        ).fetchone()
        conn.close()
        return row[0]

    def total_spent(self) -> float:
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE from_peer = ?",
            (self._peer_id,),
        ).fetchone()
        conn.close()
        return row[0]

    def recent_transactions(self, limit: int = 20) -> list[CreditTransaction]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT tx_id, from_peer, to_peer, amount, reason, timestamp, nonce, "
            "signature, from_pubkey "
            "FROM transactions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            CreditTransaction(
                tx_id=r[0], from_peer=r[1], to_peer=r[2],
                amount=r[3], reason=r[4], timestamp=r[5], nonce=r[6],
                signature=r[7] or b"", from_pubkey=r[8] or b"",
            )
            for r in rows
        ]
