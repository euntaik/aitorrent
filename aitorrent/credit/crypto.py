"""Ed25519 peer identity and transaction signing."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)


@dataclass
class PeerIdentity:
    peer_id: str
    _private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey

    @classmethod
    def generate(cls, save_path: Path | None = None) -> PeerIdentity:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        pub_bytes = public_key.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        peer_id = hashlib.sha256(pub_bytes).hexdigest()[:16]

        identity = cls(peer_id=peer_id, _private_key=private_key, public_key=public_key)
        if save_path:
            identity.save(save_path)
        return identity

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = self._private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        path.write_bytes(pem)
        pub_path = path.with_suffix(".pub")
        pub_pem = self.public_key.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        pub_path.write_bytes(pub_pem)
        logger.info("Identity saved: %s -> %s", self.peer_id, path)

    @classmethod
    def load(cls, path: Path) -> PeerIdentity:
        pem = path.read_bytes()
        private_key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("Not an Ed25519 key")
        public_key = private_key.public_key()
        pub_bytes = public_key.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        peer_id = hashlib.sha256(pub_bytes).hexdigest()[:16]
        return cls(peer_id=peer_id, _private_key=private_key, public_key=public_key)

    @classmethod
    def load_or_create(cls, path: Path) -> PeerIdentity:
        if path.exists():
            return cls.load(path)
        return cls.generate(save_path=path)

    def sign(self, message: bytes) -> bytes:
        return self._private_key.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            self.public_key.verify(signature, message)
            return True
        except Exception:
            return False

    def public_key_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )


def load_public_key(pub_bytes: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(pub_bytes)


def verify_with_pubkey(pub_bytes: bytes, message: bytes, signature: bytes) -> bool:
    try:
        key = load_public_key(pub_bytes)
        key.verify(signature, message)
        return True
    except Exception:
        return False


def transaction_message(
    tx_id: str,
    from_peer: str,
    to_peer: str,
    amount: float,
    reason: str,
    timestamp: float,
    nonce: int,
) -> bytes:
    """Canonical byte representation of a transaction for signing."""
    canonical = json.dumps(
        {
            "tx_id": tx_id,
            "from": from_peer,
            "to": to_peer,
            "amount": amount,
            "reason": reason,
            "ts": timestamp,
            "nonce": nonce,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical.encode("utf-8")
