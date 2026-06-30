from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from aitorrent.credit.ledger import CreditLedger


@dataclass
class TitForTatManager:
    ledger: CreditLedger
    unchoke_slots: int = 4
    optimistic_interval_sec: int = 30
    _unchoked: set[str] = field(default_factory=set)
    _last_optimistic: float = 0.0

    def rank_peers(self, peer_ids: list[str]) -> list[str]:
        scored = []
        for pid in peer_ids:
            balance = self.ledger.balance_with(pid)
            scored.append((pid, balance))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scored]

    def update_unchoked(self, active_peers: list[str]) -> set[str]:
        ranked = self.rank_peers(active_peers)
        self._unchoked = set(ranked[: self.unchoke_slots])

        now = time.time()
        if now - self._last_optimistic > self.optimistic_interval_sec:
            choked = [p for p in active_peers if p not in self._unchoked]
            if choked:
                lucky = random.choice(choked)
                self._unchoked.add(lucky)
            self._last_optimistic = now

        return self._unchoked

    def is_unchoked(self, peer_id: str) -> bool:
        return peer_id in self._unchoked

    def should_serve(self, peer_id: str) -> bool:
        if self.is_unchoked(peer_id):
            return True
        return self.ledger.balance_with(peer_id) > 0
