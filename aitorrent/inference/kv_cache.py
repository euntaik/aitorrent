from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    session_id: str
    cache_data: dict
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed = time.time()


class KVCacheManager:
    def __init__(self, ttl_sec: int = 300, max_sessions: int = 100):
        self._ttl = ttl_sec
        self._max_sessions = max_sessions
        self._entries: dict[str, CacheEntry] = {}

    def get(self, session_id: str) -> dict | None:
        entry = self._entries.get(session_id)
        if entry is None:
            return None
        if time.time() - entry.last_accessed > self._ttl:
            self.evict(session_id)
            return None
        entry.touch()
        return entry.cache_data

    def put(self, session_id: str, cache_data: dict) -> None:
        if len(self._entries) >= self._max_sessions:
            self._evict_oldest()
        self._entries[session_id] = CacheEntry(
            session_id=session_id, cache_data=cache_data
        )

    def evict(self, session_id: str) -> None:
        entry = self._entries.pop(session_id, None)
        if entry:
            logger.debug("Evicted KV cache for session %s", session_id)

    def _evict_oldest(self) -> None:
        if not self._entries:
            return
        oldest = min(self._entries.values(), key=lambda e: e.last_accessed)
        self.evict(oldest.session_id)

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = [
            sid for sid, entry in self._entries.items()
            if now - entry.last_accessed > self._ttl
        ]
        for sid in expired:
            self.evict(sid)
        return len(expired)

    @property
    def active_sessions(self) -> int:
        return len(self._entries)
