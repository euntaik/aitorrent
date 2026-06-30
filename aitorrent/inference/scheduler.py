from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    request_id: str
    model_id: str
    input_ids: list[int]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    queued_at: float = field(default_factory=time.time)


class InferenceScheduler:
    def __init__(self, max_batch_size: int = 8, max_wait_ms: int = 50):
        self._max_batch = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._running = False

    async def submit(self, request: InferenceRequest) -> None:
        await self._queue.put(request)

    async def get_batch(self) -> list[InferenceRequest]:
        batch = []
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            batch.append(first)
        except asyncio.TimeoutError:
            return batch

        deadline = time.time() + self._max_wait
        while len(batch) < self._max_batch and time.time() < deadline:
            try:
                remaining = max(0.001, deadline - time.time())
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()
