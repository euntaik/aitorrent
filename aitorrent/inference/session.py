from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class InferenceSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    model_id: str = ""
    total_tokens: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    total_credits_spent: float = 0.0

    def record_tokens(self, count: int, credits: float) -> None:
        self.total_tokens += count
        self.total_credits_spent += credits
        self.last_active = time.time()

    @property
    def age_sec(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_sec(self) -> float:
        return time.time() - self.last_active
