from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CreditPricer:
    base_rate: float = 0.01  # credits per token per layer

    def price_tokens(self, tokens: int, num_layers: int) -> float:
        return self.base_rate * tokens * num_layers

    def price_session(
        self, tokens: int, num_layers: int, num_peers: int
    ) -> float:
        return self.price_tokens(tokens, num_layers)

    def estimate_cost(
        self,
        prompt_tokens: int,
        max_new_tokens: int,
        total_layers: int,
    ) -> float:
        total_tokens = prompt_tokens + max_new_tokens
        return self.price_tokens(total_tokens, total_layers)
