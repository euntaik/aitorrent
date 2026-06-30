from __future__ import annotations

import random
import logging

import torch

logger = logging.getLogger(__name__)

CHALLENGE_PROBABILITY = 0.05
TOLERANCE_RTOL = 1e-3
TOLERANCE_ATOL = 1e-5


def should_challenge() -> bool:
    return random.random() < CHALLENGE_PROBABILITY


def create_challenge(
    hidden_size: int, dtype: torch.dtype = torch.float16, device: str = "cpu"
) -> torch.Tensor:
    return torch.randn(1, 1, hidden_size, dtype=dtype, device=device)


def verify_response(response: torch.Tensor, expected: torch.Tensor) -> bool:
    return torch.allclose(
        response.cpu().float(),
        expected.cpu().float(),
        rtol=TOLERANCE_RTOL,
        atol=TOLERANCE_ATOL,
    )
