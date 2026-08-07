"""BaseController — compact sensor→torque policy aligned with C++ OLIVEModel.

Policy:
    h1 = ReLU(W1 s + b1)          [STATE_DIM → D]
    h2 = ReLU(W2 h1 + b2)         [D → D]
    a  = clamp(W3 h2 + b3)        [D → ACTION_DIM]

GateRankNet (frozen after distillation, used online):
    shared hidden → α_t (gate), c_t (complexity)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ACTION_DIM,
    D,
    GATE_HIDDEN,
    STATE_DIM,
    TORQUE_MAX,
    TORQUE_MIN,
)


class GateRankNet(nn.Module):
    """Shared two-head network producing α_t and c_t ∈ (0, 1)."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = GATE_HIDDEN):
        super().__init__()
        self.Wh = nn.Linear(state_dim, hidden)
        self.wg = nn.Linear(hidden, 1)
        self.wc = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.Wh(state))
        alpha = torch.sigmoid(self.wg(h)).squeeze(-1)
        c = torch.sigmoid(self.wc(h)).squeeze(-1)
        return alpha, c


class BaseController(nn.Module):
    """Frozen base policy W0: multimodal sensors → bilateral hip torques."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden: int = D,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden = hidden
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)
        self.gate_rank = GateRankNet(state_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

    def encode(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h1, h2) intermediate features for L_feat."""
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        return h1, h2

    def forward(
        self, state: torch.Tensor, return_features: bool = False
    ):
        h1, h2 = self.encode(state)
        a = self.fc3(h2)
        a = a.clamp(TORQUE_MIN, TORQUE_MAX)
        if return_features:
            return a, h2
        return a

    def action_log_prob(
        self, state: torch.Tensor, action: torch.Tensor, std: float = 0.1
    ) -> torch.Tensor:
        """Diagonal Gaussian log-prob for KL / distribution matching."""
        mean = self.forward(state)
        var = std * std
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var + math.log(2 * math.pi * var)
        ).sum(dim=-1)
        return log_prob

    def action_distribution(
        self, state: torch.Tensor, std: float = 0.1
    ) -> torch.distributions.Normal:
        mean = self.forward(state)
        return torch.distributions.Normal(mean, std)
