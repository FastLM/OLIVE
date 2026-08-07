"""Distillation losses.

    L_KD = E[ ‖π_W0(s) − a_T‖²_2  +  β D_KL(π_T ‖ π_W0) ]
    L_feat = E[ ‖h_W0(s) − P h_T‖²_2 ]
    L_distill = L_KD + λ_feat L_feat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BETA_KL, D, LAMBDA_FEAT
from .student import BaseController


class FeatureProjector(nn.Module):
    """Learned linear projector P: teacher_feat_dim → student hidden D."""

    def __init__(self, teacher_dim: int, student_dim: int = D):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.proj(h_t)


@dataclass
class DistillLossOutput:
    total: torch.Tensor
    action_mimicry: torch.Tensor
    kl: torch.Tensor
    feat: torch.Tensor


class DistillLoss(nn.Module):
    def __init__(
        self,
        teacher_feat_dim: int,
        beta: float = BETA_KL,
        lambda_feat: float = LAMBDA_FEAT,
        student_dim: int = D,
        action_std: float = 0.1,
    ):
        super().__init__()
        self.beta = beta
        self.lambda_feat = lambda_feat
        self.action_std = action_std
        self.projector = FeatureProjector(teacher_feat_dim, student_dim)

    def forward(
        self,
        student: BaseController,
        state: torch.Tensor,
        teacher_action: torch.Tensor,
        teacher_feat: Optional[torch.Tensor] = None,
        teacher_log_prob: Optional[torch.Tensor] = None,
    ) -> DistillLossOutput:
        """
        Args:
            state:           [B, STATE_DIM] exoskeleton multimodal state
            teacher_action:  [B, ACTION_DIM] torque projection of π_T chunks
            teacher_feat:    [B, teacher_dim] optional intermediate h_T
            teacher_log_prob:[B] optional log π_T(a|o,ℓ) for KL term
        """
        student_action, h_w0 = student(state, return_features=True)

        # Action mimicry ‖π_W0(s) − a_T‖²
        action_mse = F.mse_loss(student_action, teacher_action)

        # Distribution matching β KL(π_T ‖ π_W0)
        # If teacher log-probs are provided, use reverse KL proxy via
        #   KL ≈ E_a~π_T [log π_T − log π_W0]; else use MSE-as-Gaussian KL.
        student_dist = student.action_distribution(state, std=self.action_std)
        if teacher_log_prob is not None:
            log_pw0 = student_dist.log_prob(teacher_action).sum(dim=-1)
            kl = (teacher_log_prob - log_pw0).mean().clamp_min(0.0)
        else:
            # Closed-form KL between N(a_T, σ²I) and N(μ_W0, σ²I)
            kl = (0.5 / (self.action_std ** 2)) * F.mse_loss(
                student_action, teacher_action, reduction="mean"
            )

        l_kd = action_mse + self.beta * kl

        # Feature distillation
        if teacher_feat is not None:
            projected = self.projector(teacher_feat)
            l_feat = F.mse_loss(h_w0, projected)
        else:
            l_feat = torch.zeros((), device=state.device, dtype=state.dtype)

        total = l_kd + self.lambda_feat * l_feat
        return DistillLossOutput(
            total=total,
            action_mimicry=action_mse.detach(),
            kl=kl.detach() if torch.is_tensor(kl) else torch.tensor(kl),
            feat=l_feat.detach() if torch.is_tensor(l_feat) else torch.tensor(l_feat),
        )


def distill_step(
    student: BaseController,
    criterion: DistillLoss,
    optimizer: torch.optim.Optimizer,
    state: torch.Tensor,
    teacher_action: torch.Tensor,
    teacher_feat: Optional[torch.Tensor] = None,
    teacher_log_prob: Optional[torch.Tensor] = None,
) -> DistillLossOutput:
    """One gradient step on L_distill."""
    optimizer.zero_grad(set_to_none=True)
    out = criterion(
        student, state, teacher_action, teacher_feat, teacher_log_prob
    )
    out.total.backward()
    optimizer.step()
    return out
