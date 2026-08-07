"""Teacher adapters for π0.5 / π0.6 (Physical Intelligence openpi).

The full VLA cannot run on the wearable SoC. We roll out the teacher on
aligned motion sequences and project action chunks onto bilateral hip
torques.

Submodules (GitHub click-through folders):
    teachers/pi0.5  →  https://github.com/Physical-Intelligence/openpi
    teachers/pi0.6  →  https://github.com/Physical-Intelligence/openpi
                       (π0.6 lands in the same upstream when released)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import torch

TeacherName = Literal["pi0.5", "pi0.6"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEACHER_DIRS = {
    "pi0.5": _REPO_ROOT / "teachers" / "pi0.5",
    "pi0.6": _REPO_ROOT / "teachers" / "pi0.6",
}


@dataclass
class TeacherSample:
    """One distillation tuple (s_t, a_T[, h_T, log π_T])."""

    state: torch.Tensor          # [STATE_DIM]
    action: torch.Tensor         # [ACTION_DIM] hip torques
    teacher_feat: Optional[torch.Tensor] = None
    teacher_log_prob: Optional[torch.Tensor] = None


def teacher_path(name: TeacherName = "pi0.5") -> Path:
    return _TEACHER_DIRS[name]


def ensure_teacher_on_path(name: TeacherName = "pi0.5") -> Path:
    """Add the submodule checkout to sys.path if present."""
    path = teacher_path(name)
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"Teacher checkout missing at {path}.\n"
            f"Init the git submodule:\n"
            f"  git submodule update --init --depth 1 teachers/{name}\n"
            f"GitHub link: https://github.com/Physical-Intelligence/openpi"
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


class TorqueProjector:
    """Project high-dim VLA action chunks onto bilateral hip torques.

    Default heuristic: map left/right arm or leg DoFs (indices configurable)
    through a learned/linear map into [Nm] clamp range.
    """

    def __init__(
        self,
        in_dim: int,
        left_idx: Sequence[int] = (0,),
        right_idx: Sequence[int] = (1,),
        scale: float = 20.0,
        torque_min: float = -40.0,
        torque_max: float = 40.0,
    ):
        self.in_dim = in_dim
        self.left_idx = list(left_idx)
        self.right_idx = list(right_idx)
        self.scale = scale
        self.torque_min = torque_min
        self.torque_max = torque_max
        # Optional learned map: action_chunk → 2 torques
        self.linear = torch.nn.Linear(in_dim, 2)

    def __call__(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_chunk: [..., A] continuous teacher actions
        Returns:
            torques: [..., 2]
        """
        if action_chunk.shape[-1] == 2:
            torques = action_chunk * self.scale
        else:
            # Prefer explicit DoF indices when available, else linear map
            if max(self.left_idx + self.right_idx) < action_chunk.shape[-1]:
                left = action_chunk[..., self.left_idx].mean(dim=-1)
                right = action_chunk[..., self.right_idx].mean(dim=-1)
                torques = torch.stack([left, right], dim=-1) * self.scale
            else:
                flat = action_chunk.reshape(-1, action_chunk.shape[-1])
                torques = self.linear(flat).reshape(
                    *action_chunk.shape[:-1], 2
                )
        return torques.clamp(self.torque_min, self.torque_max)


class TeacherPolicy:
    """Thin wrapper around openpi π0.5 / π0.6 checkpoints.

    When openpi is not initialised, falls back to a synthetic teacher so the
    distillation pipeline remains runnable for CI / unit tests.
    """

    def __init__(
        self,
        name: TeacherName = "pi0.5",
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        synthetic: bool = False,
        action_dim_teacher: int = 32,
        feat_dim: int = 256,
    ):
        self.name = name
        self.device = device
        self.feat_dim = feat_dim
        self.projector = TorqueProjector(in_dim=action_dim_teacher)
        self._policy = None
        self._synthetic = synthetic

        if not synthetic:
            try:
                ensure_teacher_on_path(name)
                self._try_load_openpi(checkpoint)
            except Exception as exc:  # noqa: BLE001 — fall back for offline use
                print(f"[TeacherPolicy] openpi unavailable ({exc}); using synthetic teacher")
                self._synthetic = True

    def _try_load_openpi(self, checkpoint: Optional[str]) -> None:
        # Lazy import — openpi layout varies by release.
        # Supported entry points (best-effort):
        #   openpi.policies.policy_config.get_policy
        #   openpi.models.model
        try:
            from openpi.training import config as _cfg  # type: ignore
            from openpi.policies import policy_config as _pc  # type: ignore

            cfg_name = "pi05_droid" if self.name == "pi0.5" else "pi05_droid"
            train_cfg = _cfg.get_config(cfg_name)
            ckpt = checkpoint or train_cfg.checkpoint_dir
            self._policy = _pc.create_trained_policy(train_cfg, ckpt)
            print(f"[TeacherPolicy] loaded openpi policy '{cfg_name}' from {ckpt}")
        except Exception:
            # Placeholder: mark synthetic if create_trained_policy is unavailable
            raise RuntimeError(
                "openpi is checked out but policy loader API did not resolve; "
                "pass synthetic=True or install openpi extras."
            )

    @torch.no_grad()
    def act(
        self,
        observation: dict,
        state: torch.Tensor,
        language: str = "assist hip walking",
    ) -> TeacherSample:
        """Produce a TeacherSample for one multimodal exoskeleton state."""
        if self._synthetic or self._policy is None:
            return self._synthetic_act(state)

        # observation is camera / proprio dict expected by openpi
        result = self._policy.infer({**observation, "prompt": language})
        action = result["actions"]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        # Take first action chunk step, project to hip torques
        chunk = action[0] if action.ndim > 1 else action
        torques = self.projector(chunk.to(self.device))
        feat = None
        if "policy_features" in result:
            feat = torch.as_tensor(result["policy_features"], device=self.device).float()
        return TeacherSample(state=state, action=torques, teacher_feat=feat)

    def _synthetic_act(self, state: torch.Tensor) -> TeacherSample:
        """Deterministic pseudo-teacher for pipeline smoke tests."""
        # Map IMU/joint slice → smooth hip torques
        s = state.detach().float()
        if s.ndim == 1:
            s = s.unsqueeze(0)
        # joints live at indices [12:16]
        joints = s[:, 12:16]
        left = 15.0 * torch.tanh(joints[:, 0] + 0.5 * joints[:, 1])
        right = 15.0 * torch.tanh(joints[:, 2] + 0.5 * joints[:, 3])
        action = torch.stack([left, right], dim=-1).squeeze(0)
        # Fake teacher embedding from state
        feat = torch.zeros(self.feat_dim, device=s.device)
        feat[: min(self.feat_dim, s.shape[-1])] = s.squeeze(0)[: self.feat_dim]
        return TeacherSample(
            state=state.squeeze(0) if state.ndim > 1 else state,
            action=action if action.ndim == 1 else action.squeeze(0),
            teacher_feat=feat,
        )


def collect_distillation_batch(
    teacher: TeacherPolicy,
    states: torch.Tensor,
    observations: Optional[Sequence[dict]] = None,
    language: str = "assist hip walking",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll out teacher on a batch of states → (s, a_T, h_T)."""
    actions, feats = [], []
    for i in range(states.shape[0]):
        obs = observations[i] if observations is not None else {}
        sample = teacher.act(obs, states[i], language=language)
        actions.append(sample.action)
        feat = sample.teacher_feat
        if feat is None:
            feat = torch.zeros(teacher.feat_dim)
        feats.append(feat)
    return states, torch.stack(actions), torch.stack(feats)
