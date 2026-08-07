"""Distillation datasets: synthetic motion rollouts + on-disk .npz packs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import ACTION_DIM, STATE_DIM
from .teacher import TeacherPolicy, collect_distillation_batch


class DistillDataset(Dataset):
    """In-memory distillation set D_dist = {(s_t, a_T, h_T)}."""

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
    ):
        assert states.shape[0] == actions.shape[0]
        self.states = states.float()
        self.actions = actions.float()
        self.feats = feats.float() if feats is not None else None

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        item = {
            "state": self.states[idx],
            "teacher_action": self.actions[idx],
        }
        if self.feats is not None:
            item["teacher_feat"] = self.feats[idx]
        return item


def make_synthetic_motion_states(
    n: int,
    seed: int = 0,
    gait_hz: float = 1.0,
) -> torch.Tensor:
    """Simulate multimodal exoskeleton states for offline distillation demos."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n / 100.0, n)  # 100 Hz
    w = 2 * np.pi * gait_hz
    states = np.zeros((n, STATE_DIM), dtype=np.float32)

    # IMU accel/gyro (12)
    states[:, 0] = 0.1 * np.sin(w * t)
    states[:, 1] = 9.81
    states[:, 2] = 0.05 * np.cos(w * t)
    states[:, 6] = -0.1 * np.sin(w * t)
    states[:, 7] = 9.81
    states[:, 8] = 0.05 * np.cos(w * t + np.pi)

    # Joints (4) at offset 12
    states[:, 12] = 0.3 * np.sin(w * t)
    states[:, 13] = 0.3 * w * np.cos(w * t)
    states[:, 14] = -0.3 * np.sin(w * t)
    states[:, 15] = -0.3 * w * np.cos(w * t)

    # EMG (8) at offset 16
    for i in range(8):
        states[:, 16 + i] = 50 + 10 * np.abs(np.sin(w * t + i * 0.5))

    # Vibration (2) at offset 24
    states[:, 24] = 5 + 2 * np.abs(np.sin(w * t))
    states[:, 25] = 5 + 2 * np.abs(np.sin(w * t + np.pi))

    # Context one-hot walk at offset 26
    states[:, 26] = 1.0

    # History noise
    states[:, 30:] = rng.normal(0, 0.01, size=(n, STATE_DIM - 30)).astype(np.float32)
    return torch.from_numpy(states)


def build_distill_loader(
    n_samples: int = 4096,
    batch_size: int = 64,
    teacher_name: str = "pi0.5",
    synthetic_teacher: bool = True,
    seed: int = 0,
) -> DataLoader:
    states = make_synthetic_motion_states(n_samples, seed=seed)
    teacher = TeacherPolicy(name=teacher_name, synthetic=synthetic_teacher)  # type: ignore[arg-type]
    states_b, actions, feats = collect_distillation_batch(teacher, states)
    ds = DistillDataset(states_b, actions, feats)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def save_npz(path: Path, states: torch.Tensor, actions: torch.Tensor, feats: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states.cpu().numpy(),
        actions=actions.cpu().numpy(),
        feats=feats.cpu().numpy(),
    )


def load_npz(path: Path) -> DistillDataset:
    z = np.load(path)
    return DistillDataset(
        torch.from_numpy(z["states"]),
        torch.from_numpy(z["actions"]),
        torch.from_numpy(z["feats"]),
    )
