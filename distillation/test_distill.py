#!/usr/bin/env python3
"""Smoke tests for distillation losses + W0 binary round-trip."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

# Allow `python distillation/test_distill.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillation.config import ACTION_DIM, STATE_DIM
from distillation.export_w0 import export_w0_binary, load_w0_binary
from distillation.losses import DistillLoss, distill_step
from distillation.student import BaseController
from distillation.teacher import TeacherPolicy, collect_distillation_batch


def test_forward_shapes() -> None:
    student = BaseController()
    s = torch.randn(8, STATE_DIM)
    a, h = student(s, return_features=True)
    assert a.shape == (8, ACTION_DIM)
    assert h.shape == (8, 128)
    alpha, c = student.gate_rank(s)
    assert alpha.shape == (8,) and c.shape == (8,)
    assert torch.all((alpha > 0) & (alpha < 1))
    print("PASS test_forward_shapes")


def test_distill_step_decreases_loss() -> None:
    torch.manual_seed(0)
    student = BaseController()
    criterion = DistillLoss(teacher_feat_dim=256)
    opt = torch.optim.Adam(
        list(student.parameters()) + list(criterion.parameters()), lr=1e-2
    )
    s = torch.randn(32, STATE_DIM)
    a_t = torch.randn(32, ACTION_DIM).clamp(-20, 20)
    h_t = torch.randn(32, 256)

    losses = []
    for _ in range(5):
        out = distill_step(student, criterion, opt, s, a_t, h_t)
        losses.append(out.total.item())
    assert losses[-1] < losses[0], f"loss did not drop: {losses}"
    print(f"PASS test_distill_step_decreases_loss ({losses[0]:.3f} → {losses[-1]:.3f})")


def test_export_roundtrip() -> None:
    torch.manual_seed(1)
    student = BaseController()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "w0.bin"
        export_w0_binary(student, path)
        loaded = load_w0_binary(path)
        s = torch.randn(4, STATE_DIM)
        a0 = student(s)
        a1 = loaded(s)
        err = (a0 - a1).abs().max().item()
        assert err < 1e-5, f"round-trip max abs err={err}"
        print(f"PASS test_export_roundtrip (max|Δ|={err:.2e})")


def test_synthetic_teacher_batch() -> None:
    teacher = TeacherPolicy(name="pi0.5", synthetic=True)
    states = torch.randn(16, STATE_DIM)
    s, a, h = collect_distillation_batch(teacher, states)
    assert s.shape == (16, STATE_DIM)
    assert a.shape == (16, ACTION_DIM)
    assert h.shape[0] == 16
    print("PASS test_synthetic_teacher_batch")


if __name__ == "__main__":
    test_forward_shapes()
    test_distill_step_decreases_loss()
    test_export_roundtrip()
    test_synthetic_teacher_batch()
    print("All distillation tests passed.")
