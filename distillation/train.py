#!/usr/bin/env python3
"""Train BaseController by distilling π0.5 / π0.6.

Example:
    # Smoke run with synthetic teacher (no openpi checkout required)
    python -m distillation.train --teacher pi0.5 --synthetic --steps 200 \\
        --export checkpoints/base_controller_w0.bin

    # With openpi submodule initialised
    git submodule update --init --depth 1 teachers/pi0.5
    python -m distillation.train --teacher pi0.5 --checkpoint gs://openpi-assets/checkpoints/pi05_base \\
        --export checkpoints/base_controller_w0.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import BETA_KL, DEFAULT_LR, LAMBDA_FEAT
from .dataset import build_distill_loader
from .export_w0 import export_w0_binary
from .losses import DistillLoss, distill_step
from .student import BaseController


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OLIVE π0.5/π0.6 → BaseController distillation")
    p.add_argument("--teacher", choices=("pi0.5", "pi0.6"), default="pi0.5")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic teacher (CI / smoke)")
    p.add_argument("--checkpoint", type=str, default=None, help="Teacher checkpoint path / gs:// URI")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--beta", type=float, default=BETA_KL)
    p.add_argument("--lambda-feat", type=float, default=LAMBDA_FEAT)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--export", type=str, default="checkpoints/base_controller_w0.bin")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    loader = build_distill_loader(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        teacher_name=args.teacher,
        synthetic_teacher=args.synthetic or args.checkpoint is None,
        seed=args.seed,
    )

    # Infer teacher feature dim from one batch
    batch0 = next(iter(loader))
    feat_dim = batch0["teacher_feat"].shape[-1]

    student = BaseController().to(args.device)
    criterion = DistillLoss(
        teacher_feat_dim=feat_dim,
        beta=args.beta,
        lambda_feat=args.lambda_feat,
    ).to(args.device)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(criterion.parameters()),
        lr=args.lr,
    )

    print(
        f"Distilling {args.teacher} → BaseController | "
        f"steps={args.steps} batch={args.batch_size} device={args.device}"
    )

    step = 0
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        state = batch["state"].to(args.device)
        a_t = batch["teacher_action"].to(args.device)
        h_t = batch.get("teacher_feat")
        if h_t is not None:
            h_t = h_t.to(args.device)

        out = distill_step(student, criterion, optimizer, state, a_t, h_t)
        if step % 50 == 0 or step + 1 == args.steps:
            print(
                f"[{step:5d}/{args.steps}] "
                f"L={out.total.item():.4f}  "
                f"mimic={out.action_mimicry.item():.4f}  "
                f"KL={out.kl.item():.4f}  "
                f"feat={out.feat.item():.4f}"
            )
        step += 1

    export_path = Path(args.export)
    export_w0_binary(student, export_path)
    print(f"Exported frozen W0 → {export_path.resolve()}")
    print("Load in C++:  ./olive_deploy", str(export_path))


if __name__ == "__main__":
    main()
