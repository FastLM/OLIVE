"""OLIVE distillation: π0.5 / π0.6 → BaseController (frozen W0).

    L_KD     = E[ ‖π_W0(s) − a_T‖² + β KL(π_T ‖ π_W0) ]
    L_feat   = E[ ‖h_W0(s) − P h_T(o,ℓ)‖² ]
    L_distill = L_KD + λ_feat L_feat

After training, W0 is frozen and consumed by the online C++ runtime.
"""

from .student import BaseController, GateRankNet
from .losses import DistillLoss, distill_step
from .export_w0 import export_w0_binary, load_w0_binary

__all__ = [
    "BaseController",
    "GateRankNet",
    "DistillLoss",
    "distill_step",
    "export_w0_binary",
    "load_w0_binary",
]
