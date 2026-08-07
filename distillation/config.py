"""Dimension constants mirroring include/olive/config.hpp."""

# Sensor / state (must match C++ STATE_DIM = 46)
IMU_DIM = 12
JOINT_DIM = 4
EMG_DIM = 8
VIB_DIM = 2
CTX_DIM = 4
HISTORY_DIM = 16
STATE_DIM = IMU_DIM + JOINT_DIM + EMG_DIM + VIB_DIM + CTX_DIM + HISTORY_DIM

ACTION_DIM = 2
D = 128
K = 128
GATE_HIDDEN = 64

R_MIN = 4
R_MAX = 16

TORQUE_MIN = -40.0
TORQUE_MAX = 40.0

# Distillation defaults
BETA_KL = 0.1
LAMBDA_FEAT = 0.5
DEFAULT_LR = 1e-3
