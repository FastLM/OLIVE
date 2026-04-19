#pragma once
#include <array>
#include <cstddef>

namespace olive {

// ═══════════════════════════════════════════════════════════════════
//  Hardware: Vigx Ant-H1 Pro  (ARM-based SoC, <10 ms budget)
//  Sensor rates: IMU @ 1000 Hz, EMG @ 2000 Hz, vibration @ 1000 Hz
// ═══════════════════════════════════════════════════════════════════

// ── Sensor dimensions ───────────────────────────────────────────────
// Bilateral IMU: 3-axis accel + 3-axis gyro × 2 sides
static constexpr int IMU_DIM      = 12;
// Bilateral hip joint: angle + velocity × 2 sides
static constexpr int JOINT_DIM    = 4;
// 8-channel Delsys Trigno surface EMG (bilateral)
static constexpr int EMG_DIM      = 8;
// Bilateral actuator force sensors (ground-contact vibration)
static constexpr int VIB_DIM      = 2;
// Context: one-hot intent {walk, climb, slope, uneven}
static constexpr int CTX_DIM      = 4;
// EMA motion history vector
static constexpr int HISTORY_DIM  = 16;

// Total state dimension n
static constexpr int STATE_DIM =
    IMU_DIM + JOINT_DIM + EMG_DIM + VIB_DIM + CTX_DIM + HISTORY_DIM; // = 46

// Output: bilateral hip torques [left, right]
static constexpr int ACTION_DIM   = 2;

// ── Policy network widths ────────────────────────────────────────────
// W0: Linear(STATE_DIM→D) → ReLU → Linear(D→D) → ReLU → Linear(D→ACTION_DIM)
// Low-rank adaptation applied to the middle D×D layer.
static constexpr int D = 128;   // d = k = 128 as in the paper
static constexpr int K = 128;

// ── Low-rank adaptation ──────────────────────────────────────────────
static constexpr int R_MIN    = 4;
static constexpr int R_MAX    = 16;
static constexpr int NUM_RANKS = R_MAX - R_MIN + 1;  // candidate set |R|

// Rank candidate set: {4, 8, 12, 16} — equally spaced
static constexpr std::array<int, 4> RANK_CANDIDATES = {4, 8, 12, 16};

// ── Gating / complexity network ──────────────────────────────────────
// Input: concat(s_t, h_{t-1}) dim = STATE_DIM (h is already in s)
// Hidden: 64 → shared first layer for gate + complexity heads
static constexpr int GATE_HIDDEN = 64;

// ── Online update hyperparameters ────────────────────────────────────
static constexpr float LEARNING_RATE  = 1e-3f;
static constexpr float EMA_ALPHA      = 0.95f;    // history EMA decay

// Reward weights  w1, w2, w3  (Eq. 9)
static constexpr float W_EMG          = 0.4f;
static constexpr float W_EFFORT       = 0.3f;
static constexpr float W_STABILITY    = 0.3f;

// Loss weights  λ1, λ2, λ3  (Eq. 10)
static constexpr float LAMBDA_REWARD  = 1.0f;
static constexpr float LAMBDA_SMOOTH  = 0.05f;
static constexpr float LAMBDA_STAB   = 0.05f;

// Torque output clamp [Nm]
static constexpr float TORQUE_MIN     = -40.0f;
static constexpr float TORQUE_MAX     =  40.0f;

// Safety: max Frobenius norm of ΔW (Lyapunov bound)
static constexpr float DELTA_W_MAX   = 5.0f;

// ── Runtime budget ───────────────────────────────────────────────────
// End-to-end latency budget on embedded SoC: <10 ms
static constexpr double LATENCY_BUDGET_MS = 10.0;
static constexpr double CONTROL_FREQ_HZ   = 100.0;  // 100 Hz control loop

} // namespace olive
