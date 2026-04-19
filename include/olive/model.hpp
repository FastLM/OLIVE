#pragma once
// OLIVE Adaptive Controller
//
// Policy decomposition:
//   Θ_t  =  W0  +  ΔW_t
//   ΔW_t =  A_t[:, :r_t]  ×  B_t[:, :r_t]^T       (Eq. 2)
//
// Gated action:
//   a_t  =  π_{W0 + α_t ΔW_t}(s_t)                  (Eq. 5)
//
// Where:
//   W0  — frozen base policy (D×D middle layer), initialised from Vigx-MM
//   A_t ∈ R^{D × r_max},  B_t ∈ R^{D × r_max}  — online low-rank factors
//   α_t = σ(g(s_t, h_{t-1}))          — gating scalar     (Eq. 4)
//   r_t = clip(⌊c_t·|R|⌋ + r_min, r_min, r_max)  — dynamic rank (Eq. 7)
//   c_t = σ(ψ(s_t, h_{t-1}))          — complexity estimate (Eq. 6)
//
// Full 3-layer policy network around the adaptive middle layer:
//   h1  = ReLU(W1 · s_t + b1)                       [STATE_DIM → D]
//   h2  = ReLU((W2 + α_t ΔW_t) · h1 + b2)           [D → D]
//   a_t = clamp(W3 · h2 + b3, TORQUE_MIN, TORQUE_MAX) [D → ACTION_DIM]
//
// Gating + rank-scheduling networks share their first hidden layer
// (saves 0.3 ms on the embedded SoC).

#include "config.hpp"
#include "matrix.hpp"
#include <array>

namespace olive {

// ── GateRankNet ───────────────────────────────────────────────────────────
// Shared two-head network.
//  Input : s_t ∈ R^{STATE_DIM}
//  Hidden: Linear(STATE_DIM → GATE_HIDDEN) → ReLU   [shared]
//  Head 1: Linear(GATE_HIDDEN → 1) → Sigmoid → α_t
//  Head 2: Linear(GATE_HIDDEN → 1) → Sigmoid → c_t
//
// All weights FROZEN (pretrained offline with the base policy).
struct GateRankNet {
    MatrixXf Wh;    // [GATE_HIDDEN × STATE_DIM]  shared hidden
    VectorXf bh;    // [GATE_HIDDEN]
    VectorXf wg;    // [GATE_HIDDEN]               gate head
    float    bg;
    VectorXf wc;    // [GATE_HIDDEN]               complexity head
    float    bc;

    explicit GateRankNet();

    // Returns {α_t, c_t} ∈ (0,1)²
    std::pair<float, float> forward(const VectorXf& state) const;
};

// ── OLIVEModel ────────────────────────────────────────────────────────────
class OLIVEModel {
public:
    explicit OLIVEModel();

    // Load frozen W0 weights from binary file.
    // Layout: [W1:D×n | b1:D | W2:D×D | b2:D | W3:m×D | b3:m
    //          | GateRankNet weights]
    bool load_base_weights(const char* path);

    // ── Forward pass (inference + online update) ──────────────────────
    // Returns a_t ∈ R^{ACTION_DIM} (bilateral hip torques [Nm]).
    // Internally selects r_t and computes α_t; caller can query
    // last_alpha() and last_rank() afterwards.
    VectorXf forward(const VectorXf& state);

    // ── Accessors for the trainer ─────────────────────────────────────
    float   last_alpha()     const { return alpha_t_; }
    int     last_rank()      const { return r_t_; }
    float   last_complexity()const { return c_t_; }

    // Low-rank factor references (trainer updates these in-place)
    MatrixXf& A() { return A_t_; }
    MatrixXf& B() { return B_t_; }
    const MatrixXf& A() const { return A_t_; }
    const MatrixXf& B() const { return B_t_; }

    // Base policy layers (read-only for trainer gradient computation)
    const MatrixXf& W1() const { return W1_; }
    const VectorXf& b1() const { return b1_; }
    const MatrixXf& W2() const { return W2_; }
    const VectorXf& b2() const { return b2_; }
    const MatrixXf& W3() const { return W3_; }
    const VectorXf& b3() const { return b3_; }

    // Intermediate activations cached from last forward (for backprop)
    const VectorXf& cached_h1()    const { return h1_; }
    const VectorXf& cached_h2()    const { return h2_; }
    const VectorXf& cached_state() const { return s_; }

    // Clamp ΔW_t Frobenius norm (Lyapunov stability, §3.5)
    void clamp_residual();

private:
    // ── Frozen base policy weights (W0) ───────────────────────────────
    MatrixXf W1_;   // [D × STATE_DIM]
    VectorXf b1_;   // [D]
    MatrixXf W2_;   // [D × D]   ← low-rank adaptation applied here
    VectorXf b2_;   // [D]
    MatrixXf W3_;   // [ACTION_DIM × D]
    VectorXf b3_;   // [ACTION_DIM]

    GateRankNet gate_rank_net_;

    // ── Online low-rank factors (only these are updated) ─────────────
    MatrixXf A_t_;  // [D × R_MAX]
    MatrixXf B_t_;  // [D × R_MAX]

    // ── Cached scalars from last forward ─────────────────────────────
    float alpha_t_  = 0.0f;
    float c_t_      = 0.0f;
    int   r_t_      = R_MIN;

    // ── Cached activations (needed by trainer for gradient) ───────────
    VectorXf s_;    // input state
    VectorXf h1_;   // post-ReLU hidden-1
    VectorXf h2_;   // post-ReLU hidden-2

    // Select rank from candidate set given complexity scalar
    int select_rank(float c) const;
};

} // namespace olive
