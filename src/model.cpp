#include "olive/model.hpp"
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cstring>

namespace olive {

// ── GateRankNet ──────────────────────────────────────────────────────────

GateRankNet::GateRankNet() {
    Wh = kaiming_uniform(GATE_HIDDEN, STATE_DIM);
    bh = zeros_bias(GATE_HIDDEN);
    wg = VectorXf::Random(GATE_HIDDEN) * 0.1f;
    bg = 0.0f;
    wc = VectorXf::Random(GATE_HIDDEN) * 0.1f;
    bc = 0.0f;
}

std::pair<float, float> GateRankNet::forward(const VectorXf& state) const {
    // Shared hidden layer
    VectorXf h = relu(Wh * state + bh);    // [GATE_HIDDEN]

    // Gating head → α_t ∈ (0,1)
    float alpha = sigmoid(wg.dot(h) + bg);

    // Complexity head → c_t ∈ (0,1)
    float c = sigmoid(wc.dot(h) + bc);

    return {alpha, c};
}

// ── OLIVEModel ────────────────────────────────────────────────────────────

OLIVEModel::OLIVEModel() {
    // Initialise frozen base policy W0 with Kaiming uniform
    // (overwritten by load_base_weights() in production)
    W1_ = kaiming_uniform(D, STATE_DIM);
    b1_ = zeros_bias(D);
    W2_ = kaiming_uniform(D, D);
    b2_ = zeros_bias(D);
    W3_ = kaiming_uniform(ACTION_DIM, D);
    b3_ = zeros_bias(ACTION_DIM);

    // Initialise low-rank factors:
    //   A_t ← small random (matches LoRA convention)
    //   B_t ← zero  (so ΔW_0 = 0, starts from pure base policy)
    A_t_ = MatrixXf::Random(D, R_MAX) * 0.01f;
    B_t_ = MatrixXf::Zero(D, R_MAX);

    // Cache buffers
    s_  = VectorXf::Zero(STATE_DIM);
    h1_ = VectorXf::Zero(D);
    h2_ = VectorXf::Zero(D);
}

bool OLIVEModel::load_base_weights(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    auto read_mat = [&](MatrixXf& M, int rows, int cols) {
        M.resize(rows, cols);
        f.read(reinterpret_cast<char*>(M.data()), rows * cols * sizeof(float));
    };
    auto read_vec = [&](VectorXf& v, int n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    };
    auto read_float = [&](float& x) {
        f.read(reinterpret_cast<char*>(&x), sizeof(float));
    };

    // Policy layers
    read_mat(W1_, D, STATE_DIM);
    read_vec(b1_, D);
    read_mat(W2_, D, D);
    read_vec(b2_, D);
    read_mat(W3_, ACTION_DIM, D);
    read_vec(b3_, ACTION_DIM);

    // GateRankNet weights
    read_mat(gate_rank_net_.Wh, GATE_HIDDEN, STATE_DIM);
    read_vec(gate_rank_net_.bh, GATE_HIDDEN);
    read_vec(gate_rank_net_.wg, GATE_HIDDEN);
    read_float(gate_rank_net_.bg);
    read_vec(gate_rank_net_.wc, GATE_HIDDEN);
    read_float(gate_rank_net_.bc);

    return f.good();
}

int OLIVEModel::select_rank(float c) const {
    // r_t = clip(⌊c·|R|⌋ + r_min, r_min, r_max)
    // Candidate set R = {4, 8, 12, 16}, |R| = 4
    int idx = static_cast<int>(c * static_cast<float>(RANK_CANDIDATES.size()));
    idx = std::clamp(idx, 0, static_cast<int>(RANK_CANDIDATES.size()) - 1);
    return RANK_CANDIDATES[idx];
}

VectorXf OLIVEModel::forward(const VectorXf& state) {
    s_ = state;

    // ── Step 1: Dynamic rank selection ───────────────────────────────
    auto [alpha, c] = gate_rank_net_.forward(state);
    alpha_t_ = alpha;
    c_t_     = c;
    r_t_     = select_rank(c);

    // ── Step 2: Compute ΔW_t using leading r_t columns ───────────────
    // ΔW_t = A_t[:, :r_t] × B_t[:, :r_t]^T
    MatrixXf A_r = A_t_.leftCols(r_t_);   // [D × r_t]
    MatrixXf B_r = B_t_.leftCols(r_t_);   // [D × r_t]
    MatrixXf delta_W = A_r * B_r.transpose();  // [D × D]

    // ── Step 3: Gated adaptive middle layer ──────────────────────────
    // W_eff = W2 + α_t · ΔW_t
    MatrixXf W_eff = W2_ + alpha_t_ * delta_W;  // [D × D]

    // ── Step 4: Forward through 3-layer policy ────────────────────────
    // h1 = ReLU(W1 · s_t + b1)
    h1_ = relu(W1_ * state + b1_);

    // h2 = ReLU(W_eff · h1 + b2)
    h2_ = relu(W_eff * h1_ + b2_);

    // a_t = clamp(W3 · h2 + b3)
    VectorXf a = W3_ * h2_ + b3_;
    a = a.cwiseMax(TORQUE_MIN).cwiseMin(TORQUE_MAX);

    return a;
}

void OLIVEModel::clamp_residual() {
    // Lyapunov bound: ‖α_t ΔW_t‖_F ≤ ‖A_t‖_F · ‖B_t‖_F
    A_t_ = clamp_frobenius(A_t_, DELTA_W_MAX);
    B_t_ = clamp_frobenius(B_t_, DELTA_W_MAX);
}

} // namespace olive
