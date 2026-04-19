#include "olive/trainer.hpp"
#include <cmath>

namespace olive {

OnlineTrainer::OnlineTrainer(OLIVEModel& model, float lr)
    : model_(model), lr_(lr) {}

// ── Backpropagation through output layer ──────────────────────────────────
// L_t = f(a_t),  a_t = W3·h2 + b3
// ∂L/∂h2 = W3^T · ∂L/∂a_t
// We approximate ∂L/∂a_t ≈ ∂(−λ1·r + λ2·‖Δa‖² + λ3·φ²)/∂a numerically
// via the already-computed loss scalar (one-step PG approximation).
VectorXf OnlineTrainer::backprop_output_layer(const VectorXf& dL_da) const {
    return model_.W3().transpose() * dL_da;   // [D]
}

// ── Backprop through adaptive hidden layer h2 ─────────────────────────────
// h2 = ReLU(W_eff · h1 + b2),   pre-act: z2 = W_eff·h1 + b2
// ∂L/∂z2 = ∂L/∂h2 ⊙ 1(h2 > 0)      (ReLU mask)
VectorXf OnlineTrainer::backprop_hidden2(const VectorXf& dL_dh2) const {
    const VectorXf& h2 = model_.cached_h2();
    VectorXf mask = (h2.array() > 0.0f).cast<float>();
    return dL_dh2.cwiseProduct(mask);   // [D]  = δ2 (pre-activation gradient)
}

// ── Low-rank factor gradients ─────────────────────────────────────────────
// W_eff = W2 + α_t · A_t[:,:r] · B_t[:,:r]^T
// ∂L/∂(A_t[:,:r]) = δ2 · (α_t · B_t[:,:r] · h1^T)^T  — via chain rule
//                 = α_t · δ2 · h1^T × ... simplified:
//
// Action  a = W3·h2,   h2 = ReLU(W_eff·h1 + b2)
// δ2      = ReLU'(z2) ⊙ (W3^T · ∂L/∂a)       [D]
// ∂L/∂(α·ΔW) = δ2 · h1^T                      [D×D]
// ∂L/∂A_r  = (∂L/∂(α·ΔW)) · B_r              (leading r cols) [D×r]
// ∂L/∂B_r  = (∂L/∂(α·ΔW))^T · A_r            [D×r]
void OnlineTrainer::compute_lowrank_grads(const VectorXf& delta2,
                                           int r_t,
                                           MatrixXf& grad_A,
                                           MatrixXf& grad_B) const {
    const VectorXf& h1    = model_.cached_h1();
    float           alpha = model_.last_alpha();

    // ∂L/∂(αΔW) = α_t · δ2 · h1^T    [D×D]
    MatrixXf dL_dDeltaW = alpha * delta2 * h1.transpose();

    MatrixXf B_r = model_.B().leftCols(r_t);   // [D×r]
    MatrixXf A_r = model_.A().leftCols(r_t);   // [D×r]

    // grad w.r.t. full A and B (only leading r_t columns are non-zero)
    grad_A = MatrixXf::Zero(D, R_MAX);
    grad_B = MatrixXf::Zero(D, R_MAX);

    grad_A.leftCols(r_t) = dL_dDeltaW * B_r;           // [D×r]
    grad_B.leftCols(r_t) = dL_dDeltaW.transpose() * A_r; // [D×r]
}

float OnlineTrainer::step(const RewardShaper::LossTerms& lt,
                           const VectorXf& action_now,
                           const VectorXf& action_prev) {
    // ── ∂L/∂a_t  (policy-gradient approximation) ──────────────────────
    // For the one-step PG update we use:
    //   ∂(smooth_term)/∂a_t = 2·λ2·(a_t − a_{t-1})
    //   ∂(stab_term)/∂a_t   ≈ 0  (φ is a scalar non-differentiable proxy)
    //   ∂(reward_term)/∂a_t ≈ −λ1 · ∇_a r_t  ≈ 0 (reward treated as scalar)
    // → dominant gradient from smoothness regulariser; reward shifts sign.
    VectorXf dL_da = 2.0f * LAMBDA_SMOOTH * (action_now - action_prev);
    // Flip sign from reward: actions that increased reward should be reinforced
    // (equivalent to REINFORCE with baseline = 0 for the one-step case)
    dL_da -= LAMBDA_REWARD * action_now.normalized() * std::abs(lt.reward_term);

    // ── Backprop through output layer ──────────────────────────────────
    VectorXf dL_dh2 = backprop_output_layer(dL_da);    // [D]

    // ── Backprop through adaptive hidden layer ─────────────────────────
    VectorXf delta2 = backprop_hidden2(dL_dh2);         // [D]  pre-act grad

    // ── Compute ∂L/∂A_t, ∂L/∂B_t ─────────────────────────────────────
    MatrixXf grad_A, grad_B;
    compute_lowrank_grads(delta2, model_.last_rank(), grad_A, grad_B);

    // ── Gradient descent update (Eq. 13) ──────────────────────────────
    //   A_{t+1} = A_t − η · ∂L/∂A_t
    //   B_{t+1} = B_t − η · ∂L/∂B_t
    model_.A() -= lr_ * grad_A;
    model_.B() -= lr_ * grad_B;

    // ── Enforce Lyapunov stability bound ──────────────────────────────
    model_.clamp_residual();

    // ── Log gradient norms ────────────────────────────────────────────
    grad_stats_.grad_norm_A         = grad_A.norm();
    grad_stats_.grad_norm_B         = grad_B.norm();
    grad_stats_.delta_w_frobenius   =
        (model_.A().leftCols(model_.last_rank()) *
         model_.B().leftCols(model_.last_rank()).transpose()).norm();

    return lt.total;
}

VectorXf OnlineTrainer::update_history(const VectorXf& h_prev,
                                        const VectorXf& s_t,
                                        float alpha) {
    // h_t = α · h_{t-1} + (1−α) · s_t[:HISTORY_DIM]
    // Use the first HISTORY_DIM elements of s_t as the compression input
    VectorXf s_slice = s_t.head(HISTORY_DIM);
    return ema_update(h_prev, s_slice, alpha);
}

} // namespace olive
