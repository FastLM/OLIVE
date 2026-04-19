#pragma once
// Online Incremental Update
//
// One-step proximal policy gradient restricted to the low-rank subspace:
//
//   L_t = −λ1·r_t  +  λ2·‖a_t − a_{t-1}‖²  +  λ3·‖φ(s_t,a_t)‖²
//
// Gradients w.r.t. low-rank factors A_t, B_t via chain rule through
// the adaptive middle layer:
//
//   ∂L/∂A_t  =  ∂L/∂h2  ·  (h1·B_t[:,:r]^T)^T  (leading r cols)
//   ∂L/∂B_t  =  ∂L/∂h2  ·  (h1·A_t[:,:r]^T)^T
//
// Update rule:
//   A_{t+1} = A_t − η · ∇_{A_t} L_t
//   B_{t+1} = B_t − η · ∇_{B_t} L_t
//
// W0 is held fixed throughout deployment.

#include "config.hpp"
#include "matrix.hpp"
#include "model.hpp"
#include "reward.hpp"

namespace olive {

class OnlineTrainer {
public:
    explicit OnlineTrainer(OLIVEModel& model, float lr = LEARNING_RATE);

    // Execute one online update step.
    // Returns the total loss value for monitoring.
    float step(const RewardShaper::LossTerms& loss_terms,
               const VectorXf& action_now,
               const VectorXf& action_prev);

    // EMA history update  h_t = EMA(h_{t-1}, s_t)
    static VectorXf update_history(const VectorXf& h_prev,
                                   const VectorXf& s_t,
                                   float alpha = EMA_ALPHA);

    // Gradient statistics for logging / debugging
    struct GradStats {
        float grad_norm_A;
        float grad_norm_B;
        float delta_w_frobenius;
    };
    GradStats last_grad_stats() const { return grad_stats_; }

private:
    OLIVEModel& model_;
    float       lr_;
    GradStats   grad_stats_{};

    // Backprop through the adaptive middle layer.
    // Returns ∂L/∂(pre-activation of h2) given scalar dL/d(output).
    VectorXf backprop_output_layer(const VectorXf& dL_da) const;
    VectorXf backprop_hidden2(const VectorXf& dL_dh2_pre) const;

    // Compute ∂L/∂A_t and ∂L/∂B_t for the active rank slice [:, :r_t]
    void compute_lowrank_grads(const VectorXf& delta_h2,
                               int r_t,
                               MatrixXf& grad_A,
                               MatrixXf& grad_B) const;
};

} // namespace olive
