#pragma once
// Reward shaping (paper §3.5, Eq. 9)
//
//  r_t = w1·ΔEMG_t  +  w2·(1 − E_t)  +  w3·(1 − ‖φ(s_t, a_t)‖₂)
//
//  ΔEMG_t = EMG_{t-1} − EMG_t          step-over-step muscle-activation drop
//  E_t    ∈ [0,1]                       normalised metabolic-effort proxy
//  φ(s_t, a_t) ∈ [0,1]                 CoM deviation + bilateral step asymmetry
//
// All components are dimensionless and bounded in [0,1] before weighting,
// so r_t ∈ [−(w2+w3), w1+w2+w3].
// Requires NO offline reference trajectories — computed purely from
// on-body sensors (EMG, IMU, joint encoders).

#include "config.hpp"
#include "matrix.hpp"
#include "sensor.hpp"

namespace olive {

// ── Stability signal φ(s_t, a_t) ─────────────────────────────────────────
// Two-component postural stability metric (normalised to [0,1]):
//   φ₁ = CoM deviation  — estimated from bilateral hip angle asymmetry
//   φ₂ = step asymmetry — |left_torque − right_torque| / (|left| + |right| + ε)
// φ = 0.5·φ₁ + 0.5·φ₂
float stability_phi(const VectorXf& state, const VectorXf& action);

// ── RewardShaper ─────────────────────────────────────────────────────────
class RewardShaper {
public:
    explicit RewardShaper();

    // Compute shaped reward r_t (Eq. 9).
    // Must call this AFTER action a_t is executed and sensors are updated.
    float compute(const EMGData&       emg_now,
                  const IMUData&       imu_now,
                  const JointData&     joints_now,
                  const VectorXf&      state,
                  const VectorXf&      action);

    // Loss components for trainer (Eq. 10, 11, 12)
    struct LossTerms {
        float reward_term;     // -λ1 · r_t
        float smooth_term;     // +λ2 · ‖a_t − a_{t-1}‖²
        float stab_term;       // +λ3 · ‖φ(s_t, a_t)‖²
        float total;
    };

    LossTerms compute_loss(float r_t,
                           const VectorXf& action_now,
                           const VectorXf& action_prev,
                           const VectorXf& state,
                           const VectorXf& action) const;

private:
    float prev_emg_mean_   = 0.0f;  // EMG_{t-1} for ΔEMG computation
    float emg_min_         = 0.0f;
    float emg_max_         = 1.0f;
    bool  first_step_      = true;

    // Metabolic effort proxy E_t:
    // E_t = 0.5 · var_norm(IMU_accel) + 0.5 · load_asymmetry(joints)
    float metabolic_effort(const IMUData& imu, const JointData& joints) const;

    // Running variance estimate for IMU acceleration norm
    float imu_var_ema_     = 0.0f;
    float imu_var_alpha_   = 0.05f;  // fast EMA for variance tracking
};

} // namespace olive
