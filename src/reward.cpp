#include "olive/reward.hpp"
#include <cmath>
#include <algorithm>

namespace olive {

// ── Stability metric φ(s_t, a_t) ─────────────────────────────────────────
// φ₁ = CoM deviation estimated from bilateral hip angle asymmetry
// φ₂ = bilateral torque asymmetry
float stability_phi(const VectorXf& state, const VectorXf& action) {
    // Extract bilateral hip angles from state: indices 12, 14 (angle_l, angle_r)
    float angle_l = state(IMU_DIM);                   // offset 12
    float angle_r = state(IMU_DIM + JOINT_DIM / 2);   // offset 14

    // φ₁: CoM deviation ≈ normalised bilateral angle difference
    float angle_asym = std::abs(angle_l - angle_r);
    float phi1 = std::min(1.0f, angle_asym / 0.5f);   // 0.5 rad saturation

    // φ₂: bilateral torque asymmetry
    float tl = std::abs(action(0));
    float tr = std::abs(action(1));
    float phi2 = (tl + tr > 1e-4f)
                 ? std::abs(tl - tr) / (tl + tr)
                 : 0.0f;

    return 0.5f * phi1 + 0.5f * phi2;
}

// ── RewardShaper ──────────────────────────────────────────────────────────

RewardShaper::RewardShaper() = default;

float RewardShaper::metabolic_effort(const IMUData& imu,
                                      const JointData& joints) const {
    // E_t = 0.5 · IMU_effort + 0.5 · load_asymmetry

    // IMU acceleration variance across bilateral sensors (deviation from g)
    float g = 9.81f;
    auto acc_norm = [](const float a[3]) {
        return std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    };
    float an_l = acc_norm(imu.accel_l);
    float an_r = acc_norm(imu.accel_r);
    float imu_dev = 0.5f * (std::abs(an_l - g) + std::abs(an_r - g));
    float imu_effort = std::min(1.0f, imu_dev / g);   // normalise by g

    // Contralateral load asymmetry
    float angle_diff = std::abs(joints.angle_l - joints.angle_r);
    float vel_diff   = std::abs(joints.vel_l   - joints.vel_r);
    float asymmetry  = std::min(1.0f, 0.5f * (angle_diff / 0.3f + vel_diff / 1.0f));

    return 0.5f * imu_effort + 0.5f * asymmetry;
}

float RewardShaper::compute(const EMGData&    emg_now,
                             const IMUData&    imu_now,
                             const JointData&  joints_now,
                             const VectorXf&   state,
                             const VectorXf&   action) {
    // ── ΔEMG_t = EMG_{t-1} − EMG_t ───────────────────────────────────
    float emg_mean = 0.0f;
    for (int i = 0; i < EMG_DIM; ++i) emg_mean += emg_now.ch[i];
    emg_mean /= static_cast<float>(EMG_DIM);

    // Update running min/max for normalisation
    emg_min_ = std::min(emg_min_, emg_mean);
    emg_max_ = std::max(emg_max_, emg_mean + 1e-8f);

    float emg_norm = (emg_mean - emg_min_) / (emg_max_ - emg_min_ + 1e-8f);

    float delta_emg = 0.0f;
    if (!first_step_) {
        delta_emg = prev_emg_mean_ - emg_norm;  // positive when effort drops
        delta_emg = std::clamp(delta_emg, -1.0f, 1.0f);
    }
    prev_emg_mean_ = emg_norm;
    first_step_ = false;

    // ── E_t: metabolic effort proxy ───────────────────────────────────
    float E_t = metabolic_effort(imu_now, joints_now);

    // Online EMA smoothing of IMU variance
    float acc_n = std::sqrt(imu_now.accel_l[0]*imu_now.accel_l[0] +
                             imu_now.accel_l[1]*imu_now.accel_l[1] +
                             imu_now.accel_l[2]*imu_now.accel_l[2]);
    imu_var_ema_ = (1.0f - imu_var_alpha_) * imu_var_ema_
                 + imu_var_alpha_ * (acc_n - 9.81f) * (acc_n - 9.81f);

    // ── φ(s_t, a_t): stability ────────────────────────────────────────
    float phi = stability_phi(state, action);

    // ── Shaped reward r_t ─────────────────────────────────────────────
    float r_t = W_EMG      * delta_emg
              + W_EFFORT   * (1.0f - E_t)
              + W_STABILITY * (1.0f - phi);

    return r_t;
}

RewardShaper::LossTerms RewardShaper::compute_loss(
        float r_t,
        const VectorXf& action_now,
        const VectorXf& action_prev,
        const VectorXf& state,
        const VectorXf& action) const {

    LossTerms lt;

    // −λ1 · r_t  (reward maximisation)
    lt.reward_term = -LAMBDA_REWARD * r_t;

    // λ2 · ‖a_t − a_{t-1}‖²  (torque smoothness)
    VectorXf da = action_now - action_prev;
    lt.smooth_term = LAMBDA_SMOOTH * da.squaredNorm();

    // λ3 · ‖φ(s_t, a_t)‖²  (postural stability)
    float phi = stability_phi(state, action);
    lt.stab_term = LAMBDA_STAB * phi * phi;

    lt.total = lt.reward_term + lt.smooth_term + lt.stab_term;
    return lt;
}

} // namespace olive
