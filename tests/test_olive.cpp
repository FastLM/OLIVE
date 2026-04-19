#include "olive/config.hpp"
#include "olive/matrix.hpp"
#include "olive/sensor.hpp"
#include "olive/intent.hpp"
#include "olive/model.hpp"
#include "olive/reward.hpp"
#include "olive/trainer.hpp"

#include <cstdio>
#include <cassert>
#include <cmath>

using namespace olive;

// Simple assertion helper
#define ASSERT_NEAR(a, b, tol) \
    do { if (std::abs((a)-(b)) > (tol)) { \
        std::fprintf(stderr, "FAIL %s:%d: |%f - %f| > %f\n", \
                     __FILE__, __LINE__, (double)(a), (double)(b), (double)(tol)); \
        return 1; } } while(0)

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: " #cond "\n", __FILE__, __LINE__); \
        return 1; } } while(0)

// ── Test 1: State vector dimension ───────────────────────────────────────
int test_state_dim() {
    olive::SensorProcessor proc;
    olive::IMUData imu{}; olive::JointData joints{};
    olive::EMGData emg{}; olive::VibrationData vib{};
    olive::VectorXf history = olive::VectorXf::Zero(HISTORY_DIM);

    auto s = proc.build_state(imu, joints, emg, vib, olive::Intent::Walk, history);
    ASSERT_TRUE(s.size() == STATE_DIM);
    std::printf("PASS test_state_dim: s.size() = %ld (expected %d)\n",
                s.size(), STATE_DIM);
    return 0;
}

// ── Test 2: Intent one-hot encoding ──────────────────────────────────────
int test_intent_onehot() {
    olive::SensorProcessor proc;
    olive::IMUData imu{}; olive::JointData joints{};
    olive::EMGData emg{}; olive::VibrationData vib{};
    olive::VectorXf history = olive::VectorXf::Zero(HISTORY_DIM);

    auto s_walk  = proc.build_state(imu, joints, emg, vib, olive::Intent::Walk,  history);
    auto s_climb = proc.build_state(imu, joints, emg, vib, olive::Intent::Climb, history);

    int ctx_offset = IMU_DIM + JOINT_DIM + EMG_DIM + VIB_DIM;
    ASSERT_NEAR(s_walk(ctx_offset + 0),  1.0f, 1e-6f);  // walk = idx 0
    ASSERT_NEAR(s_climb(ctx_offset + 1), 1.0f, 1e-6f);  // climb = idx 1
    std::printf("PASS test_intent_onehot\n");
    return 0;
}

// ── Test 3: Model forward shape ───────────────────────────────────────────
int test_model_forward_shape() {
    olive::OLIVEModel model;
    olive::VectorXf state = olive::VectorXf::Random(STATE_DIM);

    auto action = model.forward(state);
    ASSERT_TRUE(action.size() == ACTION_DIM);

    // Torques must be within physical clamp bounds
    ASSERT_TRUE(action(0) >= TORQUE_MIN && action(0) <= TORQUE_MAX);
    ASSERT_TRUE(action(1) >= TORQUE_MIN && action(1) <= TORQUE_MAX);
    std::printf("PASS test_model_forward_shape: action=[%.3f, %.3f]\n",
                action(0), action(1));
    return 0;
}

// ── Test 4: Gating scalar α_t ∈ (0,1) ────────────────────────────────────
int test_gating_range() {
    olive::OLIVEModel model;
    for (int i = 0; i < 20; ++i) {
        olive::VectorXf state = olive::VectorXf::Random(STATE_DIM);
        model.forward(state);
        float alpha = model.last_alpha();
        ASSERT_TRUE(alpha > 0.0f && alpha < 1.0f);
    }
    std::printf("PASS test_gating_range\n");
    return 0;
}

// ── Test 5: Dynamic rank selection ────────────────────────────────────────
int test_rank_selection() {
    olive::OLIVEModel model;
    for (int i = 0; i < 50; ++i) {
        olive::VectorXf state = olive::VectorXf::Random(STATE_DIM);
        model.forward(state);
        int r = model.last_rank();
        ASSERT_TRUE(r >= R_MIN && r <= R_MAX);
    }
    std::printf("PASS test_rank_selection: r ∈ [%d, %d]\n", R_MIN, R_MAX);
    return 0;
}

// ── Test 6: Reward bounded ────────────────────────────────────────────────
int test_reward_bounds() {
    olive::RewardShaper rs;
    olive::VectorXf state  = olive::VectorXf::Zero(STATE_DIM);
    olive::VectorXf action = olive::VectorXf::Zero(ACTION_DIM);

    olive::IMUData imu{};
    // Simulate flat walking: left/right accel near g
    imu.accel_l[1] = 9.81f; imu.accel_r[1] = 9.81f;

    olive::EMGData emg{};
    for (int i = 0; i < EMG_DIM; ++i) emg.ch[i] = 50.0f;

    olive::JointData joints{};
    olive::VibrationData vib{};

    float r_t = rs.compute(emg, imu, joints, state, action);

    float r_max = W_EMG + W_EFFORT + W_STABILITY;
    ASSERT_TRUE(r_t >= -r_max - 0.01f && r_t <= r_max + 0.01f);
    std::printf("PASS test_reward_bounds: r_t = %.4f (max=%.2f)\n", r_t, r_max);
    return 0;
}

// ── Test 7: Online update reduces loss over N steps ───────────────────────
int test_online_convergence() {
    olive::OLIVEModel   model;
    olive::RewardShaper rs;
    olive::OnlineTrainer trainer(model, LEARNING_RATE);

    olive::VectorXf history     = olive::VectorXf::Zero(HISTORY_DIM);
    olive::VectorXf action_prev = olive::VectorXf::Zero(ACTION_DIM);
    olive::VectorXf state       = olive::VectorXf::Zero(STATE_DIM);
    // Simulate slight bilateral symmetry (flat walking)
    state(IMU_DIM + 1) = 9.81f;   // accel_l[1]

    float first_loss = 0.0f, last_loss = 0.0f;
    const int N = 200;

    for (int i = 0; i < N; ++i) {
        auto action = model.forward(state);

        olive::EMGData emg{};
        for (int c = 0; c < EMG_DIM; ++c) emg.ch[c] = 50.0f * std::exp(-0.001f * i);

        olive::IMUData imu{};
        imu.accel_l[1] = 9.81f; imu.accel_r[1] = 9.81f;
        olive::JointData joints{};

        float r_t  = rs.compute(emg, imu, joints, state, action);
        auto  loss = rs.compute_loss(r_t, action, action_prev, state, action);
        float lv   = trainer.step(loss, action, action_prev);

        history     = olive::OnlineTrainer::update_history(history, state);
        action_prev = action;
        if (i == 0)   first_loss = lv;
        if (i == N-1) last_loss  = lv;
    }

    std::printf("PASS test_online_convergence: loss %.4f → %.4f over %d steps\n",
                first_loss, last_loss, N);
    return 0;
}

// ── Test 8: Stability clamp (Lyapunov bound) ─────────────────────────────
int test_stability_clamp() {
    olive::OLIVEModel model;
    // Inflate A, B beyond the allowed norm
    model.A() = olive::MatrixXf::Ones(D, R_MAX) * 100.0f;
    model.B() = olive::MatrixXf::Ones(D, R_MAX) * 100.0f;

    model.clamp_residual();

    ASSERT_TRUE(model.A().norm() <= DELTA_W_MAX + 1e-4f);
    ASSERT_TRUE(model.B().norm() <= DELTA_W_MAX + 1e-4f);
    std::printf("PASS test_stability_clamp: ‖A‖=%.4f  ‖B‖=%.4f\n",
                model.A().norm(), model.B().norm());
    return 0;
}

// ── Test 9: History EMA shape ─────────────────────────────────────────────
int test_history_ema() {
    olive::VectorXf h = olive::VectorXf::Zero(HISTORY_DIM);
    olive::VectorXf s = olive::VectorXf::Ones(STATE_DIM);
    auto h_new = olive::OnlineTrainer::update_history(h, s);
    ASSERT_TRUE(h_new.size() == HISTORY_DIM);
    // After one step from zero, all elements should equal (1−α)
    float expected = (1.0f - EMA_ALPHA) * 1.0f;
    ASSERT_NEAR(h_new(0), expected, 1e-5f);
    std::printf("PASS test_history_ema: h_new[0]=%.5f (expected %.5f)\n",
                h_new(0), expected);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== OLIVE Unit Tests ===\n\n");
    int failures = 0;
    failures += test_state_dim();
    failures += test_intent_onehot();
    failures += test_model_forward_shape();
    failures += test_gating_range();
    failures += test_rank_selection();
    failures += test_reward_bounds();
    failures += test_online_convergence();
    failures += test_stability_clamp();
    failures += test_history_ema();

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures;
}
