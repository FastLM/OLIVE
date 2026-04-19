#include "olive/config.hpp"
#include "olive/sensor.hpp"
#include "olive/intent.hpp"
#include "olive/model.hpp"
#include "olive/reward.hpp"
#include "olive/trainer.hpp"

#include <chrono>
#include <thread>

using namespace olive;
#include <cstdio>
#include <cstring>
#include <cmath>
#include <csignal>
#include <atomic>

// ── Global shutdown flag ──────────────────────────────────────────────────
static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running = false; }

// ── Simulated hardware read (replace with real HAL calls on Ant-H1 Pro) ───
// In production these functions call the ARM SoC peripheral drivers:
//   - IMU  : SPI @ 1000 Hz via DMA ring buffer
//   - EMG  : USB HID via Delsys Trigno base station
//   - Vib  : I²C force sensor ADC
//   - Joint: CAN encoder frames

static void read_imu(olive::IMUData& d) {
    // placeholder: simulate gentle walking — ~1g + small perturbations
    static float t = 0.0f;  t += 0.01f;
    float w = 2.0f * 3.14159f * 1.0f;  // 1 Hz gait cycle
    d.accel_l[0] =  0.1f * std::sin(w*t); d.accel_l[1] = 9.81f; d.accel_l[2] = 0.05f * std::cos(w*t);
    d.gyro_l[0]  =  0.2f * std::sin(w*t); d.gyro_l[1]  = 0.0f;  d.gyro_l[2]  = 0.1f;
    d.accel_r[0] = -0.1f * std::sin(w*t); d.accel_r[1] = 9.81f; d.accel_r[2] = 0.05f * std::cos(w*t + 3.14159f);
    d.gyro_r[0]  = -0.2f * std::sin(w*t); d.gyro_r[1]  = 0.0f;  d.gyro_r[2]  = 0.1f;
}

static void read_joints(olive::JointData& d) {
    static float t = 0.0f;  t += 0.01f;
    float w = 2.0f * 3.14159f * 1.0f;
    d.angle_l =  0.3f * std::sin(w*t);
    d.vel_l   =  0.3f * w * std::cos(w*t);
    d.angle_r = -0.3f * std::sin(w*t);
    d.vel_r   = -0.3f * w * std::cos(w*t);
}

static void read_emg(olive::EMGData& d) {
    static float t = 0.0f;  t += 0.01f;
    float base = 50.0f;  // μV baseline
    for (int i = 0; i < olive::EMG_DIM; ++i)
        d.ch[i] = base + 10.0f * std::abs(std::sin(2.0f * 3.14159f * 1.0f * t + i * 0.5f));
}

static void read_vibration(olive::VibrationData& d) {
    static float t = 0.0f;  t += 0.01f;
    d.left  = 5.0f + 2.0f * std::abs(std::sin(2.0f * 3.14159f * 1.0f * t));
    d.right = 5.0f + 2.0f * std::abs(std::sin(2.0f * 3.14159f * 1.0f * t + 3.14159f));
}

static void send_torques(const olive::VectorXf& action) {
    // In production: write to CAN bus → bilateral hip actuator drivers
    (void)action;
}

// ── Logging ───────────────────────────────────────────────────────────────
static void log_step(int step, float reward, int rank, float alpha,
                     double latency_ms, const olive::VectorXf& action) {
    if (step % 100 == 0) {
        std::printf("[step %5d] r=%.4f  r_t=%2d  α=%.3f  a=[%.2f, %.2f]  lat=%.2f ms\n",
                    step, reward, rank, alpha, action(0), action(1), latency_ms);
        std::fflush(stdout);
    }
}

// ═════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    const char* weights_path = (argc > 1) ? argv[1] : nullptr;

    // ── Init subsystems ───────────────────────────────────────────────
    olive::SensorProcessor sensor_proc;
    olive::IntentRecognizer intent_net;
    olive::OLIVEModel       model;
    olive::RewardShaper     reward_shaper;
    olive::OnlineTrainer    trainer(model, LEARNING_RATE);

    if (weights_path) {
        if (!model.load_base_weights(weights_path))
            std::fprintf(stderr, "Warning: could not load %s — using random W0\n", weights_path);
        else
            std::printf("Loaded base weights from %s\n", weights_path);
    }

    // ── State ─────────────────────────────────────────────────────────
    olive::VectorXf history    = olive::VectorXf::Zero(HISTORY_DIM);
    olive::VectorXf action_prev= olive::VectorXf::Zero(ACTION_DIM);

    std::printf("OLIVE online deployment started — target latency <%.0f ms\n",
                LATENCY_BUDGET_MS);
    std::printf("State dim: %d,  Policy: %d→%d→%d→%d,  r∈[%d,%d]\n",
                STATE_DIM, STATE_DIM, D, D, ACTION_DIM, R_MIN, R_MAX);

    int step = 0;

    // ── Control loop @ 100 Hz ─────────────────────────────────────────
    auto next_tick = std::chrono::steady_clock::now();
    const auto tick_period = std::chrono::microseconds(
        static_cast<long>(1e6 / CONTROL_FREQ_HZ));

    while (g_running) {
        auto t_start = std::chrono::steady_clock::now();

        // ── 1. Read sensors ───────────────────────────────────────────
        olive::IMUData       imu;
        olive::JointData     joints;
        olive::EMGData       emg;
        olive::VibrationData vib;
        read_imu(imu);
        read_joints(joints);
        read_emg(emg);
        read_vibration(vib);

        // ── 2. Intent recognition (<2 ms)  ───────────────────────────
        // Build temp state for classification (history not yet updated)
        olive::VectorXf s_tmp = sensor_proc.build_state(
            imu, joints, emg, vib, olive::Intent::Walk, history);
        olive::Intent intent = intent_net.classify(s_tmp);

        // ── 3. Assemble full state vector s_t ─────────────────────────
        olive::VectorXf state = sensor_proc.build_state(
            imu, joints, emg, vib, intent, history);

        // ── 4. OLIVE forward pass: select r_t, compute α_t, output a_t ─
        olive::VectorXf action = model.forward(state);

        // ── 5. Execute torque command ─────────────────────────────────
        send_torques(action);

        // ── 6. Compute shaped reward from on-body sensor feedback ─────
        float r_t = reward_shaper.compute(emg, imu, joints, state, action);

        // ── 7. Compute loss terms ─────────────────────────────────────
        auto loss = reward_shaper.compute_loss(
            r_t, action, action_prev, state, action);

        // ── 8. Online gradient update on A_t, B_t ────────────────────
        trainer.step(loss, action, action_prev);

        // ── 9. Update EMA history ─────────────────────────────────────
        history     = olive::OnlineTrainer::update_history(history, state);
        action_prev = action;

        // ── 10. Latency check ─────────────────────────────────────────
        auto t_end = std::chrono::steady_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            t_end - t_start).count();

        if (latency_ms > LATENCY_BUDGET_MS)
            std::fprintf(stderr, "[WARN] step %d: latency %.2f ms > budget %.0f ms\n",
                         step, latency_ms, LATENCY_BUDGET_MS);

        log_step(step, r_t, model.last_rank(), model.last_alpha(),
                 latency_ms, action);
        ++step;

        // ── 11. Enforce real-time schedule ────────────────────────────
        next_tick += tick_period;
        std::this_thread::sleep_until(next_tick);
    }

    std::printf("\nShutdown after %d steps. Goodbye.\n", step);
    return 0;
}
