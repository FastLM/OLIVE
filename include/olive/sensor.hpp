#pragma once
// Sensor layer: four complementary motion modalities (paper §2)
//
//  (i)  Electronic: IMU (accel + gyro @1000 Hz) + bilateral joint encoder
//  (ii) EMG: 8-ch Delsys Trigno surface EMG, bilateral muscle activation
//  (iii)Vibration: ground-contact vibration from hip actuator force sensors
//  (iv) Context: inferred terrain/activity one-hot vector
//
// All signals are simply preprocessed (bandpass, normalise) then
// CONCATENATED into s_t ∈ R^n — no learnable fusion network.

#include "config.hpp"
#include "matrix.hpp"
#include <array>
#include <deque>

namespace olive {

// ── Raw sensor packets ────────────────────────────────────────────────────

struct IMUData {
    // 3-axis accel [m/s²] + 3-axis gyro [rad/s], bilateral (left, right)
    float accel_l[3];   // left  hip IMU acceleration
    float gyro_l[3];    // left  hip IMU angular velocity
    float accel_r[3];   // right hip IMU acceleration
    float gyro_r[3];    // right hip IMU angular velocity
};

struct JointData {
    float angle_l;      // left  hip joint angle   [rad]
    float vel_l;        // left  hip joint velocity [rad/s]
    float angle_r;      // right hip joint angle   [rad]
    float vel_r;        // right hip joint velocity [rad/s]
};

struct EMGData {
    // 8-channel bilateral surface EMG [μV], rectified + smoothed
    float ch[EMG_DIM];
};

struct VibrationData {
    // Bilateral actuator force sensor → ground-contact vibration amplitude
    float left;         // left  hip actuator [N]
    float right;        // right hip actuator [N]
};

// Intent label (output of IntentRecognizer)
enum class Intent : int {
    Walk   = 0,
    Climb  = 1,
    Slope  = 2,
    Uneven = 3
};

// ── Sensor running statistics (for online z-score normalisation) ───────────
// Updated incrementally with Welford's algorithm — O(1) per step.
struct RunningStats {
    VectorXf mean;
    VectorXf M2;        // sum of squared deviations
    long count = 0;

    explicit RunningStats(int dim)
        : mean(VectorXf::Zero(dim)), M2(VectorXf::Zero(dim)) {}

    void update(const VectorXf& x) {
        ++count;
        VectorXf delta = x - mean;
        mean += delta / static_cast<float>(count);
        M2   += delta.cwiseProduct(x - mean);
    }

    VectorXf std_dev() const {
        if (count < 2) return VectorXf::Ones(mean.size());
        return (M2 / static_cast<float>(count - 1)).cwiseSqrt().cwiseMax(1e-8f);
    }

    VectorXf normalize(const VectorXf& x) const {
        return (x - mean).cwiseQuotient(std_dev());
    }
};

// ── SensorProcessor ────────────────────────────────────────────────────────
// Takes raw hardware packets, applies minimal signal conditioning, and
// concatenates into the unified state vector s_t ∈ R^{STATE_DIM}.
class SensorProcessor {
public:
    explicit SensorProcessor();

    // Assemble s_t = [x_imu | x_joint | x_emg | x_vib | x_ctx | h_{t-1}]
    VectorXf build_state(const IMUData&      imu,
                         const JointData&    joints,
                         const EMGData&      emg,
                         const VibrationData& vib,
                         Intent              intent,
                         const VectorXf&     history) const;

    // EMG helper: mean bilateral RMS across all channels → scalar [0,1]
    float emg_mean_activation(const EMGData& emg) const;

    // Metabolic-effort proxy E_t ∈ [0,1]: IMU variance + load asymmetry
    float metabolic_effort(const IMUData& imu, const JointData& joints) const;

    // Update online normalisation stats (call once per control step)
    void update_stats(const VectorXf& raw_state);

private:
    // Per-modality normalisation windows
    std::deque<float> emg_window_;          // sliding window for EMG baseline
    static constexpr int EMG_WIN = 50;      // 50 steps @ 100 Hz = 0.5 s

    RunningStats stats_;

    // One-hot encode intent label → R^4
    VectorXf encode_intent(Intent intent) const;
};

} // namespace olive
