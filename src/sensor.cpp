#include "olive/sensor.hpp"
#include <cmath>
#include <numeric>

namespace olive {

SensorProcessor::SensorProcessor()
    : stats_(STATE_DIM) {}

VectorXf SensorProcessor::build_state(const IMUData&       imu,
                                       const JointData&     joints,
                                       const EMGData&       emg,
                                       const VibrationData& vib,
                                       Intent               intent,
                                       const VectorXf&      history) const {
    // s_t = [x_imu(12) | x_joint(4) | x_emg(8) | x_vib(2) | x_ctx(4) | h(16)]
    VectorXf s(STATE_DIM);
    int offset = 0;

    // (i) IMU: accel + gyro, bilateral — concatenate raw values
    s(offset++)  = imu.accel_l[0]; s(offset++) = imu.accel_l[1]; s(offset++) = imu.accel_l[2];
    s(offset++)  = imu.gyro_l[0];  s(offset++) = imu.gyro_l[1];  s(offset++) = imu.gyro_l[2];
    s(offset++)  = imu.accel_r[0]; s(offset++) = imu.accel_r[1]; s(offset++) = imu.accel_r[2];
    s(offset++)  = imu.gyro_r[0];  s(offset++) = imu.gyro_r[1];  s(offset++) = imu.gyro_r[2];

    // (ii) Joint encoders: bilateral hip angle + velocity
    s(offset++) = joints.angle_l;
    s(offset++) = joints.vel_l;
    s(offset++) = joints.angle_r;
    s(offset++) = joints.vel_r;

    // (iii) Surface EMG: 8-channel bilateral, already rectified + smoothed
    for (int i = 0; i < EMG_DIM; ++i)
        s(offset++) = emg.ch[i];

    // (iv) Vibration: bilateral actuator force → ground-contact vibration
    s(offset++) = vib.left;
    s(offset++) = vib.right;

    // (v) Context: one-hot intent encoding
    VectorXf ctx = encode_intent(intent);
    for (int i = 0; i < CTX_DIM; ++i)
        s(offset++) = ctx(i);

    // (vi) EMA motion history h_{t-1}
    for (int i = 0; i < HISTORY_DIM; ++i)
        s(offset++) = history(i);

    return s;
}

float SensorProcessor::emg_mean_activation(const EMGData& emg) const {
    float sum = 0.0f;
    for (int i = 0; i < EMG_DIM; ++i) sum += emg.ch[i];
    return sum / static_cast<float>(EMG_DIM);
}

float SensorProcessor::metabolic_effort(const IMUData& imu,
                                         const JointData& joints) const {
    // Proxy E_t ∈ [0,1]:
    // Component 1: IMU acceleration variance across bilateral sensors
    float acc_norm_l = std::sqrt(imu.accel_l[0]*imu.accel_l[0] +
                                  imu.accel_l[1]*imu.accel_l[1] +
                                  imu.accel_l[2]*imu.accel_l[2]);
    float acc_norm_r = std::sqrt(imu.accel_r[0]*imu.accel_r[0] +
                                  imu.accel_r[1]*imu.accel_r[1] +
                                  imu.accel_r[2]*imu.accel_r[2]);
    // Normalise against gravity (9.81 m/s²) as reference
    float imu_effort = 0.5f * ((acc_norm_l + acc_norm_r) / (2.0f * 9.81f));
    imu_effort = std::min(1.0f, imu_effort);

    // Component 2: contralateral load asymmetry
    float angle_diff = std::abs(joints.angle_l - joints.angle_r);
    float vel_diff   = std::abs(joints.vel_l   - joints.vel_r);
    // Normalise: max expected asymmetry ~0.3 rad, ~1.0 rad/s
    float asymmetry = std::min(1.0f, 0.5f * (angle_diff / 0.3f + vel_diff / 1.0f));

    return 0.5f * imu_effort + 0.5f * asymmetry;
}

void SensorProcessor::update_stats(const VectorXf& raw_state) {
    stats_.update(raw_state);
}

VectorXf SensorProcessor::encode_intent(Intent intent) const {
    VectorXf ctx = VectorXf::Zero(CTX_DIM);
    ctx(static_cast<int>(intent)) = 1.0f;
    return ctx;
}

} // namespace olive
