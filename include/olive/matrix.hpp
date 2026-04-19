#pragma once
// Thin wrappers around Eigen types used throughout OLIVE.
// Using Eigen because it auto-generates ARM NEON intrinsics when
// compiled with -march=native on the Vigx Ant-H1 Pro SoC.

#include <Eigen/Dense>
#include <cmath>

namespace olive {

// Common aliases
using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;
using RowVectorXf = Eigen::RowVectorXf;

// Fixed-size aliases for hot paths (avoids heap allocation)
template<int Rows, int Cols>
using MatF = Eigen::Matrix<float, Rows, Cols>;

template<int N>
using VecF = Eigen::Matrix<float, N, 1>;

// ── Activations ──────────────────────────────────────────────────────
inline VectorXf relu(const VectorXf& x) {
    return x.cwiseMax(0.0f);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline VectorXf sigmoid(const VectorXf& x) {
    return x.unaryExpr([](float v){ return 1.0f / (1.0f + std::exp(-v)); });
}

inline VectorXf tanh_act(const VectorXf& x) {
    return x.unaryExpr([](float v){ return std::tanh(v); });
}

// ── Normalisation ────────────────────────────────────────────────────
inline VectorXf normalize_minmax(const VectorXf& x, float lo, float hi) {
    float range = hi - lo;
    if (range < 1e-8f) return VectorXf::Zero(x.size());
    return (x.array() - lo) / range;
}

inline VectorXf normalize_zscore(const VectorXf& x, const VectorXf& mean,
                                  const VectorXf& std_dev) {
    return (x - mean).cwiseQuotient(
        std_dev.cwiseMax(1e-8f)
    );
}

// ── EMA update ───────────────────────────────────────────────────────
inline VectorXf ema_update(const VectorXf& prev, const VectorXf& cur,
                            float alpha) {
    return alpha * prev + (1.0f - alpha) * cur;
}

// ── Kaiming uniform init for weight matrices ─────────────────────────
// Appropriate for ReLU networks; keeps gradient scale stable on ARM.
inline MatrixXf kaiming_uniform(int rows, int cols) {
    float gain = std::sqrt(2.0f / static_cast<float>(cols));
    return MatrixXf::Random(rows, cols) * gain;
}

inline VectorXf zeros_bias(int n) {
    return VectorXf::Zero(n);
}

// ── Frobenius norm clamp ─────────────────────────────────────────────
inline MatrixXf clamp_frobenius(const MatrixXf& M, float max_norm) {
    float fn = M.norm();
    if (fn > max_norm) return M * (max_norm / fn);
    return M;
}

} // namespace olive
