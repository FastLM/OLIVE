#include "olive/intent.hpp"
#include <fstream>
#include <algorithm>
#include <cstring>

namespace olive {

IntentRecognizer::IntentRecognizer() {
    // Default: random initialisation (replace with load_weights() before use)
    W1_ = kaiming_uniform(GATE_HIDDEN, STATE_DIM);
    b1_ = zeros_bias(GATE_HIDDEN);
    W2_ = kaiming_uniform(4, GATE_HIDDEN);
    b2_ = zeros_bias(4);
}

bool IntentRecognizer::load_weights(const char* path) {
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

    read_mat(W1_, GATE_HIDDEN, STATE_DIM);
    read_vec(b1_, GATE_HIDDEN);
    read_mat(W2_, 4, GATE_HIDDEN);
    read_vec(b2_, 4);

    weights_loaded_ = f.good();
    return weights_loaded_;
}

VectorXf IntentRecognizer::logits(const VectorXf& state) const {
    VectorXf h = relu(W1_ * state + b1_);  // [64]
    return W2_ * h + b2_;                   // [4]
}

Intent IntentRecognizer::classify(const VectorXf& state) const {
    VectorXf lg = logits(state);
    int idx;
    lg.maxCoeff(&idx);
    return static_cast<Intent>(idx);
}

} // namespace olive
