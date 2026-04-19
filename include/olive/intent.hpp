#pragma once
// Intent Recognizer (paper §2)
// Lightweight MLP: maps current sensor window → discrete intent u_t
//   u_t ∈ {walk, climb, slope, uneven}
// Runs in <2 ms on embedded ARM CPU (single hidden layer, 64 units).
// Weights are pretrained offline from population data and FROZEN during
// deployment (only OLIVE's low-rank factors are updated online).

#include "config.hpp"
#include "matrix.hpp"
#include "sensor.hpp"

namespace olive {

// ── IntentRecognizer ──────────────────────────────────────────────────────
// Architecture:  Linear(STATE_DIM, 64) → ReLU → Linear(64, 4) → argmax
// Input window:  single fused state vector s_t (no temporal unrolling —
//                history is already encoded in h_{t-1} via EMA).
class IntentRecognizer {
public:
    explicit IntentRecognizer();

    // Load pretrained weights from a flat binary file written row-major.
    // File layout: [W1: 64×STATE_DIM | b1: 64 | W2: 4×64 | b2: 4]
    bool load_weights(const char* path);

    // Forward pass → intent label (argmax of softmax logits)
    Intent classify(const VectorXf& state) const;

    // Raw logits before softmax (useful for confidence thresholding)
    VectorXf logits(const VectorXf& state) const;

private:
    MatrixXf W1_;   // [64 × STATE_DIM]
    VectorXf b1_;   // [64]
    MatrixXf W2_;   // [4  × 64]
    VectorXf b2_;   // [4]

    bool weights_loaded_ = false;
};

} // namespace olive
