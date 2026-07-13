#pragma once

#include "hsmm/types.hpp"

namespace hsmm {

// Result of running the explicit-duration forward-backward algorithm
// (Yu, 2010 "Hidden semi-Markov models" formulation).
struct ForwardBackwardResult {
    Matrix log_alpha;          // T x K, forward log-probabilities
    Matrix log_beta;           // T x K, backward log-probabilities
    Matrix log_gamma;          // T x K, posterior state marginals log P(z_t = k | y_1:T)
    double log_likelihood{0.0}; // log P(y_1:T | theta)
};

// Runs the forward-backward recursion for an HSMM with explicit state
// durations. `observations` is length T (univariate for now; extend to
// Matrix for multivariate emissions).
//
// This is a skeleton declaration: implementation in forward_backward.cpp
// currently contains TODOs for the duration-convolution step.
ForwardBackwardResult forward_backward(const Vector& observations,
                                        const HSMMParams& params);

}  // namespace hsmm
