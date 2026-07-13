#pragma once

#include "hsmm/types.hpp"

namespace hsmm {

// log P(d | state k), for d = 1 .. max_duration
double log_duration_pmf(std::size_t d, const DurationParams& params);

// Precomputes log P(d | k) for d = 1..max_duration into a vector for
// efficient reuse inside the forward-backward recursion.
Vector precompute_log_duration_pmf(const DurationParams& params);

// Survivor function: log P(D > d | state k) = log(1 - CDF(d)).
// Needed for the "still in this state" term in the HSMM recursions.
double log_duration_survivor(std::size_t d, const DurationParams& params);

// M-step update: given expected duration sufficient statistics collected
// during EM, refit (r, p) for the Negative-Binomial duration model via
// method-of-moments or Newton's method on r.
DurationParams fit_duration_params(const Vector& expected_duration_counts);

}  // namespace hsmm
