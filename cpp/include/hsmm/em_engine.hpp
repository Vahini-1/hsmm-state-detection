#pragma once

#include "hsmm/types.hpp"
#include "hsmm/forward_backward.hpp"

namespace hsmm {

struct EMConfig {
    std::size_t max_iters{200};
    double tol{1e-6};          // convergence threshold on log-likelihood delta
    unsigned int seed{42};
    bool verbose{true};
};

struct EMResult {
    HSMMParams params;
    std::vector<double> log_likelihood_history;
    bool converged{false};
};

// Runs Baum-Welch-style EM adapted for explicit-duration HSMMs.
// `init_params` supplies the initial guess (e.g. from k-means on
// realized volatility, or random restarts handled by the caller).
EMResult fit_hsmm_em(const Vector& observations,
                      const HSMMParams& init_params,
                      const EMConfig& config = {});

// Viterbi-style most likely state (and duration) path given fitted params.
// Returns a vector of length T with the most probable regime label per step.
std::vector<std::size_t> most_likely_state_path(const Vector& observations,
                                                  const HSMMParams& params);

}  // namespace hsmm
