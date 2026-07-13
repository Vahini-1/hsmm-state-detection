#pragma once

#include <cstddef>
#include <vector>

namespace hsmm {

// A simple dense matrix wrapper. Kept dependency-light (no Eigen requirement)
// so the skeleton compiles standalone; swap for Eigen::MatrixXd internally
// once Eigen is wired up, the public API below stays stable.
struct Matrix {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<double> data;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double fill = 0.0)
        : rows(r), cols(c), data(r * c, fill) {}

    double& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }
    double operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }
};

using Vector = std::vector<double>;

// Emission parameters for a single regime, Student-t distributed:
// y_t | z_t = k ~ StudentT(mu_k, sigma_k, nu_k)
struct EmissionParams {
    double mu{0.0};       // location
    double sigma{1.0};    // scale
    double nu{5.0};       // degrees of freedom (fat tails)
};

// Duration distribution for a regime. Modeled here as a Negative Binomial
// (a generalization of the Geometric that still allows the HSMM to reduce
// to an HMM as a special case, while supporting non-geometric persistence).
struct DurationParams {
    double r{1.0};      // number of failures parameter
    double p{0.5};      // success probability
    std::size_t max_duration{252}; // truncate support at ~1 trading year
};

// Full HSMM parameter set for K regimes.
struct HSMMParams {
    std::size_t n_states{0};
    Matrix transition;                 // K x K, diagonal should be ~0 (semi-Markov: no self-loop)
    std::vector<EmissionParams> emissions;   // size K
    std::vector<DurationParams> durations;   // size K
    Vector initial_dist;               // size K, pi_0
};

}  // namespace hsmm
