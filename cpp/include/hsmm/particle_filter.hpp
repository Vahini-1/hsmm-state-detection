#pragma once

#include "hsmm/types.hpp"
#include <random>

namespace hsmm {

// A single particle in the SMC filter: hypothesized current regime,
// time-in-state so far (needed for semi-Markov duration bookkeeping),
// and an importance weight.
struct Particle {
    std::size_t state{0};
    std::size_t time_in_state{0};
    double log_weight{0.0};
};

struct ParticleFilterConfig {
    std::size_t n_particles{2000};
    double ess_resample_threshold{0.5}; // resample when ESS / N < threshold
    unsigned int seed{42};
};

// Online particle filter for HSMM regime tracking. Call `step()` once per
// new observation as it arrives; the filter maintains its own particle
// population internally between calls, which is what makes it suitable
// for live/streaming use (millisecond-scale updates once compiled).
class ParticleFilter {
public:
    ParticleFilter(const HSMMParams& params, const ParticleFilterConfig& config = {});

    // Advance the filter by one observation. Returns the current posterior
    // distribution over regimes, P(z_t = k | y_1:t), as a length-K vector.
    Vector step(double y_t);

    // Diagnostics
    double effective_sample_size() const;
    const std::vector<Particle>& particles() const { return particles_; }

private:
    HSMMParams params_;
    ParticleFilterConfig config_;
    std::vector<Particle> particles_;
    std::mt19937 rng_;

    void initialize_particles();
    void resample_if_needed();
    void systematic_resample();
};

}  // namespace hsmm
