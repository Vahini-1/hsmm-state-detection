#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>

#include "hsmm/particle_filter.hpp"

using namespace hsmm;
using Catch::Matchers::WithinAbs;

namespace {

HSMMParams make_two_state_params() {
    HSMMParams p;
    p.n_states = 2;
    p.transition = Matrix(2, 2, 0.0);
    p.transition(0, 1) = 1.0;
    p.transition(1, 0) = 1.0;
    p.emissions = {EmissionParams{0.0, 0.3, 30.0}, EmissionParams{4.0, 0.3, 30.0}};
    p.durations = {DurationParams{10.0, 0.6, 30}, DurationParams{10.0, 0.6, 30}};
    p.initial_dist = {0.5, 0.5};
    return p;
}

// Generates a synthetic regime-switching series with known ground-truth
// state labels, alternating blocks of a fixed length. Deterministic given
// a seed, so tests can assert on exact/near-exact tracking behavior.
struct SyntheticSeries {
    std::vector<double> observations;
    std::vector<std::size_t> true_states;
};

SyntheticSeries make_synthetic_series(const HSMMParams& params, std::size_t n_blocks,
                                       std::size_t block_len, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.3);

    SyntheticSeries series;
    std::size_t state = 0;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const double mean = params.emissions[state].mu;
        for (std::size_t i = 0; i < block_len; ++i) {
            series.observations.push_back(mean + noise(rng));
            series.true_states.push_back(state);
        }
        state = 1 - state;
    }
    return series;
}

}  // namespace

TEST_CASE("particle_filter: posterior sums to 1 after every step", "[particle_filter]") {
    HSMMParams p = make_two_state_params();
    ParticleFilterConfig cfg;
    cfg.n_particles = 500;
    cfg.seed = 1;
    ParticleFilter pf(p, cfg);

    for (double y : {0.1, -0.2, 3.9, 4.1, 0.05, 3.95}) {
        Vector posterior = pf.step(y);
        double sum = 0.0;
        for (double x : posterior) sum += x;
        REQUIRE_THAT(sum, WithinAbs(1.0, 1e-9));
        for (double x : posterior) {
            REQUIRE(x >= 0.0);
            REQUIRE(x <= 1.0);
        }
    }
}

TEST_CASE("particle_filter: tracks well-separated regime switches with high accuracy",
          "[particle_filter]") {
    HSMMParams p = make_two_state_params();
    // Slightly more persistent duration prior helps the filter avoid
    // spurious single-step flips, matching typical usage.
    p.durations = {DurationParams{10.0, 0.6, 30}, DurationParams{10.0, 0.6, 30}};

    auto series = make_synthetic_series(p, /*n_blocks=*/10, /*block_len=*/8, /*seed=*/7);

    ParticleFilterConfig cfg;
    cfg.n_particles = 3000;
    cfg.seed = 42;
    ParticleFilter pf(p, cfg);

    std::size_t correct = 0;
    for (std::size_t t = 0; t < series.observations.size(); ++t) {
        Vector posterior = pf.step(series.observations[t]);
        std::size_t predicted = (posterior[1] > posterior[0]) ? 1 : 0;
        if (predicted == series.true_states[t]) ++correct;
    }

    const double accuracy = static_cast<double>(correct) / series.observations.size();
    INFO("accuracy = " << accuracy);
    REQUIRE(accuracy > 0.9);
}

TEST_CASE("particle_filter: effective sample size stays within [1, n_particles]",
          "[particle_filter]") {
    HSMMParams p = make_two_state_params();
    ParticleFilterConfig cfg;
    cfg.n_particles = 1000;
    cfg.seed = 3;
    ParticleFilter pf(p, cfg);

    auto series = make_synthetic_series(p, 6, 10, 11);
    for (double y : series.observations) {
        pf.step(y);
        double ess = pf.effective_sample_size();
        REQUIRE(ess >= 1.0);
        REQUIRE(ess <= static_cast<double>(cfg.n_particles) + 1e-6);
    }
}

TEST_CASE("particle_filter: resampling fires and restores ESS close to n_particles",
          "[particle_filter]") {
    HSMMParams p = make_two_state_params();
    ParticleFilterConfig cfg;
    cfg.n_particles = 2000;
    cfg.ess_resample_threshold = 0.5;
    cfg.seed = 5;
    ParticleFilter pf(p, cfg);

    auto series = make_synthetic_series(p, 4, 15, 9);

    bool observed_high_ess_after_transition = false;
    for (std::size_t t = 0; t < series.observations.size(); ++t) {
        pf.step(series.observations[t]);
        double ess = pf.effective_sample_size();
        // Right after a regime transition, particles that "guessed wrong"
        // about the transition timing get down-weighted sharply, then
        // resampling should kick back in and bring ESS back up close to
        // n_particles within the next couple of steps.
        if (ess > 0.95 * static_cast<double>(cfg.n_particles)) {
            observed_high_ess_after_transition = true;
        }
    }
    REQUIRE(observed_high_ess_after_transition);
}

TEST_CASE("particle_filter: all particles start consistent with the initial distribution",
          "[particle_filter]") {
    HSMMParams p = make_two_state_params();
    p.initial_dist = {1.0, 0.0};  // deterministic: everyone starts in state 0

    ParticleFilterConfig cfg;
    cfg.n_particles = 500;
    cfg.seed = 2;
    ParticleFilter pf(p, cfg);

    for (const auto& particle : pf.particles()) {
        REQUIRE(particle.state == 0);
        REQUIRE(particle.time_in_state == 1);
    }
}

TEST_CASE("particle_filter: single-state model never transitions", "[particle_filter][edge]") {
    HSMMParams p;
    p.n_states = 1;
    p.transition = Matrix(1, 1, 0.0);
    p.emissions = {EmissionParams{0.0, 1.0, 30.0}};
    p.durations = {DurationParams{2.0, 0.5, 50}};
    p.initial_dist = {1.0};

    ParticleFilterConfig cfg;
    cfg.n_particles = 200;
    cfg.seed = 4;
    ParticleFilter pf(p, cfg);

    for (int t = 0; t < 20; ++t) {
        Vector posterior = pf.step(0.1 * t);
        REQUIRE(posterior.size() == 1);
        REQUIRE_THAT(posterior[0], WithinAbs(1.0, 1e-9));
    }
    for (const auto& particle : pf.particles()) {
        REQUIRE(particle.state == 0);
    }
}
