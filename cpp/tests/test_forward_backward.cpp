#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <functional>
#include <random>

#include "hsmm/forward_backward.hpp"
#include "hsmm/duration_distributions.hpp"
#include "hsmm/emissions.hpp"

using namespace hsmm;
using Catch::Matchers::WithinAbs;

namespace {

// A small, well-separated 2-state HSMM used across several tests below:
// state 0 clusters around y=0, state 1 clusters around y=4, both with
// tight variance so the "correct" posterior is unambiguous and tests can
// assert on it directly rather than just on internal consistency.
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

}  // namespace

TEST_CASE("forward_backward: posterior gammas sum to 1 at every timestep", "[forward_backward]") {
    HSMMParams p = make_two_state_params();
    Vector obs = {0.1, -0.2, 0.3, 0.0, -0.1, 3.9, 4.1, 3.8, 4.0, 3.95};

    ForwardBackwardResult result = forward_backward(obs, p);

    REQUIRE(result.log_likelihood > -std::numeric_limits<double>::infinity());

    for (std::size_t t = 0; t < obs.size(); ++t) {
        double sum = 0.0;
        for (std::size_t k = 0; k < p.n_states; ++k) {
            sum += std::exp(result.log_gamma(t, k));
        }
        INFO("t = " << t);
        REQUIRE_THAT(sum, WithinAbs(1.0, 1e-6));
    }
}

TEST_CASE("forward_backward: recovers the true regime for well-separated clusters",
          "[forward_backward]") {
    HSMMParams p = make_two_state_params();
    // First half of the series clearly belongs to state 0, second half to
    // state 1; posterior should reflect that with high confidence.
    Vector obs = {0.05, -0.05, 0.10, -0.10, 0.02, 3.95, 4.05, 3.90, 4.10, 4.00};

    ForwardBackwardResult result = forward_backward(obs, p);

    for (std::size_t t = 0; t < 5; ++t) {
        double p_state0 = std::exp(result.log_gamma(t, 0));
        INFO("t = " << t << " P(state0) = " << p_state0);
        REQUIRE(p_state0 > 0.9);
    }
    for (std::size_t t = 5; t < 10; ++t) {
        double p_state1 = std::exp(result.log_gamma(t, 1));
        INFO("t = " << t << " P(state1) = " << p_state1);
        REQUIRE(p_state1 > 0.9);
    }
}

TEST_CASE("forward_backward: single observation degenerates cleanly", "[forward_backward][edge]") {
    HSMMParams p = make_two_state_params();
    Vector obs = {0.0};

    ForwardBackwardResult result = forward_backward(obs, p);

    REQUIRE(result.log_alpha.rows == 1);
    REQUIRE(result.log_likelihood > -std::numeric_limits<double>::infinity());

    double sum = 0.0;
    for (std::size_t k = 0; k < p.n_states; ++k) sum += std::exp(result.log_gamma(0, k));
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-6));

    // With an observation near state 0's mean, state 0 should dominate the
    // single-step posterior.
    REQUIRE(std::exp(result.log_gamma(0, 0)) > std::exp(result.log_gamma(0, 1)));
}

TEST_CASE("forward_backward: empty observation sequence returns empty result",
          "[forward_backward][edge]") {
    HSMMParams p = make_two_state_params();
    Vector obs;

    ForwardBackwardResult result = forward_backward(obs, p);

    REQUIRE(result.log_alpha.rows == 0);
    REQUIRE(result.log_beta.rows == 0);
    REQUIRE(result.log_gamma.rows == 0);
    REQUIRE(result.log_likelihood == -std::numeric_limits<double>::infinity());
}

TEST_CASE("forward_backward: log-likelihood matches brute-force enumeration for tiny T",
          "[forward_backward]") {
    // For a small enough T, we can brute-force enumerate every possible
    // segmentation of [1..T] into state/duration segments and sum their
    // probabilities directly, then compare against the recursive result.
    // This is the strongest correctness check available short of an
    // independent reference implementation.
    HSMMParams p = make_two_state_params();
    p.durations[0].max_duration = 4;
    p.durations[1].max_duration = 4;
    Vector obs = {0.1, 4.2, 0.0};  // T = 3, small enough to enumerate fully

    ForwardBackwardResult result = forward_backward(obs, p);

    // Brute force: recursively enumerate every (state, duration) segment
    // sequence covering exactly T=3 steps, accumulating total probability.
    std::function<double(std::size_t, std::size_t)> total_prob_from =
        [&](std::size_t start, std::size_t prev_state_plus_one) -> double {
        // start: next uncovered time index (0-indexed, 0..T)
        // prev_state_plus_one: previous segment's state + 1, or 0 if this
        // is the first segment (used to select initial_dist vs transition).
        if (start == obs.size()) return 1.0;

        double total = 0.0;
        for (std::size_t k = 0; k < p.n_states; ++k) {
            double state_entry_prob;
            if (prev_state_plus_one == 0) {
                state_entry_prob = p.initial_dist[k];
            } else {
                std::size_t prev_k = prev_state_plus_one - 1;
                if (prev_k == k) continue;  // no self-transitions in this HSMM
                state_entry_prob = p.transition(prev_k, k);
            }
            if (state_entry_prob <= 0.0) continue;

            for (std::size_t d = 1; d <= p.durations[k].max_duration; ++d) {
                if (start + d > obs.size()) break;
                double dur_prob = std::exp(log_duration_pmf(d, p.durations[k]));
                double emission_prob = 1.0;
                for (std::size_t t = start; t < start + d; ++t) {
                    emission_prob *= std::exp(log_student_t_pdf(obs[t], p.emissions[k]));
                }
                double rest = total_prob_from(start + d, k + 1);
                total += state_entry_prob * dur_prob * emission_prob * rest;
            }
        }
        return total;
    };

    double brute_force_likelihood = total_prob_from(0, 0);
    double recursive_likelihood = std::exp(result.log_likelihood);

    REQUIRE_THAT(recursive_likelihood, WithinAbs(brute_force_likelihood, 1e-9));
}

TEST_CASE("forward_backward: higher emission likelihood under the correct model "
          "than a badly mismatched one",
          "[forward_backward]") {
    HSMMParams good = make_two_state_params();

    HSMMParams bad = good;
    bad.emissions = {EmissionParams{100.0, 0.3, 30.0}, EmissionParams{-100.0, 0.3, 30.0}};

    Vector obs = {0.1, -0.1, 0.05, 3.9, 4.1, 3.95};

    auto good_result = forward_backward(obs, good);
    auto bad_result = forward_backward(obs, bad);

    REQUIRE(good_result.log_likelihood > bad_result.log_likelihood);
}
