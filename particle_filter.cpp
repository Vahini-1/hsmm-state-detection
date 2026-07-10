#include "hsmm/particle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "hsmm/duration_distributions.hpp"
#include "hsmm/emissions.hpp"

namespace hsmm {

namespace {

// Draw a discrete index from an unnormalized weight vector (linear scale,
// not log). Used for sampling the next regime when a particle's sojourn
// ends and it must transition.
std::size_t sample_categorical(const Vector& weights, std::mt19937& rng) {
    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
    return dist(rng);
}

}  // namespace

ParticleFilter::ParticleFilter(const HSMMParams& params, const ParticleFilterConfig& config)
    : params_(params), config_(config), rng_(config.seed) {
    initialize_particles();
}

void ParticleFilter::initialize_particles() {
    particles_.assign(config_.n_particles, Particle{});
    const double log_w0 = -std::log(static_cast<double>(config_.n_particles));

    std::discrete_distribution<std::size_t> init_dist(params_.initial_dist.begin(),
                                                        params_.initial_dist.end());
    for (auto& particle : particles_) {
        particle.state = init_dist(rng_);
        particle.time_in_state = 1;  // just entered its first regime at t=1
        particle.log_weight = log_w0;
    }
}

double ParticleFilter::effective_sample_size() const {
    // ESS = 1 / sum(w_i^2), with w_i the *normalized* (linear-scale)
    // weights. Recompute normalization here from log_weight for accuracy.
    double max_lw = -std::numeric_limits<double>::infinity();
    for (const auto& p : particles_) max_lw = std::max(max_lw, p.log_weight);

    double sum_exp = 0.0;
    for (const auto& p : particles_) sum_exp += std::exp(p.log_weight - max_lw);

    double sum_sq = 0.0;
    for (const auto& p : particles_) {
        const double w = std::exp(p.log_weight - max_lw) / sum_exp;
        sum_sq += w * w;
    }
    if (sum_sq <= 0.0) return 0.0;
    return 1.0 / sum_sq;
}

void ParticleFilter::systematic_resample() {
    const std::size_t N = particles_.size();

    // Normalize weights to linear scale.
    double max_lw = -std::numeric_limits<double>::infinity();
    for (const auto& p : particles_) max_lw = std::max(max_lw, p.log_weight);
    Vector w(N);
    double sum_w = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        w[i] = std::exp(particles_[i].log_weight - max_lw);
        sum_w += w[i];
    }
    for (double& wi : w) wi /= sum_w;

    // Cumulative distribution for systematic resampling.
    Vector cdf(N);
    std::partial_sum(w.begin(), w.end(), cdf.begin());
    cdf.back() = 1.0;  // guard against floating point drift

    std::uniform_real_distribution<double> u01(0.0, 1.0 / static_cast<double>(N));
    const double u0 = u01(rng_);

    std::vector<Particle> resampled;
    resampled.reserve(N);
    std::size_t cdf_idx = 0;
    for (std::size_t m = 0; m < N; ++m) {
        const double u = u0 + static_cast<double>(m) / static_cast<double>(N);
        while (cdf_idx + 1 < N && cdf[cdf_idx] < u) {
            ++cdf_idx;
        }
        resampled.push_back(particles_[cdf_idx]);
    }

    particles_ = std::move(resampled);
    const double uniform_log_w = -std::log(static_cast<double>(N));
    for (auto& p : particles_) p.log_weight = uniform_log_w;
}

void ParticleFilter::resample_if_needed() {
    const double ess = effective_sample_size();
    const double threshold = config_.ess_resample_threshold * static_cast<double>(particles_.size());
    if (ess < threshold) {
        systematic_resample();
    }
}

Vector ParticleFilter::step(double y_t) {
    const std::size_t K = params_.n_states;

    // --- Propagate: for each particle, decide whether its current
    //     sojourn continues or a transition occurs, using the semi-Markov
    //     duration model's hazard (discrete-time survival) function:
    //
    //     P(leave state now | survived time_in_state so far)
    //       = 1 - P(D > time_in_state | D >= time_in_state)
    //       = 1 - survivor(time_in_state) / survivor(time_in_state - 1)
    //
    //     i.e. the standard discrete hazard rate derived from the duration
    //     survivor function, so states with fat-tailed/long-duration
    //     distributions naturally persist longer without needing an
    //     explicit "self-transition probability" (which the HSMM
    //     transition matrix does not model, by design). ---
    for (auto& particle : particles_) {
        const std::size_t k = particle.state;
        const auto& dur_params = params_.durations[k];

        const double log_surv_prev =
            log_duration_survivor(particle.time_in_state - 1, dur_params);
        const double log_surv_curr =
            log_duration_survivor(particle.time_in_state, dur_params);

        double hazard;
        if (log_surv_prev == -std::numeric_limits<double>::infinity()) {
            // Already past the support of the duration distribution;
            // force a transition.
            hazard = 1.0;
        } else {
            const double surv_ratio = std::exp(log_surv_curr - log_surv_prev);
            hazard = std::max(0.0, std::min(1.0, 1.0 - surv_ratio));
        }

        std::bernoulli_distribution transition_now(hazard);
        if (transition_now(rng_)) {
            // Sample the next state according to the transition matrix's
            // row for the current state.
            Vector row(K, 0.0);
            for (std::size_t j = 0; j < K; ++j) {
                row[j] = (j == k) ? 0.0 : params_.transition(k, j);
            }
            const double row_sum = std::accumulate(row.begin(), row.end(), 0.0);
            if (row_sum > 0.0) {
                particle.state = sample_categorical(row, rng_);
            }
            // else: no valid outgoing transition defined; stay in place
            // rather than crash (defensive fallback for malformed params).
            particle.time_in_state = 1;
        } else {
            particle.time_in_state += 1;
        }
    }

    // --- Weight update: incorporate the new observation's likelihood
    //     under each particle's (possibly just-updated) current state. ---
    for (auto& particle : particles_) {
        const double log_lik = log_student_t_pdf(y_t, params_.emissions[particle.state]);
        particle.log_weight += log_lik;
    }

    // --- Normalize weights (in log-space, then renormalize to sum to 1
    //     for reporting / resampling stability) and compute the posterior
    //     regime distribution as a particle-weighted histogram over
    //     states. ---
    double max_lw = -std::numeric_limits<double>::infinity();
    for (const auto& p : particles_) max_lw = std::max(max_lw, p.log_weight);

    Vector posterior(K, 0.0);
    double sum_exp = 0.0;
    for (const auto& p : particles_) {
        const double w = std::exp(p.log_weight - max_lw);
        sum_exp += w;
        posterior[p.state] += w;
    }
    for (double& p : posterior) p /= sum_exp;

    // Renormalize log_weight so it doesn't drift to extreme magnitudes
    // over long streaming runs (equivalent posterior, better numerics).
    for (auto& particle : particles_) {
        particle.log_weight = std::log(std::exp(particle.log_weight - max_lw) / sum_exp);
    }

    resample_if_needed();

    return posterior;
}

}  // namespace hsmm
