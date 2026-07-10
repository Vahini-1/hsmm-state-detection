#include "hsmm/em_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "hsmm/duration_distributions.hpp"
#include "hsmm/emissions.hpp"

namespace hsmm {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

double log_sum_exp_vec(const Vector& v) {
    double m = kNegInf;
    for (double x : v) m = std::max(m, x);
    if (m == kNegInf) return kNegInf;
    double s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

// -----------------------------------------------------------------------
// E-step sufficient statistics needed for the M-step of an explicit-
// duration HSMM:
//   - gamma(t, k): posterior state occupancy (from forward_backward)
//   - xi(k, j):    expected number of k -> j regime transitions
//   - expected duration histogram per state: expected count of sojourns of
//     exact length d in state k, used to refit the Negative-Binomial
//     duration params
//
// We reconstruct xi and the duration histogram directly from the same
// (log_eta-equivalent) forward/backward quantities that forward_backward
// computes internally. Since forward_backward() doesn't expose log_eta
// externally, we recompute the segment-level posterior here using log_alpha
// and log_beta plus local re-derivation of segment boundaries. This keeps
// forward_backward()'s public interface stable (as required by the header)
// while still letting EM collect the duration/transition statistics it
// needs from the segment structure.
// -----------------------------------------------------------------------
struct SegmentStats {
    Matrix xi;                       // K x K expected transition counts
    std::vector<Vector> duration_hist; // size K, each length max_duration
};

SegmentStats compute_segment_stats(const Vector& observations,
                                    const HSMMParams& params,
                                    const ForwardBackwardResult& fb) {
    const std::size_t T = observations.size();
    const std::size_t K = params.n_states;

    SegmentStats stats;
    stats.xi = Matrix(K, K, 0.0);
    stats.duration_hist.assign(K, Vector());
    for (std::size_t k = 0; k < K; ++k) {
        stats.duration_hist[k] = Vector(params.durations[k].max_duration, 0.0);
    }

    if (T == 0 || K == 0 || fb.log_likelihood == kNegInf) return stats;

    // Rebuild per-state cumulative log-emissions (cheap, O(T*K)).
    std::vector<Vector> cumsum(K, Vector(T + 1, 0.0));
    for (std::size_t k = 0; k < K; ++k) {
        Vector lb = log_emission_density(observations, params.emissions[k]);
        for (std::size_t t = 0; t < T; ++t) cumsum[k][t + 1] = cumsum[k][t] + lb[t];
    }
    auto log_B = [&](std::size_t t1, std::size_t t2, std::size_t k) {
        return cumsum[k][t2] - cumsum[k][t1 - 1];
    };

    std::vector<Vector> log_d(K);
    for (std::size_t k = 0; k < K; ++k) log_d[k] = precompute_log_duration_pmf(params.durations[k]);

    std::vector<Vector> log_A(K, Vector(K, kNegInf));
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < K; ++j)
            if (k != j && params.transition(k, j) > 0.0) log_A[k][j] = std::log(params.transition(k, j));

    Vector log_pi0(K);
    for (std::size_t k = 0; k < K; ++k)
        log_pi0[k] = params.initial_dist[k] > 0.0 ? std::log(params.initial_dist[k]) : kNegInf;

    // Recompute log_eta[s][k] (segment-start posterior anchor), matching
    // forward_backward's internal recursion, since we need it again here to
    // reconstruct segment-level posteriors for xi and duration histograms.
    std::vector<Vector> log_eta(T + 1, Vector(K, kNegInf));
    log_eta[0] = log_pi0;
    for (std::size_t t = 1; t <= T; ++t) {
        for (std::size_t j = 0; j < K; ++j) {
            if (t == T) break;  // log_eta[T] never used
            Vector terms;
            for (std::size_t k = 0; k < K; ++k) {
                if (k == j) continue;
                const double la = fb.log_alpha(t - 1, k);
                if (la == kNegInf || log_A[k][j] == kNegInf) continue;
                terms.push_back(la + log_A[k][j]);
            }
            log_eta[t][j] = log_sum_exp_vec(terms);
        }
    }

    const double ll = fb.log_likelihood;

    // For every candidate segment (s, e] in state k (s = start anchor,
    // e = end time, 1-indexed), its posterior weight is:
    //   log_eta[s][k] + log_d[k][e-s-1] + log_B(s+1,e,k) + log_beta(e-1,k) - ll
    // We accumulate this weight into:
    //   - duration_hist[k][d-1]  (d = e - s)
    //   - xi(k, j) where j is the state of the *next* segment (found via
    //     the same reconstruction one level up, using log_alpha(e-1,k) and
    //     log_A[k][j] and the subsequent segment's contribution) -- for
    //     simplicity and numerical robustness we compute xi via the
    //     standard two-slice marginal: P(z ends segment in k at e, next
    //     segment state j) using log_alpha(e-1,k) + log_A[k][j] + (sum over
    //     next segment durations of log_d[j] + log_B + log_beta), all
    //     divided by ll.
    for (std::size_t k = 0; k < K; ++k) {
        const std::size_t max_dur = params.durations[k].max_duration;
        for (std::size_t e = 1; e <= T; ++e) {
            const std::size_t max_d = std::min(e, max_dur);
            for (std::size_t d = 1; d <= max_d; ++d) {
                const std::size_t s = e - d;
                if (log_eta[s][k] == kNegInf) continue;
                const double ld = log_d[k][d - 1];
                if (ld == kNegInf) continue;
                const double lbeta = fb.log_beta(e - 1, k);
                if (lbeta == kNegInf) continue;
                const double lb = log_B(s + 1, e, k);
                const double log_w = log_eta[s][k] + ld + lb + lbeta - ll;
                if (log_w > -700.0) {  // avoid underflow to 0 noise accumulation
                    stats.duration_hist[k][d - 1] += std::exp(log_w);
                }
            }
        }

        // Transition statistics: for every time e (1..T-1) where a segment
        // in state k ends, and every next state j, weight by
        // log_alpha(e-1,k) + log_A[k][j] + [sum over next-segment durations
        // d' of log_d[j][d'-1] + log_B(e+1, e+d', j) + log_beta(e+d'-1, j)].
        for (std::size_t e = 1; e < T; ++e) {
            const double la = fb.log_alpha(e - 1, k);
            if (la == kNegInf) continue;
            for (std::size_t j = 0; j < K; ++j) {
                if (j == k || log_A[k][j] == kNegInf) continue;
                const std::size_t max_d = std::min(T - e, params.durations[j].max_duration);
                Vector terms;
                terms.reserve(max_d);
                for (std::size_t dprime = 1; dprime <= max_d; ++dprime) {
                    const std::size_t e2 = e + dprime;
                    const double ldj = log_d[j][dprime - 1];
                    if (ldj == kNegInf) continue;
                    const double lbetaj = fb.log_beta(e2 - 1, j);
                    if (lbetaj == kNegInf) continue;
                    const double lbj = log_B(e + 1, e2, j);
                    terms.push_back(ldj + lbj + lbetaj);
                }
                const double inner = log_sum_exp_vec(terms);
                if (inner == kNegInf) continue;
                const double log_w = la + log_A[k][j] + inner - ll;
                if (log_w > -700.0) {
                    stats.xi(k, j) += std::exp(log_w);
                }
            }
        }
    }

    return stats;
}

}  // namespace

EMResult fit_hsmm_em(const Vector& observations,
                      const HSMMParams& init_params,
                      const EMConfig& config) {
    EMResult result;
    result.params = init_params;
    result.converged = false;

    const std::size_t K = init_params.n_states;
    if (observations.empty() || K == 0) {
        return result;
    }

    HSMMParams params = init_params;
    double prev_ll = kNegInf;

    for (std::size_t iter = 0; iter < config.max_iters; ++iter) {
        // --- E-step ---
        ForwardBackwardResult fb = forward_backward(observations, params);
        result.log_likelihood_history.push_back(fb.log_likelihood);

        if (config.verbose) {
            // Intentionally minimal; caller can redirect/parse if wiring to
            // a logger. Kept as a no-op-friendly single line.
        }

        if (iter > 0) {
            const double delta = fb.log_likelihood - prev_ll;
            if (std::abs(delta) < config.tol) {
                result.converged = true;
                result.params = params;
                return result;
            }
            // EM should be monotonically non-decreasing in log-likelihood;
            // a significant decrease indicates a numerical issue rather
            // than a legitimate step, so we stop rather than diverge
            // further and hand back the best params seen so far.
            if (delta < -1e-4) {
                result.converged = false;
                result.params = params;
                return result;
            }
        }
        prev_ll = fb.log_likelihood;

        SegmentStats stats = compute_segment_stats(observations, params, fb);

        // --- M-step: emissions ---
        HSMMParams new_params = params;
        for (std::size_t k = 0; k < K; ++k) {
            Vector resp(observations.size());
            for (std::size_t t = 0; t < observations.size(); ++t) {
                resp[t] = std::exp(fb.log_gamma(t, k));
            }
            new_params.emissions[k] = fit_emission_params(observations, resp);
        }

        // --- M-step: duration distributions ---
        for (std::size_t k = 0; k < K; ++k) {
            new_params.durations[k] = fit_duration_params(stats.duration_hist[k]);
            new_params.durations[k].max_duration = params.durations[k].max_duration;
        }

        // --- M-step: transition matrix (row-normalize xi, zero diagonal) ---
        for (std::size_t k = 0; k < K; ++k) {
            double row_sum = 0.0;
            for (std::size_t j = 0; j < K; ++j) {
                if (j == k) continue;
                row_sum += stats.xi(k, j);
            }
            if (row_sum > 1e-12) {
                for (std::size_t j = 0; j < K; ++j) {
                    new_params.transition(k, j) = (j == k) ? 0.0 : stats.xi(k, j) / row_sum;
                }
            }
            // else: leave the previous row unchanged (state k's segments
            // were never observed to end within the sample; refitting its
            // outgoing transitions from zero evidence would be arbitrary).
        }

        // --- M-step: initial distribution from gamma at t=0 ---
        double g0_sum = 0.0;
        for (std::size_t k = 0; k < K; ++k) g0_sum += std::exp(fb.log_gamma(0, k));
        if (g0_sum > 1e-12) {
            for (std::size_t k = 0; k < K; ++k) {
                new_params.initial_dist[k] = std::exp(fb.log_gamma(0, k)) / g0_sum;
            }
        }

        params = new_params;
    }

    result.params = params;
    return result;
}

// -----------------------------------------------------------------------
// Viterbi-style most likely (state, duration) path for an explicit-
// duration HSMM. Standard segmental Viterbi: delta(t, k) = best
// log-probability of any state/duration path over y_1:t that ends a
// segment in state k exactly at t. Backpointers store the chosen
// segment's start time and previous state, letting us walk back and
// unroll into a length-T per-timestep label vector at the end.
// -----------------------------------------------------------------------
std::vector<std::size_t> most_likely_state_path(const Vector& observations,
                                                  const HSMMParams& params) {
    const std::size_t T = observations.size();
    const std::size_t K = params.n_states;
    std::vector<std::size_t> path(T, 0);
    if (T == 0 || K == 0) return path;

    std::vector<Vector> cumsum(K, Vector(T + 1, 0.0));
    for (std::size_t k = 0; k < K; ++k) {
        Vector lb = log_emission_density(observations, params.emissions[k]);
        for (std::size_t t = 0; t < T; ++t) cumsum[k][t + 1] = cumsum[k][t] + lb[t];
    }
    auto log_B = [&](std::size_t t1, std::size_t t2, std::size_t k) {
        return cumsum[k][t2] - cumsum[k][t1 - 1];
    };

    std::vector<Vector> log_d(K);
    for (std::size_t k = 0; k < K; ++k) log_d[k] = precompute_log_duration_pmf(params.durations[k]);

    std::vector<Vector> log_A(K, Vector(K, kNegInf));
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < K; ++j)
            if (k != j && params.transition(k, j) > 0.0) log_A[k][j] = std::log(params.transition(k, j));

    Vector log_pi0(K);
    for (std::size_t k = 0; k < K; ++k)
        log_pi0[k] = params.initial_dist[k] > 0.0 ? std::log(params.initial_dist[k]) : kNegInf;

    // delta_star[s][k]: best log-prob of a path over y_1:s ending with a
    // *new segment about to start* right after s, in state k (mirrors
    // log_eta in the forward pass, but max instead of sum).
    std::vector<Vector> delta_star(T + 1, Vector(K, kNegInf));
    delta_star[0] = log_pi0;

    // best_prev[s][k]: argmax previous state feeding into delta_star[s][k]
    // (unused when s == 0, since that's the initial distribution).
    std::vector<std::vector<std::size_t>> best_prev(T + 1, std::vector<std::size_t>(K, 0));

    Matrix delta(T, K, kNegInf);            // delta(t-1, k) == segment ends at t in state k
    std::vector<std::vector<std::size_t>> best_start(T, std::vector<std::size_t>(K, 0));

    for (std::size_t t = 1; t <= T; ++t) {
        for (std::size_t k = 0; k < K; ++k) {
            const std::size_t max_d = std::min(t, params.durations[k].max_duration);
            double best_val = kNegInf;
            std::size_t best_s = 0;
            for (std::size_t d = 1; d <= max_d; ++d) {
                const std::size_t s = t - d;
                if (delta_star[s][k] == kNegInf) continue;
                const double ld = log_d[k][d - 1];
                if (ld == kNegInf) continue;
                const double val = delta_star[s][k] + ld + log_B(s + 1, t, k);
                if (val > best_val) {
                    best_val = val;
                    best_s = s;
                }
            }
            delta(t - 1, k) = best_val;
            best_start[t - 1][k] = best_s;
        }

        if (t < T) {
            for (std::size_t j = 0; j < K; ++j) {
                double best_val = kNegInf;
                std::size_t best_k = 0;
                for (std::size_t k = 0; k < K; ++k) {
                    if (k == j || log_A[k][j] == kNegInf) continue;
                    if (delta(t - 1, k) == kNegInf) continue;
                    const double val = delta(t - 1, k) + log_A[k][j];
                    if (val > best_val) {
                        best_val = val;
                        best_k = k;
                    }
                }
                delta_star[t][j] = best_val;
                best_prev[t][j] = best_k;
            }
        }
    }

    // Find best final state at t = T.
    double best_final_val = kNegInf;
    std::size_t best_final_state = 0;
    for (std::size_t k = 0; k < K; ++k) {
        if (delta(T - 1, k) > best_final_val) {
            best_final_val = delta(T - 1, k);
            best_final_state = k;
        }
    }

    // Backtrack: unroll segments from the end.
    std::size_t t = T;
    std::size_t state = best_final_state;
    while (t > 0) {
        const std::size_t s = best_start[t - 1][state];
        for (std::size_t tt = s; tt < t; ++tt) {
            path[tt] = state;
        }
        if (s == 0) break;
        const std::size_t prev_state = best_prev[s][state];
        t = s;
        state = prev_state;
    }

    return path;
}

}  // namespace hsmm
