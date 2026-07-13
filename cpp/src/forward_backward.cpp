#include "hsmm/forward_backward.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "hsmm/duration_distributions.hpp"
#include "hsmm/emissions.hpp"

namespace hsmm {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

double log_sum_exp(double a, double b) {
    if (a == kNegInf) return b;
    if (b == kNegInf) return a;
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

double log_sum_exp_vec(const Vector& v) {
    double m = kNegInf;
    for (double x : v) m = std::max(m, x);
    if (m == kNegInf) return kNegInf;
    double s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

}  // namespace

// ---------------------------------------------------------------------------
// Explicit-duration HSMM forward-backward (Yu, 2010 formulation).
//
// Notation (T = number of observations, K = number of states):
//   log_b(t, k)      = log P(y_t | z_t = k)              [instantaneous emission]
//   log_B(t1, t2, k) = log P(y_{t1..t2} | z = k)          [segment log-lik, i.i.d.
//                                                          emissions within a
//                                                          sojourn, so this is
//                                                          just the sum of
//                                                          log_b over the segment]
//   log_d(d, k)      = log P(duration = d | state k)
//   log_A(k, j)      = log P(next state = j | previous state = k), k != j,
//                      rows of `transition` (self-transitions structurally 0
//                      since the HSMM handles sojourn length explicitly).
//
// Forward "segment-ending" variable (this implementation's log_alpha holds
// this quantity, indexed by end time t and state k):
//
//   log_alpha(t, k) = log P(y_1:t, segment ends at t in state k)
//
// defined via the recursion, for each candidate duration d = 1..max_duration
// of the segment ending at t:
//
//   log_alpha(t, k) = logsumexp_d [ log_eta(t-d, k) + log_d(d,k)
//                                    + log_B(t-d+1, t, k) ]
//
// where log_eta(s, k) = log P(y_1:s, new segment starts right after s in
// state k) = logsumexp_j [ log_alpha(s, j) + log_A(j, k) ],  for s >= 1
// and log_eta(0, k) = log(pi_0[k]) (the initial distribution), handling the
// case where the very first segment starts at t=1.
//
// Backward is the mirror image: log_beta(t, k) = log P(y_{t+1:T} | segment
// that *state k occupies* ends at t), built by looking forward over the
// duration of the *next* segment.
//
// Posterior state marginals log_gamma(t, k) = log P(z_t = k | y_1:T) are
// then reconstructed by summing, over all segments [s+1, e] covering time t
// in state k, the product of: probability of everything before the segment
// starts (eta at s), duration pmf, emission likelihood of the segment, and
// probability of everything after the segment ends (beta at e). This is the
// standard "occupancy" reconstruction for EDHSMMs.
// ---------------------------------------------------------------------------

ForwardBackwardResult forward_backward(const Vector& observations,
                                        const HSMMParams& params) {
    const std::size_t T = observations.size();
    const std::size_t K = params.n_states;

    ForwardBackwardResult result;
    result.log_alpha = Matrix(T, K, kNegInf);
    result.log_beta = Matrix(T, K, kNegInf);
    result.log_gamma = Matrix(T, K, kNegInf);
    result.log_likelihood = kNegInf;

    if (T == 0 || K == 0) {
        return result;
    }

    // --- Precompute per-state instantaneous log-emissions and cumulative
    //     sums, so any segment log-likelihood log_B(t1, t2, k) is an O(1)
    //     lookup: cumsum[k][t2] - cumsum[k][t1 - 1].
    std::vector<Vector> log_b(K, Vector(T));
    std::vector<Vector> cumsum(K, Vector(T + 1, 0.0));
    for (std::size_t k = 0; k < K; ++k) {
        log_b[k] = log_emission_density(observations, params.emissions[k]);
        for (std::size_t t = 0; t < T; ++t) {
            cumsum[k][t + 1] = cumsum[k][t] + log_b[k][t];
        }
    }
    auto log_B = [&](std::size_t t1_idx1based, std::size_t t2_idx1based, std::size_t k) {
        // segment [t1, t2] inclusive, 1-indexed in the math above;
        // cumsum is 0-indexed with cumsum[i] = sum of log_b[0..i-1].
        return cumsum[k][t2_idx1based] - cumsum[k][t1_idx1based - 1];
    };

    // --- Precompute log duration pmfs per state. ---
    std::vector<Vector> log_d(K);
    for (std::size_t k = 0; k < K; ++k) {
        log_d[k] = precompute_log_duration_pmf(params.durations[k]);
    }

    // --- Precompute log transition (no self-loops; entries on the diagonal
    //     are expected to already be effectively -inf / 0 probability, but
    //     we don't rely on that — we simply never use A(k,k) in the sums). ---
    std::vector<Vector> log_A(K, Vector(K, kNegInf));
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t j = 0; j < K; ++j) {
            if (k == j) continue;
            const double p = params.transition(k, j);
            log_A[k][j] = (p > 0.0) ? std::log(p) : kNegInf;
        }
    }

    Vector log_pi0(K);
    for (std::size_t k = 0; k < K; ++k) {
        const double p0 = params.initial_dist[k];
        log_pi0[k] = (p0 > 0.0) ? std::log(p0) : kNegInf;
    }

    // -----------------------------------------------------------------
    // Forward pass.
    //
    // log_eta[s][k] = log P(y_1:s, a new segment begins right after time s
    //                  in state k), for s = 0 .. T-1 (s=0 is "before any
    //                  observation", i.e. the initial-distribution case).
    // We store log_eta indexed 0..T (s=0..T), but only s=0..T-1 are ever
    // used as the "segment start" anchor since a segment starting at s=T
    // would have zero length remaining.
    // -----------------------------------------------------------------
    std::vector<Vector> log_eta(T + 1, Vector(K, kNegInf));
    log_eta[0] = log_pi0;

    for (std::size_t t = 1; t <= T; ++t) {  // t is 1-indexed end-of-segment time
        for (std::size_t k = 0; k < K; ++k) {
            const std::size_t max_d = std::min(t, params.durations[k].max_duration);
            Vector terms;
            terms.reserve(max_d);
            for (std::size_t d = 1; d <= max_d; ++d) {
                const std::size_t s = t - d;  // segment covers (s, t] i.e. s+1..t
                if (log_eta[s][k] == kNegInf) continue;
                const double ld = log_d[k][d - 1];
                if (ld == kNegInf) continue;
                const double lb = log_B(s + 1, t, k);
                terms.push_back(log_eta[s][k] + ld + lb);
            }
            result.log_alpha(t - 1, k) = log_sum_exp_vec(terms);
        }

        if (t < T) {
            // Advance log_eta to s = t: probability that, having just ended
            // a segment at t in state k (log_alpha(t,k)), we transition to
            // state j for the next segment.
            for (std::size_t j = 0; j < K; ++j) {
                Vector terms;
                terms.reserve(K);
                for (std::size_t k = 0; k < K; ++k) {
                    if (k == j) continue;
                    if (result.log_alpha(t - 1, k) == kNegInf) continue;
                    if (log_A[k][j] == kNegInf) continue;
                    terms.push_back(result.log_alpha(t - 1, k) + log_A[k][j]);
                }
                log_eta[t][j] = log_sum_exp_vec(terms);
            }
        }
    }

    // log-likelihood: total probability that a segment ends exactly at T.
    {
        Vector final_terms(K);
        for (std::size_t k = 0; k < K; ++k) final_terms[k] = result.log_alpha(T - 1, k);
        result.log_likelihood = log_sum_exp_vec(final_terms);
    }

    // -----------------------------------------------------------------
    // Backward pass.
    //
    // log_beta(t, k) = log P(y_{t+1:T} | a segment occupying state k has
    //                   just ended at time t).
    // Recursion looks forward over the duration d of the *next* segment,
    // which starts at t+1 and ends at t+d, in some state j != (whatever
    // led here) -- but since beta is defined purely in terms of "segment
    // just ended at t in state k", the *next* segment's state j ranges
    // over all states with k -> j transition probability, and its
    // contribution folds in log_beta at the new end point t+d.
    //
    // Base case: log_beta(T, k) = 0 for all k (nothing left to explain).
    // -----------------------------------------------------------------
    for (std::size_t k = 0; k < K; ++k) {
        result.log_beta(T - 1, k) = 0.0;
    }

    // We iterate t from T-1 down to 0 (0-indexed time positions,
    // corresponding to "segment ended at 1-indexed time t+1").
    for (std::size_t t_plus_1 = T - 1; t_plus_1 >= 1; --t_plus_1) {
        const std::size_t t = t_plus_1 - 1;  // 0-indexed "ended at" position - 1 step back... 
        for (std::size_t k = 0; k < K; ++k) {
            Vector outer_terms;
            outer_terms.reserve(K);
            for (std::size_t j = 0; j < K; ++j) {
                if (j == k) continue;
                if (log_A[k][j] == kNegInf) continue;
                const std::size_t max_d =
                    std::min(T - (t_plus_1), params.durations[j].max_duration);
                Vector inner_terms;
                inner_terms.reserve(max_d);
                for (std::size_t d = 1; d <= max_d; ++d) {
                    const std::size_t seg_end_1idx = t_plus_1 + d;  // 1-indexed end
                    if (seg_end_1idx > T) continue;
                    const double ld = log_d[j][d - 1];
                    if (ld == kNegInf) continue;
                    const double lb = log_B(t_plus_1 + 1, seg_end_1idx, j);
                    const double lbeta_next = result.log_beta(seg_end_1idx - 1, j);
                    if (lbeta_next == kNegInf) continue;
                    inner_terms.push_back(ld + lb + lbeta_next);
                }
                const double inner = log_sum_exp_vec(inner_terms);
                if (inner == kNegInf) continue;
                outer_terms.push_back(log_A[k][j] + inner);
            }
            result.log_beta(t, k) = log_sum_exp_vec(outer_terms);
        }
        if (t_plus_1 == 1) break;  // guard against unsigned underflow
    }

    // -----------------------------------------------------------------
    // Posterior state occupancy: log_gamma(t, k) = log P(z_t = k | y_1:T).
    //
    // Reconstructed by summing over all segments [s+1, e] (0-indexed times
    // s..e-1 covering position t, i.e. s < t <= e-1 in 0-indexed terms /
    // s+1 <= t+1 <= e in 1-indexed terms) in state k:
    //
    //   P(segment [s+1,e] in state k, y_1:T)
    //     = log_eta[s][k] + log_d[k](e-s) + log_B(s+1,e,k) + log_beta(e-1,k)
    //
    // for every (s, e) pair such that the segment covers time t (1-indexed:
    // s+1 <= t <= e), normalized by log_likelihood.
    //
    // Direct O(T^2 K) summation (acceptable for the sizes this engine
    // targets - daily data over a handful of years, T in the low thousands,
    // K small). A smarter O(T*K*max_duration) formulation exists but is
    // deferred; this is correct and serves as the reference implementation.
    // -----------------------------------------------------------------
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t t1idx = 1; t1idx <= T; ++t1idx) {  // t, 1-indexed
            Vector terms;
            for (std::size_t s = 0; s < t1idx; ++s) {  // segment start anchor
                if (log_eta[s][k] == kNegInf) continue;
                const std::size_t min_e = t1idx;  // segment must reach at least t
                const std::size_t max_d_possible =
                    std::min(T - s, params.durations[k].max_duration);
                for (std::size_t d = 1; d <= max_d_possible; ++d) {
                    const std::size_t e = s + d;  // 1-indexed end
                    if (e < min_e) continue;      // segment doesn't reach t yet
                    if (e > T) break;
                    const double ld = log_d[k][d - 1];
                    if (ld == kNegInf) continue;
                    const double lb = log_B(s + 1, e, k);
                    const double lbeta = result.log_beta(e - 1, k);
                    if (lbeta == kNegInf) continue;
                    terms.push_back(log_eta[s][k] + ld + lb + lbeta);
                }
            }
            const double numerator = log_sum_exp_vec(terms);
            result.log_gamma(t1idx - 1, k) = numerator - result.log_likelihood;
        }
    }

    return result;
}

}  // namespace hsmm
