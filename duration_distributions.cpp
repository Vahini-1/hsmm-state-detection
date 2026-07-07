#include "hsmm/duration_distributions.hpp"

#include <cmath>
#include <limits>

namespace hsmm {

namespace {

// log Negative-Binomial PMF for d = 1, 2, 3, ... (shifted so the minimum
// sojourn time is 1 period, matching how regime durations are counted
// elsewhere in the engine: a state entered and left on the same step has
// duration 1, not 0).
//
// We parameterize the "number of trials until r failures" NB in terms of
// (r, p) and treat k = d - 1 as the number of successes before the r-th
// failure, i.e. the standard NB(r, p) support on {0, 1, 2, ...} shifted by
// one so d >= 1:
//   P(D = d) = C(d - 1 + r - 1, d - 1) * p^r * (1 - p)^(d - 1)
double log_nb_pmf(std::size_t d, double r, double p) {
    if (d < 1) return -std::numeric_limits<double>::infinity();
    const double k = static_cast<double>(d - 1);
    // log C(k + r - 1, k) = lgamma(k + r) - lgamma(r) - lgamma(k + 1)
    const double log_binom = std::lgamma(k + r) - std::lgamma(r) - std::lgamma(k + 1.0);
    return log_binom + r * std::log(p) + k * std::log(1.0 - p);
}

}  // namespace

double log_duration_pmf(std::size_t d, const DurationParams& params) {
    if (d < 1 || d > params.max_duration) {
        return -std::numeric_limits<double>::infinity();
    }
    return log_nb_pmf(d, params.r, params.p);
}

Vector precompute_log_duration_pmf(const DurationParams& params) {
    Vector out(params.max_duration, -std::numeric_limits<double>::infinity());
    for (std::size_t d = 1; d <= params.max_duration; ++d) {
        out[d - 1] = log_nb_pmf(d, params.r, params.p);
    }
    return out;
}

// log P(D > d) = log(1 - CDF(d)), computed by summing the PMF in log-space
// via log-sum-exp for numerical stability, then taking log1p(-exp(logCDF)).
//
// For the HSMM recursions we need this for d = 0 .. max_duration (the
// "still surviving past d periods" term), so d = 0 should return log(1) = 0.
double log_duration_survivor(std::size_t d, const DurationParams& params) {
    if (d == 0) return 0.0;
    if (d >= params.max_duration) {
        return -std::numeric_limits<double>::infinity();
    }

    // log-sum-exp over log P(D = 1), ..., log P(D = d) to get log CDF(d).
    double max_log = -std::numeric_limits<double>::infinity();
    Vector logs(d);
    for (std::size_t j = 1; j <= d; ++j) {
        logs[j - 1] = log_nb_pmf(j, params.r, params.p);
        if (logs[j - 1] > max_log) max_log = logs[j - 1];
    }
    if (max_log == -std::numeric_limits<double>::infinity()) {
        return 0.0;  // CDF(d) == 0, so survivor == log(1) == 0
    }
    double sum_exp = 0.0;
    for (double lg : logs) sum_exp += std::exp(lg - max_log);
    const double log_cdf = max_log + std::log(sum_exp);

    const double cdf = std::exp(log_cdf);
    if (cdf >= 1.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return std::log1p(-cdf);
}

// Method-of-moments fit for Negative-Binomial (r, p) from expected duration
// sufficient statistics collected during the EM E-step. We expect
// `expected_duration_counts[d-1]` to hold the expected number of times a
// sojourn of exact length d was observed under the posterior (i.e. the
// duration histogram weighted by posterior responsibility), for
// d = 1 .. max_duration.
//
// Given shifted support (k = d - 1 >= 0 counts "successes" before failure),
// standard NB method-of-moments on k gives:
//   mean(k) = r(1-p)/p,  var(k) = r(1-p)/p^2
//   => p = mean(k) / var(k),  r = mean(k) * p / (1 - p)
// with guards for degenerate/underdispersed cases (var <= mean), where we
// fall back to a near-geometric distribution (large r, correspondingly
// adjusted p) rather than producing an invalid negative r.
DurationParams fit_duration_params(const Vector& expected_duration_counts) {
    DurationParams out;
    out.max_duration = expected_duration_counts.size();

    double total_count = 0.0;
    double sum_k = 0.0;
    for (std::size_t i = 0; i < expected_duration_counts.size(); ++i) {
        const double d = static_cast<double>(i + 1);
        const double k = d - 1.0;
        total_count += expected_duration_counts[i];
        sum_k += expected_duration_counts[i] * k;
    }

    if (total_count <= 0.0) {
        // No data assigned to this regime's durations; return a weakly
        // informative default (moderate persistence, geometric-like).
        out.r = 1.0;
        out.p = 0.5;
        return out;
    }

    const double mean_k = sum_k / total_count;

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < expected_duration_counts.size(); ++i) {
        const double d = static_cast<double>(i + 1);
        const double k = d - 1.0;
        const double diff = k - mean_k;
        sum_sq += expected_duration_counts[i] * diff * diff;
    }
    const double var_k = sum_sq / total_count;

    constexpr double kEps = 1e-6;
    constexpr double kMaxR = 1e4;

    if (var_k <= mean_k + kEps) {
        // Under- or equi-dispersed relative to NB's minimum variance
        // (var >= mean for NB with r > 0); push r large so the NB
        // approaches a degenerate/near-Poisson shape without breaking the
        // parameterization used elsewhere (r, p) with p in (0, 1).
        out.r = kMaxR;
        out.p = out.r / (out.r + mean_k);
    } else {
        double p = mean_k / var_k;
        p = std::max(1e-4, std::min(p, 1.0 - 1e-4));
        double r = mean_k * p / (1.0 - p);
        r = std::max(1e-3, std::min(r, kMaxR));
        out.r = r;
        out.p = p;
    }

    return out;
}

}  // namespace hsmm
