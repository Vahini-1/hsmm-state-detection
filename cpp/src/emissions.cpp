#include "hsmm/emissions.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hsmm {

namespace {

// log of the multivariate/univariate Student-t normalizing constant:
// log C(nu, sigma) = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi) - log(sigma)
double log_norm_const(double sigma, double nu) {
    return std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) -
           0.5 * std::log(nu * M_PI) - std::log(sigma);
}

// Digamma via Newton/asymptotic recurrence, used for the Newton update
// on the Student-t degrees-of-freedom parameter.
double digamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    double f = 1.0 / (x * x);
    result += std::log(x) - 0.5 / x -
               f * (1.0 / 12.0 -
                    f * (1.0 / 120.0 -
                         f * (1.0 / 252.0 - f * (1.0 / 240.0))));
    return result;
}

double trigamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    double f = 1.0 / (x * x);
    result += 1.0 / x + f / 2.0 +
              (1.0 / 6.0 - f * (1.0 / 30.0 - f * (1.0 / 42.0 - f / 30.0))) / x;
    return result;
}

}  // namespace

double log_student_t_pdf(double y, const EmissionParams& params) {
    const double mu = params.mu;
    const double sigma = params.sigma;
    const double nu = params.nu;

    if (sigma <= 0.0 || nu <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    const double z = (y - mu) / sigma;
    const double log_kernel = -((nu + 1.0) / 2.0) * std::log1p((z * z) / nu);
    return log_norm_const(sigma, nu) + log_kernel;
}

Vector log_emission_density(const Vector& observations, const EmissionParams& params) {
    Vector out(observations.size());
    for (std::size_t t = 0; t < observations.size(); ++t) {
        out[t] = log_student_t_pdf(observations[t], params);
    }
    return out;
}

// Weighted MLE for Student-t (mu, sigma, nu) via ECM-style iterations:
// Given responsibilities w_t (posterior weight of this regime at time t),
// we alternate:
//   1. E-step-within-M-step: compute per-point "weights" u_t that come from
//      treating the Student-t as a Gaussian scale-mixture (u_t down-weights
//      outliers), given current (mu, sigma, nu).
//   2. Weighted-mean/variance update for (mu, sigma) using w_t * u_t.
//   3. 1-D Newton step on nu using the profile log-likelihood.
EmissionParams fit_emission_params(const Vector& observations,
                                    const Vector& responsibilities) {
    if (observations.size() != responsibilities.size()) {
        throw std::invalid_argument(
            "fit_emission_params: observations and responsibilities size mismatch");
    }
    const std::size_t T = observations.size();
    if (T == 0) {
        return EmissionParams{};
    }

    double sum_w = 0.0;
    for (double w : responsibilities) sum_w += w;
    if (sum_w <= 0.0) {
        // No mass assigned to this regime; return a harmless default rather
        // than dividing by zero. Caller (EM engine) should guard against
        // regime starvation separately.
        return EmissionParams{};
    }

    // Initialize from weighted mean / weighted std (Gaussian MLE) as a
    // starting point for the EM-within-EM iterations.
    double mu = 0.0;
    for (std::size_t t = 0; t < T; ++t) mu += responsibilities[t] * observations[t];
    mu /= sum_w;

    double var = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        const double d = observations[t] - mu;
        var += responsibilities[t] * d * d;
    }
    var /= sum_w;
    double sigma = std::sqrt(std::max(var, 1e-12));
    double nu = 5.0;  // reasonable fat-tailed starting point

    constexpr int kMaxOuterIters = 25;
    constexpr double kTol = 1e-7;

    double prev_mu = mu, prev_sigma = sigma, prev_nu = nu;

    for (int iter = 0; iter < kMaxOuterIters; ++iter) {
        // --- E-step: scale-mixture weights u_t = (nu + 1) / (nu + z_t^2) ---
        Vector u(T);
        double sum_wu = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            const double z = (observations[t] - mu) / sigma;
            u[t] = (nu + 1.0) / (nu + z * z);
            sum_wu += responsibilities[t] * u[t];
        }

        // --- M-step: weighted mean using w_t * u_t as effective weights ---
        double new_mu = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            new_mu += responsibilities[t] * u[t] * observations[t];
        }
        new_mu /= sum_wu;

        double new_var = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            const double d = observations[t] - new_mu;
            new_var += responsibilities[t] * u[t] * d * d;
        }
        new_var /= sum_w;
        double new_sigma = std::sqrt(std::max(new_var, 1e-12));

        // --- Newton step on nu, maximizing the weighted Student-t
        //     log-likelihood profile w.r.t. nu given (mu, sigma) fixed. ---
        double new_nu = nu;
        for (int newton_it = 0; newton_it < 10; ++newton_it) {
            double g = 0.0;   // gradient of avg log-lik wrt nu
            double h = 0.0;   // (negative) second derivative, for Newton step
            const double half_nu = new_nu / 2.0;
            const double half_nu1 = (new_nu + 1.0) / 2.0;
            const double dg_half_nu1 = digamma(half_nu1);
            const double dg_half_nu = digamma(half_nu);
            const double tg_half_nu1 = trigamma(half_nu1);
            const double tg_half_nu = trigamma(half_nu);

            for (std::size_t t = 0; t < T; ++t) {
                const double w = responsibilities[t];
                const double z = (observations[t] - new_mu) / new_sigma;
                const double z2_over_nu = (z * z) / new_nu;
                const double log_term = std::log1p(z2_over_nu);
                const double frac = z2_over_nu / (1.0 + z2_over_nu);

                // d/dnu of log-density (derivation from Student-t log-pdf):
                const double dgi =
                    0.5 * dg_half_nu1 - 0.5 * dg_half_nu - 1.0 / (2.0 * new_nu) -
                    0.5 * log_term + 0.5 * (new_nu + 1.0) * frac / new_nu;

                g += w * dgi;

                // Approximate second derivative (diagonal Newton, robust
                // enough given the small dimensionality and few iterations).
                const double d2gi =
                    0.25 * tg_half_nu1 - 0.25 * tg_half_nu + 1.0 / (2.0 * new_nu * new_nu);
                h += w * d2gi;
            }
            g /= sum_w;
            h /= sum_w;

            if (std::abs(h) < 1e-10) break;
            double step = g / h;
            // Damp + clamp to keep nu in a sane, positive range.
            double candidate = new_nu - step;
            candidate = std::max(2.01, std::min(candidate, 200.0));
            if (std::abs(candidate - new_nu) < 1e-8) {
                new_nu = candidate;
                break;
            }
            new_nu = candidate;
        }

        mu = new_mu;
        sigma = new_sigma;
        nu = new_nu;

        const double delta = std::abs(mu - prev_mu) + std::abs(sigma - prev_sigma) +
                              std::abs(nu - prev_nu);
        prev_mu = mu;
        prev_sigma = sigma;
        prev_nu = nu;
        if (delta < kTol) break;
    }

    EmissionParams out;
    out.mu = mu;
    out.sigma = sigma;
    out.nu = nu;
    return out;
}

}  // namespace hsmm
