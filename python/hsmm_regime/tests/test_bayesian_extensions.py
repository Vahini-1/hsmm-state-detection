"""
Tests for hsmm_regime.gibbs (FFBS + Metropolis-within-Gibbs sampler) and
hsmm_regime.ms_var (regime-conditioned VAR + impulse responses).

The FFBS cross-validation test is the most important one here: it checks
the Gibbs sampler's path-sampling step against the independently-verified
C++ forward-backward engine's exact posterior, which is about as strong a
correctness guarantee as is practical without a second from-scratch
reference implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsmm_regime import _core
from hsmm_regime.gibbs import (
    ffbs_sample_path,
    _duration_log_pmf_vector,
    run_gibbs_sampler,
    posterior_summary,
    estimate_marginal_likelihood,
)
from hsmm_regime import ms_var


# ---------------------------------------------------------------------------
# FFBS cross-validation against the C++ engine's exact forward-backward
# ---------------------------------------------------------------------------

class TestFFBSCorrectness:
    def test_ffbs_marginals_match_exact_forward_backward(self):
        rng = np.random.default_rng(0)
        mu = np.array([0.0, 4.0])
        sigma = np.array([0.3, 0.3])
        sigma2 = sigma ** 2

        p = _core.HSMMParams()
        p.n_states = 2
        p.transition = np.array([[0.0, 1.0], [1.0, 0.0]])
        # Large nu -> Student-t approximately Gaussian, so the C++ engine's
        # exact posterior is comparable to the Gibbs sampler's Gaussian
        # emission model.
        p.emissions = [_core.EmissionParams(mu[0], sigma[0], 1000.0),
                       _core.EmissionParams(mu[1], sigma[1], 1000.0)]
        p.durations = [_core.DurationParams(10.0, 0.6, 30), _core.DurationParams(10.0, 0.6, 30)]
        p.initial_dist = np.array([0.5, 0.5])

        obs = np.concatenate([
            rng.normal(0.0, 0.3, 6), rng.normal(4.0, 0.3, 8),
            rng.normal(0.0, 0.3, 5), rng.normal(4.0, 0.3, 7),
        ])
        T = len(obs)

        fb = _core.forward_backward(obs, p)
        exact_gamma = np.exp(fb.log_gamma)

        duration_log_pmf = [_duration_log_pmf_vector(10.0, 0.6, 30) for _ in range(2)]
        transition = np.array([[0.0, 1.0], [1.0, 0.0]])
        initial_dist = np.array([0.5, 0.5])

        n_draws = 3000
        counts = np.zeros((T, 2))
        for _ in range(n_draws):
            path, _ = ffbs_sample_path(obs, mu, sigma2, duration_log_pmf, transition,
                                        initial_dist, rng)
            for t in range(T):
                counts[t, path[t]] += 1
        mc_gamma = counts / n_draws

        max_abs_diff = np.abs(exact_gamma - mc_gamma).max()
        assert max_abs_diff < 0.08, (
            f"FFBS sampled-path marginals (max diff {max_abs_diff:.4f}) do not match "
            f"exact forward-backward posterior"
        )

    def test_ffbs_log_likelihood_matches_exact_forward_backward(self):
        """The log-likelihood FFBS reports as a byproduct of the forward
        filter should exactly match the C++ engine's forward_backward
        result for the same parameters (both compute the same forward
        recursion, just in different languages)."""
        rng = np.random.default_rng(3)
        mu = np.array([0.0, 4.0])
        sigma2 = np.array([0.09, 0.09])

        p = _core.HSMMParams()
        p.n_states = 2
        p.transition = np.array([[0.0, 1.0], [1.0, 0.0]])
        p.emissions = [_core.EmissionParams(0.0, 0.3, 1000.0), _core.EmissionParams(4.0, 0.3, 1000.0)]
        p.durations = [_core.DurationParams(10.0, 0.6, 30), _core.DurationParams(10.0, 0.6, 30)]
        p.initial_dist = np.array([0.5, 0.5])

        obs = np.array([0.1, -0.2, 3.9, 4.1, 0.05, 3.95, -0.1, 4.05])
        fb = _core.forward_backward(obs, p)

        duration_log_pmf = [_duration_log_pmf_vector(10.0, 0.6, 30) for _ in range(2)]
        _, ffbs_log_lik = ffbs_sample_path(
            obs, mu, sigma2, duration_log_pmf, np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0.5, 0.5]), rng,
        )

        # nu=1000 Student-t is a very close but not bit-exact approximation
        # to Gaussian (the two densities differ by design, just by a tiny
        # amount at large nu), so we expect close agreement between the
        # C++ engine's Student-t-based log-likelihood and the Gibbs
        # sampler's exactly-Gaussian one -- not exact equality.
        assert abs(ffbs_log_lik - fb.log_likelihood) < 0.01

    def test_ffbs_path_has_valid_states_and_correct_length(self):
        rng = np.random.default_rng(1)
        mu = np.array([0.0, 4.0, -3.0])
        sigma2 = np.array([0.1, 0.1, 0.1])
        duration_log_pmf = [_duration_log_pmf_vector(3.0, 0.4, 40) for _ in range(3)]
        transition = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        initial_dist = np.array([1 / 3, 1 / 3, 1 / 3])

        obs = rng.normal(0, 1, 50)
        path, log_lik = ffbs_sample_path(obs, mu, sigma2, duration_log_pmf, transition,
                                          initial_dist, rng)

        assert len(path) == len(obs)
        assert set(np.unique(path)).issubset({0, 1, 2})
        assert np.isfinite(log_lik)


# ---------------------------------------------------------------------------
# Gibbs sampler end-to-end: posterior recovery and diagnostics
# ---------------------------------------------------------------------------

class TestGibbsSamplerRecovery:
    @pytest.fixture(scope="class")
    def fitted_result(self):
        rng = np.random.default_rng(5)
        true_mu = [0.0, 4.0]
        true_sigma = [0.3, 0.3]
        obs = []
        state = 0
        for _ in range(10):
            length = rng.integers(15, 30)
            obs.extend(rng.normal(true_mu[state], true_sigma[state], length))
            state = 1 - state
        obs = np.array(obs)

        result = run_gibbs_sampler(obs, n_states=2, n_iter=700, n_burn_in=300,
                                    max_duration=50, seed=2)
        return result, true_mu, true_sigma

    def test_true_means_fall_within_95_credible_intervals(self, fitted_result):
        result, true_mu, _ = fitted_result
        summary = posterior_summary(result, n_states=2)

        for k in range(2):
            ci = summary[k]["mu"]["ci95"]
            closest_true = min(true_mu, key=lambda m: abs(m - summary[k]["mu"]["mean"]))
            assert ci[0] <= closest_true <= ci[2], (
                f"True mu {closest_true} outside regime {k}'s 95% CI {ci}"
            )

    def test_duration_acceptance_rate_is_healthy(self, fitted_result):
        result, _, _ = fitted_result
        # A healthy Metropolis acceptance rate is neither near-0 (proposal
        # too wide / never moves) nor near-1 (proposal too narrow /
        # insufficient exploration).
        assert 0.1 < result.acceptance_rate_duration < 0.9

    def test_posterior_samples_count_matches_post_burn_in(self, fitted_result):
        result, _, _ = fitted_result
        assert len(result.samples) == 700 - 300

    def test_marginal_likelihood_estimate_is_finite(self, fitted_result):
        result, _, _ = fitted_result
        ml = estimate_marginal_likelihood(result)
        assert np.isfinite(ml)

    def test_credible_intervals_are_ordered_correctly(self, fitted_result):
        result, _, _ = fitted_result
        summary = posterior_summary(result, n_states=2)
        for k in range(2):
            for param in ("mu", "sigma", "expected_duration"):
                lo, mid, hi = summary[k][param]["ci95"]
                assert lo <= mid <= hi


class TestGibbsSamplerEdgeCases:
    def test_single_state_runs_without_error(self):
        rng = np.random.default_rng(9)
        obs = rng.normal(0, 1, 40)
        result = run_gibbs_sampler(obs, n_states=1, n_iter=50, n_burn_in=10,
                                    max_duration=40, seed=1)
        assert len(result.samples) == 40
        for s in result.samples:
            assert (s.state_path == 0).all()

    def test_short_series_does_not_crash(self):
        rng = np.random.default_rng(4)
        obs = rng.normal(0, 1, 10)
        result = run_gibbs_sampler(obs, n_states=2, n_iter=30, n_burn_in=5,
                                    max_duration=8, seed=1)
        assert len(result.samples) == 25


# ---------------------------------------------------------------------------
# MS-VAR
# ---------------------------------------------------------------------------

class TestMSVAR:
    @pytest.fixture
    def two_regime_data(self):
        rng = np.random.default_rng(42)
        true_A = {0: np.array([[0.3, 0.05], [0.02, 0.35]]),
                  1: np.array([[-0.4, 0.15], [0.05, 0.25]])}
        true_c = {0: np.array([0.0002, 0.0001]), 1: np.array([-0.0008, 0.0003])}
        true_sigma = {0: np.array([[0.0001, 1e-5], [1e-5, 0.00008]]),
                       1: np.array([[0.0008, 0.0005], [0.0005, 0.0005]])}

        T = 2000
        data = np.zeros((T, 2))
        regime_path = np.zeros(T, dtype=int)
        state, t, y_prev = 0, 0, np.zeros(2)
        while t < T:
            block = min(rng.integers(30, 80), T - t)
            L = np.linalg.cholesky(true_sigma[state])
            for _ in range(block):
                y_t = true_c[state] + true_A[state] @ y_prev + L @ rng.standard_normal(2)
                data[t] = y_t
                regime_path[t] = state
                y_prev = y_t
                t += 1
            state = 1 - state

        df = pd.DataFrame(data, columns=["asset", "macro"])
        return df, regime_path, true_A, true_c

    def test_fit_ms_var_runs_and_returns_all_regimes(self, two_regime_data):
        df, regime_path, _, _ = two_regime_data
        result = ms_var.fit_ms_var(df, regime_path, lag_order=1, min_obs_per_regime=50)
        assert set(result.regime_params.keys()) == {0, 1}
        for k, params in result.regime_params.items():
            assert params.intercept.shape == (2,)
            assert params.coefficients[0].shape == (2, 2)
            assert params.sigma.shape == (2, 2)

    def test_starved_regime_raises_value_error(self, two_regime_data):
        df, regime_path, _, _ = two_regime_data
        with pytest.raises(ValueError):
            ms_var.fit_ms_var(df.iloc[:60], regime_path[:60], lag_order=1, min_obs_per_regime=50)

    def test_mismatched_lengths_raise(self, two_regime_data):
        df, regime_path, _, _ = two_regime_data
        with pytest.raises(ValueError):
            ms_var.fit_ms_var(df, regime_path[:-10], lag_order=1)

    def test_impulse_responses_have_correct_shape(self, two_regime_data):
        df, regime_path, _, _ = two_regime_data
        result = ms_var.fit_ms_var(df, regime_path, lag_order=1, min_obs_per_regime=50)
        irf = ms_var.compute_impulse_responses(result, regime=0, horizon=15)
        assert irf.shape == (16, 2, 2)
        assert np.isfinite(irf).all()

    def test_orthogonalized_horizon_zero_equals_cholesky_factor(self, two_regime_data):
        df, regime_path, _, _ = two_regime_data
        result = ms_var.fit_ms_var(df, regime_path, lag_order=1, min_obs_per_regime=50)
        irf = ms_var.compute_impulse_responses(result, regime=1, horizon=5, orthogonalize=True)
        L = np.linalg.cholesky(result.regime_params[1].sigma)
        assert np.allclose(irf[0], L)

    def test_unknown_regime_raises_key_error(self, two_regime_data):
        df, regime_path, _, _ = two_regime_data
        result = ms_var.fit_ms_var(df, regime_path, lag_order=1, min_obs_per_regime=50)
        with pytest.raises(KeyError):
            ms_var.compute_impulse_responses(result, regime=99)

    def test_coefficients_recovered_within_tolerance_averaged_over_seeds(self):
        """Averaged over several seeds (see the design-note discussion of
        why single-seed point comparisons are statistically fragile for
        masked/regime-conditioned regression), fitted coefficients should
        converge close to the true generating values."""
        true_A = {0: np.array([[0.3, 0.05], [0.02, 0.35]]),
                  1: np.array([[-0.4, 0.15], [0.05, 0.25]])}
        true_c = {0: np.array([0.0002, 0.0001]), 1: np.array([-0.0008, 0.0003])}
        true_sigma = {0: np.array([[0.0001, 1e-5], [1e-5, 0.00008]]),
                       1: np.array([[0.0008, 0.0005], [0.0005, 0.0005]])}

        A0_estimates, A1_estimates = [], []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            T = 2000
            data = np.zeros((T, 2))
            regime_path = np.zeros(T, dtype=int)
            state, t, y_prev = 0, 0, np.zeros(2)
            while t < T:
                block = min(rng.integers(30, 80), T - t)
                L = np.linalg.cholesky(true_sigma[state])
                for _ in range(block):
                    y_t = true_c[state] + true_A[state] @ y_prev + L @ rng.standard_normal(2)
                    data[t] = y_t
                    regime_path[t] = state
                    y_prev = y_t
                    t += 1
                state = 1 - state
            df = pd.DataFrame(data, columns=["asset", "macro"])
            result = ms_var.fit_ms_var(df, regime_path, lag_order=1, min_obs_per_regime=50)
            A0_estimates.append(result.regime_params[0].coefficients[0])
            A1_estimates.append(result.regime_params[1].coefficients[0])

        A0_mean = np.mean(A0_estimates, axis=0)
        A1_mean = np.mean(A1_estimates, axis=0)
        assert np.abs(A0_mean - true_A[0]).max() < 0.04
        assert np.abs(A1_mean - true_A[1]).max() < 0.04
