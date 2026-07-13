"""
Tests for hsmm_regime.strategy: no-look-ahead correctness of the vectorized
backtester and sanity checks on the performance metrics it produces.

The no-look-ahead tests are the most important ones in this file: they
directly verify that a position on day t is a function only of information
knowable at the close of day t-1 (see docstring in strategy.py). This is
enforced by (1) unit-testing the shift discipline explicitly, and (2) a
perturbation test that changes only *future* data and asserts past
positions are completely unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsmm_regime import strategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dates():
    return pd.date_range("2020-01-01", periods=100, freq="B")


@pytest.fixture
def asset_returns(dates):
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates, name="ret")


@pytest.fixture
def block_posterior(dates):
    """
    A 2-state posterior that flips between confidently-bull and
    confidently-crash every 20 days, matching the style used in the
    strategy sanity script from batch 3.
    """
    K = 2
    post = np.zeros((len(dates), K))
    state = 0
    i = 0
    while i < len(dates):
        block = min(20, len(dates) - i)
        if state == 0:
            post[i:i + block, 0] = 0.15
            post[i:i + block, 1] = 0.85
        else:
            post[i:i + block, 0] = 0.85
            post[i:i + block, 1] = 0.15
        i += block
        state = 1 - state
    return pd.DataFrame(post, index=dates, columns=[0, 1])


# ---------------------------------------------------------------------------
# No-look-ahead tests
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    def test_position_at_t_matches_manually_lagged_posterior(self, asset_returns, block_posterior):
        """Position at day t should be derivable purely from
        block_posterior.shift(1) at day t, for the regime_momentum rule."""
        result = strategy.run_regime_backtest(
            asset_returns, block_posterior, strategy="regime_momentum",
            bull_state=1, crash_state=0, bull_threshold=0.8, crash_threshold=0.8,
        )

        lagged = block_posterior.shift(1)
        expected = strategy.regime_momentum_positions(
            lagged, bull_state=1, crash_state=0, bull_threshold=0.8, crash_threshold=0.8,
        ).reindex(asset_returns.index).fillna(0.0)

        pd.testing.assert_series_equal(
            result.positions, expected, check_names=False, atol=1e-12
        )

    def test_perturbing_future_posterior_does_not_change_past_positions(
        self, asset_returns, block_posterior
    ):
        """
        Changing the regime posterior on day t+5 onward must not change
        the position held on any day <= t. This is the strongest possible
        black-box test for look-ahead bugs: it doesn't assume anything
        about the internal shift implementation, only about the causal
        structure the backtester is supposed to guarantee.
        """
        split_idx = 50

        perturbed = block_posterior.copy()
        # Flip the posterior completely for everything from split_idx onward.
        perturbed.iloc[split_idx:] = 1.0 - perturbed.iloc[split_idx:].values

        original_result = strategy.run_regime_backtest(
            asset_returns, block_posterior, strategy="regime_momentum",
            bull_state=1, crash_state=0,
        )
        perturbed_result = strategy.run_regime_backtest(
            asset_returns, perturbed, strategy="regime_momentum",
            bull_state=1, crash_state=0,
        )

        # Positions strictly before split_idx must be untouched. Position
        # at exactly split_idx depends on posterior at split_idx - 1 (due
        # to the shift), which is also untouched, so it must match too;
        # only split_idx + 1 onward is allowed to diverge.
        pd.testing.assert_series_equal(
            original_result.positions.iloc[: split_idx + 1],
            perturbed_result.positions.iloc[: split_idx + 1],
            check_names=False,
        )

    def test_perturbing_future_returns_does_not_change_past_strategy_returns(
        self, asset_returns, block_posterior
    ):
        """Changing asset_returns after day t must not affect realized
        strategy returns on or before day t (positions are lagged, and
        realization on day t only uses return_t, not future returns)."""
        split_idx = 60
        perturbed_returns = asset_returns.copy()
        perturbed_returns.iloc[split_idx:] = -perturbed_returns.iloc[split_idx:]

        original_result = strategy.run_regime_backtest(
            asset_returns, block_posterior, strategy="regime_momentum",
            bull_state=1, crash_state=0,
        )
        perturbed_result = strategy.run_regime_backtest(
            perturbed_returns, block_posterior, strategy="regime_momentum",
            bull_state=1, crash_state=0,
        )

        pd.testing.assert_series_equal(
            original_result.strategy_returns.iloc[:split_idx],
            perturbed_result.strategy_returns.iloc[:split_idx],
            check_names=False,
        )

    def test_first_day_position_is_flat_due_to_missing_lag(self, asset_returns, block_posterior):
        """On the very first day there is no t-1 posterior to lag from, so
        the position must be the safe default (0), never inferred from
        day-0's own (unlagged) posterior."""
        result = strategy.run_regime_backtest(
            asset_returns, block_posterior, strategy="regime_momentum",
            bull_state=1, crash_state=0,
        )
        assert result.positions.iloc[0] == 0.0

    def test_trend_following_does_not_use_same_day_return(self, dates):
        """A single-day return spike on day t should not be visible in
        that day's trend-following position, only from t+1 onward."""
        returns = pd.Series(0.0, index=dates)
        returns.iloc[30] = 0.5  # a huge one-day spike

        post = pd.DataFrame(0.5, index=dates, columns=[0, 1])  # unused by this strategy

        result = strategy.run_regime_backtest(
            returns, post, strategy="trend_following", lookback=10,
        )
        # The lookback window at day 30 (computed on returns[21:31], which
        # includes the spike since rolling windows look backward) combined
        # with the extra shift(1) means the spike can only first influence
        # a position starting at day 31, never at day 30 itself.
        trailing_incl_spike = (1 + returns.iloc[21:31]).prod() - 1
        assert trailing_incl_spike > 0.4  # sanity: spike is indeed in that window
        assert result.positions.iloc[30] == 0.0  # day 30's position predates the spike's visibility


# ---------------------------------------------------------------------------
# Metric correctness / sanity
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_buy_and_hold_position_is_always_fully_invested(self, asset_returns, block_posterior):
        result = strategy.run_regime_backtest(asset_returns, block_posterior, strategy="buy_and_hold")
        assert (result.positions == 1.0).all()

    def test_flat_positions_produce_zero_return_and_flat_equity(self, dates):
        returns = pd.Series(0.01, index=dates)
        post = pd.DataFrame(0.0, index=dates, columns=[0, 1])
        post[0] = 1.0  # always "crash" state -> regime_momentum should stay flat

        result = strategy.run_regime_backtest(
            returns, post, strategy="regime_momentum", bull_state=1, crash_state=0,
            crash_threshold=0.5,
        )
        assert (result.positions == 0.0).all()
        assert (result.strategy_returns == 0.0).all()
        assert (result.equity_curve == 1.0).all()

    def test_known_deterministic_returns_give_known_sharpe(self, dates):
        # Constant positive daily return with zero volatility is a
        # degenerate case for Sharpe (std=0); compute_metrics should
        # return NaN rather than raising or dividing silently to inf.
        returns = pd.Series(0.001, index=dates)
        metrics = strategy.compute_metrics(returns)
        assert np.isnan(metrics["sharpe_ratio"])
        assert metrics["annualized_return"] > 0

    def test_max_drawdown_is_non_positive(self, asset_returns, block_posterior):
        result = strategy.run_regime_backtest(
            asset_returns, block_posterior, strategy="regime_momentum",
            bull_state=1, crash_state=0,
        )
        assert result.metrics["max_drawdown"] <= 0.0

    def test_transaction_costs_reduce_returns_relative_to_zero_cost(self, asset_returns, block_posterior):
        zero_cost = strategy.run_regime_backtest(
            asset_returns, block_posterior, strategy="regime_momentum",
            bull_state=1, crash_state=0, transaction_cost_bps=0.0,
        )
        with_cost = strategy.run_regime_backtest(
            asset_returns, block_posterior, strategy="regime_momentum",
            bull_state=1, crash_state=0, transaction_cost_bps=50.0,  # 50bps, deliberately large
        )
        assert with_cost.equity_curve.iloc[-1] <= zero_cost.equity_curve.iloc[-1]

    def test_mismatched_index_raises(self, asset_returns):
        bad_posterior = pd.DataFrame(
            0.5, index=asset_returns.index[:-1], columns=[0, 1]
        )  # deliberately shorter index
        with pytest.raises(ValueError):
            strategy.run_regime_backtest(
                asset_returns, bad_posterior, strategy="regime_momentum",
                bull_state=1, crash_state=0,
            )

    def test_unknown_strategy_raises(self, asset_returns, block_posterior):
        with pytest.raises(ValueError):
            strategy.run_regime_backtest(asset_returns, block_posterior, strategy="not_a_real_strategy")


# ---------------------------------------------------------------------------
# Risk-parity specific checks
# ---------------------------------------------------------------------------

class TestRiskParity:
    def test_higher_realized_vol_gives_lower_exposure(self, dates):
        returns = pd.Series(0.0005, index=dates)
        post = pd.DataFrame(0.0, index=dates, columns=[0, 1])

        low_vol = pd.Series(0.05, index=dates)
        high_vol = pd.Series(0.40, index=dates)

        result_low_vol = strategy.run_regime_backtest(
            returns, post, strategy="regime_risk_parity",
            realized_vol=low_vol, target_vol=0.10, crash_state=None,
        )
        result_high_vol = strategy.run_regime_backtest(
            returns, post, strategy="regime_risk_parity",
            realized_vol=high_vol, target_vol=0.10, crash_state=None,
        )

        # Compare positions from day 2 onward (day 0 is always flat due to
        # the lag; day 1 depends on day-0 vol which is identical warm-up).
        assert (result_low_vol.positions.iloc[2:] >= result_high_vol.positions.iloc[2:]).all()

    def test_exposure_is_capped_at_max_leverage(self, dates):
        returns = pd.Series(0.0, index=dates)
        post = pd.DataFrame(0.0, index=dates, columns=[0, 1])
        tiny_vol = pd.Series(1e-6, index=dates)  # would imply huge leverage uncapped

        result = strategy.run_regime_backtest(
            returns, post, strategy="regime_risk_parity",
            realized_vol=tiny_vol, target_vol=0.10, crash_state=None, max_leverage=2.0,
        )
        assert (result.positions <= 2.0 + 1e-9).all()
