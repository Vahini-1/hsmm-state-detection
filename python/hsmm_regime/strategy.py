"""
Regime-conditioned trading strategies and a vectorized, no-look-ahead
backtester.

Core discipline enforced throughout this module: the *signal* used to size
a position on day t is only ever a function of information available at
the close of day t-1 (regime posterior computed from data up to and
including t-1). Returns are then realized from t-1 close to t close. This
module structures the computation to make that shift explicit and hard to
get backwards by accident, rather than relying on the caller to remember
to shift things correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series          # growth of $1, indexed like returns
    strategy_returns: pd.Series      # realized daily strategy returns
    positions: pd.Series             # position (exposure) held each day
    metrics: dict[str, float]


def _annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    growth = (1.0 + daily_returns).prod()
    n = len(daily_returns)
    if n == 0 or growth <= 0:
        return float("nan")
    return growth ** (periods_per_year / n) - 1.0


def _annualized_vol(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    return daily_returns.std() * np.sqrt(periods_per_year)


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()


_DEGENERATE_STD_TOL = 1e-12  # below this, treat volatility as numerically zero


def _sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.0,
                   periods_per_year: int = 252) -> float:
    excess = daily_returns - risk_free / periods_per_year
    std = excess.std()
    if np.isnan(std) or std < _DEGENERATE_STD_TOL:
        return float("nan")
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def _sortino_ratio(daily_returns: pd.Series, risk_free: float = 0.0,
                    periods_per_year: int = 252) -> float:
    excess = daily_returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std()
    if np.isnan(downside_std) or downside_std < _DEGENERATE_STD_TOL:
        return float("nan")
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def compute_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Standard performance metrics for a realized daily-return series."""
    equity = (1.0 + daily_returns).cumprod()
    return {
        "annualized_return": _annualized_return(daily_returns, periods_per_year),
        "annualized_vol": _annualized_vol(daily_returns, periods_per_year),
        "sharpe_ratio": _sharpe_ratio(daily_returns, periods_per_year=periods_per_year),
        "sortino_ratio": _sortino_ratio(daily_returns, periods_per_year=periods_per_year),
        "max_drawdown": _max_drawdown(equity),
        "total_return": equity.iloc[-1] - 1.0 if len(equity) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Position-sizing rules. Each takes the *lagged* regime posterior (already
# shifted by the caller / backtest harness so index t reflects information
# known at the close of t-1) and asset returns, and returns a position
# series (exposure, e.g. 1.0 = fully long, -1.0 = fully short, 0 = cash).
# ---------------------------------------------------------------------------

def regime_momentum_positions(
    regime_posterior: pd.DataFrame,
    bull_state: int,
    crash_state: int,
    bull_threshold: float = 0.80,
    crash_threshold: float = 0.75,
) -> pd.Series:
    """
    Phase 4 spec:
      - P(next = crash) > crash_threshold -> liquidate to cash (position 0)
      - P(next = bull) > bull_threshold -> max leverage on momentum (position 1)
      - otherwise -> neutral/partial exposure scaled by bull-state confidence

    `regime_posterior` must already be lagged (i.e. row t is the posterior
    computed using data up to t-1) before being passed in here; this
    function performs no shifting itself so the lag discipline lives in one
    place (see `run_regime_backtest`).
    """
    positions = regime_posterior[bull_state].copy()
    positions.name = "position"

    crash_mask = regime_posterior[crash_state] > crash_threshold
    bull_mask = regime_posterior[bull_state] > bull_threshold

    positions = positions.where(~bull_mask, 1.0)
    positions = positions.where(~crash_mask, 0.0)
    # Anywhere neither threshold triggers, positions already holds
    # P(bull), giving smooth exposure scaling with regime confidence
    # rather than a hard neutral/binary jump.
    return positions.clip(0.0, 1.0)


def regime_risk_parity_positions(
    regime_posterior: pd.DataFrame,
    realized_vol: pd.Series,
    target_vol: float = 0.10,
    crash_state: int | None = None,
    crash_threshold: float = 0.75,
    max_leverage: float = 2.0,
) -> pd.Series:
    """
    Simple regime-aware risk-parity sizing: scale exposure inversely with
    (lagged) realized volatility to target a constant annualized vol,
    then additionally de-risk to cash when posterior crash probability
    crosses `crash_threshold`. `realized_vol` must already be lagged, same
    discipline as regime_posterior.
    """
    vol = realized_vol.replace(0, np.nan)
    positions = (target_vol / vol).clip(upper=max_leverage).fillna(0.0)
    positions.name = "position"

    if crash_state is not None:
        crash_mask = regime_posterior[crash_state] > crash_threshold
        positions = positions.where(~crash_mask, 0.0)

    return positions


def run_regime_backtest(
    asset_returns: pd.Series,
    regime_posterior: pd.DataFrame,
    strategy: str = "regime_momentum",
    transaction_cost_bps: float = 1.0,
    **strategy_kwargs,
) -> BacktestResult:
    """
    Vectorized backtest with an explicit, single point of lag application:
    regime_posterior is shifted forward by one period here (row t becomes
    the posterior that was actually knowable at the close of t-1, i.e. the
    posterior computed from y_1..t-1), and everything downstream —
    position sizing and return realization — operates strictly on
    already-lagged information. This is the one place look-ahead bugs
    would hide, so it is deliberately centralized rather than left to each
    strategy function.

    `asset_returns` and `regime_posterior` must share the same index
    (dates), with regime_posterior columns 0..K-1 giving P(z_t = k | y_1:t)
    (the *unlagged*, "as of time t" posterior — lagging happens inside this
    function, not before it).
    """
    if not asset_returns.index.equals(regime_posterior.index):
        raise ValueError("asset_returns and regime_posterior must share the same index")

    lagged_posterior = regime_posterior.shift(1)

    if strategy == "regime_momentum":
        bull_state = strategy_kwargs.pop("bull_state")
        crash_state = strategy_kwargs.pop("crash_state")
        positions = regime_momentum_positions(
            lagged_posterior, bull_state=bull_state, crash_state=crash_state, **strategy_kwargs
        )
    elif strategy == "regime_risk_parity":
        realized_vol = strategy_kwargs.pop("realized_vol")
        lagged_vol = realized_vol.shift(1)
        positions = regime_risk_parity_positions(
            lagged_posterior, realized_vol=lagged_vol, **strategy_kwargs
        )
    elif strategy == "buy_and_hold":
        positions = pd.Series(1.0, index=asset_returns.index)
    elif strategy == "trend_following":
        # Simple, standard baseline: long if trailing N-day return is
        # positive, computed strictly on data available at t-1 (the
        # rolling window itself is already only looking backward, and we
        # additionally shift by 1 to avoid using today's return to decide
        # today's position).
        lookback = strategy_kwargs.pop("lookback", 60)
        trailing = asset_returns.rolling(lookback).apply(lambda r: (1 + r).prod() - 1, raw=False)
        positions = (trailing.shift(1) > 0).astype(float)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    positions = positions.reindex(asset_returns.index).fillna(0.0)

    # Transaction costs charged on position *changes*, applied to the
    # return realized that day (a simplification, but conservative in that
    # it front-loads the cost rather than deferring it).
    position_changes = positions.diff().abs().fillna(positions.abs())
    costs = position_changes * (transaction_cost_bps / 1e4)

    strategy_returns = positions * asset_returns - costs
    strategy_returns.name = "strategy_return"

    equity_curve = (1.0 + strategy_returns).cumprod()
    equity_curve.name = "equity"

    metrics = compute_metrics(strategy_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        positions=positions,
        metrics=metrics,
    )
