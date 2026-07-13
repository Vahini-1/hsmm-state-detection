#!/usr/bin/env python3
"""
Phase 4: backtest a regime-conditioned strategy against buy-and-hold and
trend-following baselines, using the online particle-filter posterior
(the only regime signal available strictly without look-ahead — the
offline EM/gamma posterior uses the full sample and is not a valid
backtest input).

Usage:
    python python/scripts/backtest_strategy.py --ticker SPY --strategy regime_momentum
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import hsmm_regime as hr
from hsmm_regime.data import PROCESSED_DATA_DIR
from hsmm_regime.strategy import run_regime_backtest, compute_metrics
from hsmm_regime.plotting import plot_strategy_performance

ONLINE_OUT_DIR = Path("data/processed/online")
MODELS_DIR = Path("data/processed/models")
RESULTS_DIR = Path("data/processed/backtests")
FIGURES_DIR = Path("docs/figures")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest a regime-conditioned strategy.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--strategy", default="regime_momentum",
                         choices=["regime_momentum", "regime_risk_parity"])
    parser.add_argument("--bull-threshold", type=float, default=0.80)
    parser.add_argument("--crash-threshold", type=float, default=0.75)
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--trend-lookback", type=int, default=60)
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--online-dir", default=str(ONLINE_OUT_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--fig-dir", default=str(FIGURES_DIR))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    posterior_path = Path(args.online_dir) / f"{args.ticker}_online_posterior.csv"
    if not posterior_path.exists():
        print(f"No online posterior found at {posterior_path}. Run run_online_filter.py first.",
              file=sys.stderr)
        return 1
    posterior_df = pd.read_csv(posterior_path, index_col=0, parse_dates=True)
    posterior_df.columns = [int(c) for c in posterior_df.columns]

    meta_path = Path(args.models_dir) / f"{args.ticker}_hsmm_meta.json"
    labels = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        labels = {int(k): v for k, v in meta.get("labels", {}).items()}

    features_path = Path(args.processed_dir) / f"{args.ticker}_features.csv"
    if not features_path.exists():
        print(f"No processed features found at {features_path}. Run build_features.py first.",
              file=sys.stderr)
        return 1
    features = pd.read_csv(features_path, index_col="date", parse_dates=True)
    asset_returns = features["log_return"].reindex(posterior_df.index).apply(np.expm1)
    # np.expm1 converts log return -> simple return, since the strategy
    # backtester compounds simple returns (positions * simple_return).

    bull_state = max(labels, key=lambda k: labels[k] == "Low-Vol / Bull") if labels else 1
    crash_state = max(labels, key=lambda k: labels[k] == "High-Vol / Crash") if labels else 0
    # Fallback if label heuristic didn't actually find distinct roles:
    if bull_state == crash_state:
        n_states = posterior_df.shape[1]
        bull_state, crash_state = (n_states - 1), 0

    print(f"Using bull_state={bull_state} crash_state={crash_state} "
          f"(labels={labels or 'none found, using default 0/last'})")

    strategies_to_run = {
        "buy_and_hold": dict(strategy="buy_and_hold"),
        "trend_following": dict(strategy="trend_following", lookback=args.trend_lookback),
    }
    if args.strategy == "regime_momentum":
        strategies_to_run["regime_momentum"] = dict(
            strategy="regime_momentum", bull_state=bull_state, crash_state=crash_state,
            bull_threshold=args.bull_threshold, crash_threshold=args.crash_threshold,
        )
    else:
        strategies_to_run["regime_risk_parity"] = dict(
            strategy="regime_risk_parity", realized_vol=features["realized_vol"].reindex(posterior_df.index),
            target_vol=args.target_vol, crash_state=crash_state, crash_threshold=args.crash_threshold,
        )

    results = {}
    for name, kwargs in strategies_to_run.items():
        results[name] = run_regime_backtest(
            asset_returns, posterior_df, transaction_cost_bps=args.transaction_cost_bps, **kwargs
        )

    print("\n=== Backtest Results ===")
    summary_rows = []
    for name, res in results.items():
        row = {"strategy": name, **res.metrics}
        summary_rows.append(row)
        print(f"\n{name}:")
        for metric, value in res.metrics.items():
            print(f"  {metric}: {value:.4f}")

    summary_df = pd.DataFrame(summary_rows).set_index("strategy")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.ticker}_{args.strategy}_summary.csv"
    summary_df.to_csv(summary_path)
    print(f"\nSaved summary to {summary_path}")

    for name, res in results.items():
        res.equity_curve.to_csv(out_dir / f"{args.ticker}_{name}_equity.csv")

    if not args.no_plot:
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        equity_curves = {name: res.equity_curve for name, res in results.items()}
        fig = plot_strategy_performance(
            equity_curves, title=f"{args.ticker}: {args.strategy} vs Baselines"
        )
        fig_path = fig_dir / f"{args.ticker}_{args.strategy}_performance.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Saved plot to {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
