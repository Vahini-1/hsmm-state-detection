#!/usr/bin/env python3
"""
Phase 5 (extension): fit a regime-conditioned MS-VAR on an asset's log
returns plus one or more macro/cross-asset factor series, using the
already-fitted HSMM's decoded regime path (see fit_offline_hsmm.py) as the
conditioning state. Computes and plots orthogonalized impulse response
functions per regime.

See hsmm_regime/ms_var.py's module docstring for why this is
regime-*conditioned* VAR rather than full joint MS-VAR EM estimation, and
why the regime path used here (smoothed/offline) is appropriate for this
descriptive/IRF analysis but must not be fed into the Phase 4 backtester.

Usage:
    python python/scripts/fit_ms_var.py --ticker SPY --macro-ticker TLT
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import hsmm_regime as hr
from hsmm_regime import ms_var
from hsmm_regime.data import PROCESSED_DATA_DIR
from hsmm_regime.plotting import plot_impulse_responses

MODELS_DIR = Path("data/processed/models")
FIGURES_DIR = Path("docs/figures")
MSVAR_OUT_DIR = Path("data/processed/msvar")


def load_fitted_params(model_path: Path):
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    params = hr._core.HSMMParams()
    params.n_states = payload["n_states"]
    params.transition = np.asarray(payload["transition"])
    params.emissions = [hr._core.EmissionParams(mu, sigma, nu) for mu, sigma, nu in payload["emissions"]]
    params.durations = [hr._core.DurationParams(r, p, md) for r, p, md in payload["durations"]]
    params.initial_dist = np.asarray(payload["initial_dist"])
    return params, payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a regime-conditioned MS-VAR and compute impulse response functions."
    )
    parser.add_argument("--ticker", required=True, help="Primary asset (regime source)")
    parser.add_argument("--macro-tickers", nargs="+", required=True,
                         help="One or more additional series (tickers) to include in the VAR")
    parser.add_argument("--lag-order", type=int, default=1)
    parser.add_argument("--min-obs-per-regime", type=int, default=50)
    parser.add_argument("--irf-horizon", type=int, default=20)
    parser.add_argument("--no-orthogonalize", action="store_true")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--out-dir", default=str(MSVAR_OUT_DIR))
    parser.add_argument("--fig-dir", default=str(FIGURES_DIR))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model_path = Path(args.models_dir) / f"{args.ticker}_hsmm.pkl"
    if not model_path.exists():
        print(f"No fitted HSMM found at {model_path}. Run fit_offline_hsmm.py first.", file=sys.stderr)
        return 1
    params, payload = load_fitted_params(model_path)

    processed_dir = Path(args.processed_dir)
    primary_path = processed_dir / f"{args.ticker}_features.csv"
    if not primary_path.exists():
        print(f"No processed features at {primary_path}. Run build_features.py first.", file=sys.stderr)
        return 1
    primary = pd.read_csv(primary_path, index_col="date", parse_dates=True)["log_return"]

    # Decode the offline (smoothed) regime path -- see module docstring
    # in ms_var.py for why this is the right choice for this analysis
    # specifically, vs. the online posterior used in backtest_strategy.py.
    regime_path = hr.decode_regimes(primary, params)

    series = {args.ticker: primary}
    for macro_ticker in args.macro_tickers:
        macro_path = processed_dir / f"{macro_ticker}_features.csv"
        if not macro_path.exists():
            print(f"No processed features at {macro_path}. Run build_features.py for "
                  f"{macro_ticker} first.", file=sys.stderr)
            return 1
        series[macro_ticker] = pd.read_csv(macro_path, index_col="date", parse_dates=True)["log_return"]

    combined = pd.DataFrame(series).dropna()
    # Re-align regime_path to the (possibly shorter, after dropna) combined
    # index by locating each combined-index date's position in the
    # original primary series' index.
    primary_index_pos = {date: i for i, date in enumerate(primary.index)}
    aligned_regime_path = np.array([regime_path[primary_index_pos[d]] for d in combined.index])

    print(f"Fitting MS-VAR on {list(combined.columns)} "
          f"({len(combined)} obs, {params.n_states} regimes, lag={args.lag_order})...")

    try:
        result = ms_var.fit_ms_var(
            combined, aligned_regime_path, lag_order=args.lag_order,
            min_obs_per_regime=args.min_obs_per_regime,
        )
    except ValueError as e:
        print(f"fit_ms_var failed: {e}", file=sys.stderr)
        return 1

    labels = payload.get("labels") or hr.regime_label_map(params)
    labels = {int(k): v for k, v in labels.items()}

    for k, regime_var in result.regime_params.items():
        label = labels.get(k, f"Regime {k}")
        print(f"\n=== {label} (n_obs={regime_var.n_obs}) ===")
        print("Intercept:", regime_var.intercept)
        for l, A_l in enumerate(regime_var.coefficients):
            print(f"A_{l+1}:\n{A_l}")
        print("Residual covariance (Sigma):\n", regime_var.sigma)

    irfs = ms_var.compare_regime_dynamics(
        result, horizon=args.irf_horizon, orthogonalize=not args.no_orthogonalize,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, irf_array in irfs.items():
        np.save(out_dir / f"{args.ticker}_msvar_irf_regime{k}.npy", irf_array)

    summary = {
        "ticker": args.ticker,
        "macro_tickers": args.macro_tickers,
        "variable_names": result.variable_names,
        "lag_order": result.lag_order,
        "n_obs_per_regime": {int(k): v.n_obs for k, v in result.regime_params.items()},
        "orthogonalized": not args.no_orthogonalize,
        "cholesky_ordering_note": (
            "Orthogonalized IRFs depend on variable ordering "
            f"{result.variable_names}; the first variable is assumed to "
            "affect all others contemporaneously but not vice versa."
        ),
    }
    with open(out_dir / f"{args.ticker}_msvar_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved MS-VAR results to {out_dir}/")

    if not args.no_plot:
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        # One figure per shock source, so IRFs are readable rather than a
        # single dense n_vars x n_vars x n_regimes grid.
        for shock_idx, shock_var in enumerate(result.variable_names):
            fig = plot_impulse_responses(
                irfs, result.variable_names, state_labels=labels, shock_variable=shock_idx,
            )
            fig_path = fig_dir / f"{args.ticker}_msvar_irf_{shock_var}_shock.png"
            fig.savefig(fig_path, dpi=150)
            print(f"Saved {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
