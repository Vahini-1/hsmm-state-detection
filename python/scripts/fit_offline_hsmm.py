#!/usr/bin/env python3
"""
Phase 2: fit the offline HSMM (forward-backward + EM) to a ticker's
processed return series, decode the most-likely regime path, and save a
validation plot overlaying regimes on price.

Usage:
    python python/scripts/fit_offline_hsmm.py --ticker SPY --n-states 3
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
from hsmm_regime.data import load_primary_series, FeatureConfig, PROCESSED_DATA_DIR
from hsmm_regime.plotting import (
    plot_regime_overlay,
    plot_duration_diagnostics,
    plot_transition_matrix,
)

MODELS_DIR = Path("data/processed/models")
FIGURES_DIR = Path("docs/figures")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit an offline HSMM via EM.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--max-duration", type=int, default=252)
    parser.add_argument("--em-max-iters", type=int, default=200)
    parser.add_argument("--em-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--primary-series", default="log_return",
                         choices=["log_return", "realized_vol"])
    parser.add_argument("--out-dir", default=str(MODELS_DIR))
    parser.add_argument("--fig-dir", default=str(FIGURES_DIR))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    feat_config = FeatureConfig(primary_series=args.primary_series)
    try:
        series = load_primary_series(args.ticker, args.processed_dir, feat_config)
    except (FileNotFoundError, KeyError) as e:
        print(f"fit_offline_hsmm failed: {e}", file=sys.stderr)
        return 1

    hsmm_config = hr.HSMMConfig(
        n_states=args.n_states,
        max_duration=args.max_duration,
        em_max_iters=args.em_max_iters,
        em_tol=args.em_tol,
        em_seed=args.seed,
        verbose=args.verbose,
    )

    print(f"Fitting {args.n_states}-state HSMM on {args.ticker} "
          f"({args.primary_series}, {len(series)} obs)...")
    result = hr.fit_offline(series, hsmm_config)

    print(f"Converged: {result.converged}  "
          f"Final log-likelihood: {result.log_likelihood_history[-1]:.4f}  "
          f"Iterations: {len(result.log_likelihood_history)}")

    for k, emission in enumerate(result.params.emissions):
        dur = result.params.durations[k]
        expected_dur = dur.r * (1 - dur.p) / dur.p + 1  # NB mean on shifted support
        print(f"  Regime {k}: mu={emission.mu:.5f} sigma={emission.sigma:.5f} "
              f"nu={emission.nu:.2f}  E[duration]~{expected_dur:.1f} days")

    path = hr.decode_regimes(series, result.params)
    labels = hr.regime_label_map(result.params)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.ticker}_hsmm.pkl"
    with open(model_path, "wb") as f:
        # HSMMParams (a pybind11 class) isn't directly picklable, so persist
        # a plain-Python dict of its fields instead; the fit script's own
        # reconstruction is the canonical "load" path (see run_online_filter.py).
        payload = {
            "n_states": result.params.n_states,
            "transition": np.asarray(result.params.transition),
            "emissions": [(e.mu, e.sigma, e.nu) for e in result.params.emissions],
            "durations": [(d.r, d.p, d.max_duration) for d in result.params.durations],
            "initial_dist": np.asarray(result.params.initial_dist),
            "log_likelihood_history": result.log_likelihood_history,
            "converged": result.converged,
            "ticker": args.ticker,
            "primary_series": args.primary_series,
        }
        pickle.dump(payload, f)
    print(f"Saved fitted params to {model_path}")

    meta_path = out_dir / f"{args.ticker}_hsmm_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "ticker": args.ticker,
            "n_states": args.n_states,
            "converged": result.converged,
            "final_log_likelihood": result.log_likelihood_history[-1],
            "labels": labels,
        }, f, indent=2)

    if not args.no_plot:
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        price_like = series if args.primary_series != "log_return" else series.cumsum()
        price_like.name = args.ticker

        fig1 = plot_regime_overlay(price_like, path, args.n_states, state_labels=labels,
                                     title=f"{args.ticker}: Decoded Regimes ({args.primary_series})")
        fig1.savefig(fig_dir / f"{args.ticker}_regime_overlay.png", dpi=150)

        fig2 = plot_duration_diagnostics(result.params.durations, state_labels=labels)
        fig2.savefig(fig_dir / f"{args.ticker}_duration_diagnostics.png", dpi=150)

        fig3 = plot_transition_matrix(np.asarray(result.params.transition), state_labels=labels)
        fig3.savefig(fig_dir / f"{args.ticker}_transition_matrix.png", dpi=150)

        print(f"Saved validation plots to {fig_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
