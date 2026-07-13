#!/usr/bin/env python3
"""
Phase 2b (extension): fit the offline HSMM via Gibbs sampling (FFBS +
conjugate/Metropolis updates), producing posterior samples and credible
intervals rather than a single EM point estimate.

See hsmm_regime/gibbs.py's module docstring for why this is a
Metropolis-within-Gibbs sampler at fixed K rather than full reversible-
jump MCMC over K, and for the Gaussian- (not Student-t-) emission
tradeoff made to keep the sampler conjugate and tractable to verify.

Runtime note: this pure-Python FFBS implementation is meant for
exploratory/small-sample Bayesian analysis (hundreds of observations,
a few thousand iterations complete in minutes) rather than production-
scale fitting over years of daily data -- see the timing note printed at
the end of this script. For large samples, prefer fit_offline_hsmm.py
(the C++ EM engine) for the point estimate, and use this script on a
representative subsample or a shorter recent window to get credible
intervals around it.

Usage:
    python python/scripts/fit_gibbs_hsmm.py --ticker SPY --n-states 2 --n-iter 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hsmm_regime.data import load_primary_series, FeatureConfig, PROCESSED_DATA_DIR
from hsmm_regime.gibbs import run_gibbs_sampler, posterior_summary, estimate_marginal_likelihood, GibbsPriors

MODELS_DIR = Path("data/processed/models")
FIGURES_DIR = Path("docs/figures")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit an HSMM via Gibbs sampling (MCMC).")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--n-states", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=3000)
    parser.add_argument("--n-burn-in", type=int, default=1000)
    parser.add_argument("--max-duration", type=int, default=100)
    parser.add_argument("--max-obs", type=int, default=1000,
                         help="Subsample to at most this many most-recent observations "
                              "(pure-Python FFBS does not scale to full multi-year daily "
                              "history at reasonable runtime -- see module docstring).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--out-dir", default=str(MODELS_DIR))
    parser.add_argument("--primary-series", default="log_return",
                         choices=["log_return", "realized_vol"])
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
        print(f"fit_gibbs_hsmm failed: {e}", file=sys.stderr)
        return 1

    if len(series) > args.max_obs:
        print(f"Subsampling to the most recent {args.max_obs} of {len(series)} observations "
              f"(pure-Python FFBS runtime scales with T; use --max-obs to adjust).")
        series = series.iloc[-args.max_obs:]

    print(f"Running Gibbs sampler: {args.n_states} states, {args.n_iter} iterations "
          f"({args.n_burn_in} burn-in), {len(series)} observations...")

    t0 = time.time()
    result = run_gibbs_sampler(
        series.values, n_states=args.n_states, n_iter=args.n_iter, n_burn_in=args.n_burn_in,
        max_duration=args.max_duration, seed=args.seed, verbose=args.verbose,
    )
    elapsed = time.time() - t0

    print(f"\nSampling complete in {elapsed:.1f}s "
          f"({1000 * elapsed / args.n_iter:.1f}ms/iteration, {len(result.samples)} post-burn-in samples)")
    print(f"Duration-parameter Metropolis acceptance rate: {result.acceptance_rate_duration:.3f} "
          f"(healthy range is roughly 0.2-0.6; well outside that suggests the proposal "
          f"step sizes in GibbsPriors need adjusting)")

    summary = posterior_summary(result, n_states=args.n_states)
    print("\n=== Posterior summary ===")
    for k in range(args.n_states):
        print(f"\nRegime {k}:")
        for param, stats_dict in summary[k].items():
            print(f"  {param}: mean={stats_dict['mean']:.5f}  "
                  f"95% CI=[{stats_dict['ci95'][0]:.5f}, {stats_dict['ci95'][2]:.5f}]")

    ml_estimate = estimate_marginal_likelihood(result)
    print(f"\nHarmonic-mean marginal log-likelihood estimate: {ml_estimate:.2f}")
    print("(a coarse model-comparison diagnostic across different --n-states choices; "
          "see gibbs.py's estimate_marginal_likelihood docstring for its known limitations)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.ticker}_gibbs_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "ticker": args.ticker,
            "n_states": args.n_states,
            "n_iter": args.n_iter,
            "n_burn_in": args.n_burn_in,
            "n_obs": len(series),
            "acceptance_rate_duration": result.acceptance_rate_duration,
            "marginal_log_likelihood_estimate": ml_estimate,
            "posterior_summary": {
                int(k): {p: {"mean": v["mean"], "ci95": v["ci95"]} for p, v in regime.items()}
                for k, regime in summary.items()
            },
        }, f, indent=2)
    print(f"\nSaved posterior summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
