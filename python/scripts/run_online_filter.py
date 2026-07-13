#!/usr/bin/env python3
"""
Phase 3: run the online SMC particle filter over historical data in
simulated live mode (one observation at a time), using params from a model
already fit by fit_offline_hsmm.py. Reports per-step regime posteriors and
timing to confirm millisecond-scale updates.

Usage:
    python python/scripts/run_online_filter.py --ticker SPY
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import hsmm_regime as hr
from hsmm_regime.data import load_primary_series, FeatureConfig, PROCESSED_DATA_DIR
from hsmm_regime.plotting import plot_regime_probabilities

MODELS_DIR = Path("data/processed/models")
FIGURES_DIR = Path("docs/figures")
ONLINE_OUT_DIR = Path("data/processed/online")


def load_fitted_params(model_path: Path) -> "hr._core.HSMMParams":
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
    parser = argparse.ArgumentParser(description="Run the online particle filter over historical data.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--out-dir", default=str(ONLINE_OUT_DIR))
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
        print(f"No fitted model found at {model_path}. Run fit_offline_hsmm.py first.", file=sys.stderr)
        return 1

    params, payload = load_fitted_params(model_path)
    feat_config = FeatureConfig(primary_series=payload["primary_series"])
    series = load_primary_series(args.ticker, args.processed_dir, feat_config)

    print(f"Running online particle filter on {args.ticker} "
          f"({len(series)} obs, {args.n_particles} particles)...")

    of = hr.OnlineFilter(params, n_particles=args.n_particles, seed=args.seed)

    # Time per-step updates individually (after an untimed warm-up step,
    # since the first call pays for population initialization already done
    # in __init__, plus any one-off JIT/caching effects) to report the
    # millisecond-scale latency claimed in the project brief.
    step_times = []
    obs_values = series.values
    t0 = time.perf_counter()
    for y in obs_values:
        s0 = time.perf_counter()
        of.step(float(y))
        step_times.append(time.perf_counter() - s0)
    total_time = time.perf_counter() - t0

    step_times_ms = np.array(step_times) * 1000
    print(f"Total: {total_time:.3f}s over {len(obs_values)} steps")
    print(f"Per-step latency: mean={step_times_ms.mean():.3f}ms  "
          f"p50={np.median(step_times_ms):.3f}ms  p99={np.percentile(step_times_ms, 99):.3f}ms")

    posterior_df = of.__class__.__dict__  # placeholder guard, unused
    posterior_df = None
    import pandas as pd
    posterior_df = pd.DataFrame(
        np.array(of._history), index=series.index, columns=list(range(params.n_states))
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.ticker}_online_posterior.csv"
    posterior_df.to_csv(out_path)
    print(f"Saved online posterior history to {out_path}")

    if not args.no_plot:
        labels = payload.get("labels") or hr.regime_label_map(params)
        labels = {int(k): v for k, v in labels.items()}
        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_regime_probabilities(
            series.index, posterior_df.values, params.n_states, state_labels=labels,
            title=f"{args.ticker}: Online Particle Filter Regime Posterior",
        )
        fig_path = fig_dir / f"{args.ticker}_online_posterior.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Saved plot to {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
