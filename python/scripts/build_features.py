#!/usr/bin/env python3
"""
Phase 1: compute log returns, rolling realized volatility, and a liquidity
proxy from raw OHLCV CSVs.

Usage:
    python python/scripts/build_features.py
    python python/scripts/build_features.py --tickers SPY TLT --vol-window 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsmm_regime.data import build_features, FeatureConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Build return/volatility features from raw OHLCV data.")
    parser.add_argument("--tickers", nargs="*", default=None,
                         help="Tickers to process (default: infer from raw-dir contents)")
    parser.add_argument("--raw-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--out-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--no-annualize-vol", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = FeatureConfig(
        vol_window=args.vol_window,
        annualize_vol=not args.no_annualize_vol,
    )

    try:
        features = build_features(
            tickers=args.tickers, raw_dir=args.raw_dir, out_dir=args.out_dir, config=config,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"build_features failed: {e}", file=sys.stderr)
        return 1

    print(f"Built features for {len(features)} ticker(s):")
    for ticker, df in features.items():
        print(f"  {ticker}: {len(df)} rows, columns={list(df.columns)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
