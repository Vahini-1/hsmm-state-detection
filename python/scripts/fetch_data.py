#!/usr/bin/env python3
"""
Phase 1: pull daily OHLCV data for a ticker universe.

Usage:
    python python/scripts/fetch_data.py --tickers SPY TLT GLD QQQ --start 2010-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsmm_regime.data import fetch_data, RAW_DATA_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch daily OHLCV data via yfinance.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols, e.g. SPY TLT GLD")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD (default: today)")
    parser.add_argument("--out-dir", default=str(RAW_DATA_DIR), help="Output directory for raw CSVs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        frames = fetch_data(args.tickers, args.start, args.end, out_dir=args.out_dir)
    except (RuntimeError, ImportError) as e:
        print(f"fetch_data failed: {e}", file=sys.stderr)
        return 1

    print(f"Fetched {len(frames)} ticker(s):")
    for ticker, df in frames.items():
        print(f"  {ticker}: {len(df)} rows, {df.index.min().date()} .. {df.index.max().date()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
