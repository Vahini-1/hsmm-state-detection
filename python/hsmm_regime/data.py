"""
Data acquisition and feature engineering for the HSMM regime pipeline.

Handles:
  - Pulling daily OHLCV data for a universe of tickers (via yfinance).
  - Adjusting for splits/dividends (yfinance's auto_adjust handles this
    upstream; we additionally guard against remaining NaNs/gaps here).
  - Computing log returns, rolling realized volatility, and an optional
    liquidity proxy (volume z-score) as HSMM emission inputs.

The HSMM engine in this package is currently univariate (see
`hsmm.forward_backward` / `types.hpp::EmissionParams`), so `build_features`
produces a single primary series (log returns by default) intended to feed
the C++ engine, while retaining the full feature frame for diagnostics,
plotting, and eventual MS-VAR extension (which is genuinely multivariate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    vol_window: int = 20            # rolling realized-vol window (trading days)
    annualize_vol: bool = True      # scale realized vol to annualized units
    trading_days_per_year: int = 252
    volume_zscore_window: int = 60  # rolling window for the liquidity proxy
    primary_series: str = "log_return"  # column fed to the univariate HSMM


def fetch_data(
    tickers: list[str],
    start: str,
    end: str | None = None,
    out_dir: Path | str = RAW_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Pull daily OHLCV data for each ticker via yfinance and write raw CSVs.

    Returns a dict of {ticker: DataFrame} with columns
    [Open, High, Low, Close, Volume] (auto-adjusted for splits/dividends).

    Raises a RuntimeError with a clear message (rather than a bare
    yfinance exception) if a ticker returns no data at all, since an empty
    frame silently propagating into feature engineering is a common and
    hard-to-diagnose failure mode.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required for fetch_data(); install it via "
            "`pip install yfinance` or the project's environment.yml."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        logger.info("Fetching %s from %s to %s", ticker, start, end or "today")
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for ticker '{ticker}'. Check the symbol, "
                f"date range ({start} .. {end or 'today'}), and network access."
            )

        # yfinance sometimes returns a MultiIndex column frame even for a
        # single ticker (depending on version); flatten defensively.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "date"
        frames[ticker] = df

        out_path = out_dir / f"{ticker}.csv"
        df.to_csv(out_path)
        logger.info("Wrote %d rows to %s", len(df), out_path)

    return frames


def _load_raw(ticker: str, raw_dir: Path | str = RAW_DATA_DIR) -> pd.DataFrame:
    path = Path(raw_dir) / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data file found at {path}. Run fetch_data() first, "
            f"or pass a different raw_dir."
        )
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def _sanity_check(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Guard against the failure modes Phase 1 explicitly calls out: missing
    data, stale/duplicated timestamps, and non-positive prices (which would
    make log returns undefined). We forward-fill short gaps (<=3 sessions,
    typical of holidays misalignment across a multi-ticker universe) and
    drop rows that remain broken, logging what was removed rather than
    failing silently.
    """
    n_before = len(df)

    df = df[~df.index.duplicated(keep="last")].sort_index()

    if (df["Close"] <= 0).any():
        bad = (df["Close"] <= 0).sum()
        logger.warning("%s: dropping %d rows with non-positive Close", ticker, bad)
        df = df[df["Close"] > 0]

    df["Close"] = df["Close"].ffill(limit=3)
    df = df.dropna(subset=["Close"])

    n_after = len(df)
    if n_after < n_before:
        logger.info("%s: sanity check removed %d/%d rows", ticker, n_before - n_after, n_before)

    return df


def build_features(
    tickers: list[str] | None = None,
    raw_dir: Path | str = RAW_DATA_DIR,
    out_dir: Path | str = PROCESSED_DATA_DIR,
    config: FeatureConfig = field(default_factory=FeatureConfig),
) -> dict[str, pd.DataFrame]:
    """
    Load raw OHLCV CSVs, compute log returns / realized volatility / a
    liquidity proxy, and write processed feature CSVs.

    If `tickers` is None, infers the universe from whatever CSVs exist in
    `raw_dir`.
    """
    if isinstance(config, type(field)):  # dataclasses.field sentinel guard
        config = FeatureConfig()

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        tickers = sorted(p.stem for p in raw_dir.glob("*.csv"))
        if not tickers:
            raise FileNotFoundError(
                f"No raw CSVs found in {raw_dir}; run fetch_data() first."
            )

    features: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = _load_raw(ticker, raw_dir)
        df = _sanity_check(df, ticker)

        out = pd.DataFrame(index=df.index)
        out["close"] = df["Close"]
        out["log_return"] = np.log(df["Close"]).diff()

        realized_vol = out["log_return"].rolling(config.vol_window).std()
        if config.annualize_vol:
            realized_vol = realized_vol * np.sqrt(config.trading_days_per_year)
        out["realized_vol"] = realized_vol

        if "Volume" in df.columns:
            vol_roll_mean = df["Volume"].rolling(config.volume_zscore_window).mean()
            vol_roll_std = df["Volume"].rolling(config.volume_zscore_window).std()
            out["volume_zscore"] = (df["Volume"] - vol_roll_mean) / vol_roll_std.replace(0, np.nan)

        out = out.dropna()

        if out.empty:
            raise RuntimeError(
                f"{ticker}: feature frame is empty after warm-up windows "
                f"(vol_window={config.vol_window}). Fetch a longer history."
            )

        features[ticker] = out
        out_path = out_dir / f"{ticker}_features.csv"
        out.to_csv(out_path)
        logger.info("%s: wrote %d feature rows to %s", ticker, len(out), out_path)

    return features


def load_primary_series(
    ticker: str,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    config: FeatureConfig | None = None,
) -> pd.Series:
    """
    Convenience loader: read a ticker's processed feature file and return
    the single column the univariate HSMM engine should be fit on (log
    returns by default; realized_vol is also a common choice for
    volatility-regime models specifically).
    """
    config = config or FeatureConfig()
    path = Path(processed_dir) / f"{ticker}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No processed feature file at {path}; run build_features() first."
        )
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if config.primary_series not in df.columns:
        raise KeyError(
            f"'{config.primary_series}' not in processed columns {list(df.columns)}"
        )
    return df[config.primary_series]
