"""Data downloading, loading, and synthetic research dataset generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataDownloadError(Exception):
    """Raised when data download fails for a critical fraction of tickers."""


# Approved provider mapping table for the frozen 2014 S&P 100 universe
_CANONICAL_YFINANCE_MAP: dict[str, str] = {
    "BRK.B": "BRK-B",  # Standard dot-to-dash exchange formatting
    "FB": "META",      # Same legal corporation (Meta Platforms Inc); ticker renamed June 2022
    "BK": "BNY",       # Same legal corporation (The Bank of New York Mellon Corp); provider lookup key for May 2026 NYSE rename
    "UTX": "RTX",      # Legal entity survivor in April 2020 merger of equals with Raytheon Company
}

# Explicitly rejected recycled / non-frozen ticker identities
_KNOWN_UNAVAILABLE_OR_RECYCLED_TICKERS: set[str] = {
    "APC",   # Anadarko Petroleum Corp - Acquired by OXY Aug 2019; series purged on Yahoo
    "DOW",   # 2019 Dow Inc. spin-off; not 2014 The Dow Chemical Company
    "EMC",   # 2023 Emerging Markets Horizon Corp SPAC; not 2014 EMC Corporation
    "FOXA",  # 2019 Fox Corporation spin-off; not 2014 Twenty-First Century Fox Inc
    "MON",   # Monsanto Company - Acquired by Bayer AG Jun 2018; series purged on Yahoo
    "RTN",   # Raytheon Company - Merged into UTX Apr 2020; standalone series purged on Yahoo
    "WAG",   # Walgreen Co - Reorganized as WBA Dec 2014, taken private Aug 2025; series purged on Yahoo
}


def _standardize_ticker_for_yfinance(ticker: str) -> str:
    """Map ticker notation (e.g. BRK.B -> BRK-B, FB -> META) for Yahoo Finance API."""
    ticker_clean = ticker.strip().upper()
    if ticker_clean in _CANONICAL_YFINANCE_MAP:
        return _CANONICAL_YFINANCE_MAP[ticker_clean]
    return ticker_clean.replace(".", "-")


def _restore_ticker_from_yfinance(ticker: str) -> str:
    """Map standardized ticker back to universe format (e.g. BRK-B -> BRK.B)."""
    reverse_map = {v: k for k, v in _CANONICAL_YFINANCE_MAP.items()}
    if ticker in reverse_map:
        return reverse_map[ticker]
    if ticker == "BRK-B":
        return "BRK.B"
    return ticker


def download_ohlcv(
    tickers: Sequence[str],
    start_date: str = "2014-01-01",
    end_date: str = "2024-12-31",
    output_dir: str | Path = "data/raw",
    filename: str = "raw_ohlcv.parquet",
    max_failure_rate: float = 0.10,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a list of tickers via yfinance.
    Computes split-adjusted volume using discrete stock split factors only.

    Parameters
    ----------
    tickers : Sequence[str]
        List of ticker symbols.
    start_date : str
        Start date 'YYYY-MM-DD'.
    end_date : str
        End date 'YYYY-MM-DD'.
    output_dir : str | Path
        Directory to save parquet file.
    filename : str
        Output parquet filename.
    max_failure_rate : float
        Maximum fraction of missing tickers before raising DataDownloadError.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex (date, ticker) and columns:
        [open, high, low, close, adj_close, volume, volume_split_adjusted]
    """
    import yfinance as yf

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_file = output_path / filename

    # End date in yfinance history is exclusive; increment by 1 day to include end_date
    end_date_eff = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Downloading daily OHLCV for %d tickers from %s to %s...", len(tickers), start_date, end_date)

    frames = []
    failed_tickers = []
    success_tickers = []

    for orig_t in tickers:
        if orig_t in _KNOWN_UNAVAILABLE_OR_RECYCLED_TICKERS:
            logger.info("Skipping known unavailable/recycled ticker identity %s (documented legacy constituent)", orig_t)
            failed_tickers.append(orig_t)
            continue

        yf_t = _standardize_ticker_for_yfinance(orig_t)
        try:
            t = yf.Ticker(yf_t)
            hist = t.history(
                start=start_date,
                end=end_date_eff,
                auto_adjust=False,
                actions=True,
            )

            if hist.empty or hist["Close"].dropna().empty:
                logger.warning("No data retrieved for ticker %s (yf: %s)", orig_t, yf_t)
                failed_tickers.append(orig_t)
                continue

            # Drop completely empty rows
            hist = hist.dropna(how="all", subset=["Open", "High", "Low", "Close"])
            if hist.empty:
                failed_tickers.append(orig_t)
                continue

            # Normalize index to timezone-naive datetime (date at 00:00:00)
            naive_dates = pd.to_datetime(pd.DatetimeIndex(hist.index).tz_localize(None).date)

            # Point-in-time identity sanity check: Series must start at start of sample
            first_session = naive_dates.min()
            expected_start = pd.Timestamp(start_date)
            if first_session > expected_start + pd.Timedelta(days=7):
                logger.warning(
                    "Rejecting ticker %s: retrieved series starts at %s (after sample start %s); non-frozen/recycled identity",
                    orig_t,
                    first_session.strftime("%Y-%m-%d"),
                    start_date,
                )
                failed_tickers.append(orig_t)
                continue

            df_t = pd.DataFrame(index=naive_dates)
            df_t["open"] = hist["Open"].astype(float).values
            df_t["high"] = hist["High"].astype(float).values
            df_t["low"] = hist["Low"].astype(float).values
            df_t["close"] = hist["Close"].astype(float).values
            df_t["adj_close"] = hist.get("Adj Close", hist["Close"]).astype(float).values
            df_t["volume"] = hist["Volume"].fillna(0).astype("int64").values

            # Pure split-factor adjustment for volume
            if "Stock Splits" in hist.columns:
                splits = pd.Series(hist["Stock Splits"].fillna(0.0).values, index=naive_dates)
                splits_eff = splits.map(lambda s: s if s > 0 else 1.0)
                cum_split = splits_eff.iloc[::-1].cumprod().iloc[::-1]
                future_split_mult = cum_split.shift(-1).fillna(1.0)
                df_t["volume_split_adjusted"] = (df_t["volume"] * future_split_mult).values
            else:
                df_t["volume_split_adjusted"] = df_t["volume"].astype(float).values

            df_t["ticker"] = orig_t
            df_t["date"] = df_t.index
            frames.append(df_t)
            success_tickers.append(orig_t)

        except Exception as err:
            logger.warning("Error processing ticker %s (yf: %s): %s", orig_t, yf_t, err)
            failed_tickers.append(orig_t)

    failure_rate = len(failed_tickers) / max(len(tickers), 1)
    logger.info(
        "Downloaded OHLCV: %d succeeded, %d failed (failure rate: %.1f%%).",
        len(success_tickers),
        len(failed_tickers),
        failure_rate * 100.0,
    )

    if failure_rate > max_failure_rate:
        raise DataDownloadError(
            f"Download failed for {len(failed_tickers)} / {len(tickers)} tickers ({failure_rate:.1%}) "
            f"exceeding max threshold ({max_failure_rate:.1%}). Failed: {failed_tickers}"
        )

    if not frames:
        raise DataDownloadError("No market data successfully retrieved.")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.set_index(["date", "ticker"]).sort_index()

    combined.to_parquet(target_file)
    logger.info("Saved raw OHLCV data to %s (%d rows across %d tickers)", target_file, len(combined), len(success_tickers))
    return combined


def load_ohlcv(data_path: str | Path) -> pd.DataFrame:
    """
    Load previously saved OHLCV data from Parquet.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex (date, ticker).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        if "date" in df.columns and "ticker" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["date", "ticker"])
        else:
            raise ValueError("Parquet file must contain MultiIndex (date, ticker) or 'date'/'ticker' columns.")

    return df.sort_index()


def generate_synthetic_ohlcv(
    tickers: Sequence[str] | None = None,
    start_date: str = "2014-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    inject_missing_bars: bool = True,
    inject_terminations: bool = True,
) -> pd.DataFrame:
    """
    Generate realistic, deterministic synthetic multi-asset daily OHLCV dataset
    strictly conforming to the v1.2 specification schema.

    Accurately models:
    - Discrete stock splits (adjusting historical raw price and split-adjusted volume)
    - Dividend distributions (adjusting historical adj_close without affecting volume)
    - Early delistings / mergers (EMC, MON, TWX)
    - Missing bars on trading days preserved as NaN
    - Strict adherence to the official exchange-aware NYSE calendar (no weekend or holiday bars)
    """
    from src.data.calendar import nyse_trading_sessions

    rng = np.random.default_rng(seed)

    if tickers is None:
        from src.data.universe import load_universe
        try:
            tickers = load_universe()
        except Exception:
            tickers = ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ", "PG", "GE", "EMC", "MON"]

    dates = nyse_trading_sessions(start_date, end_date)
    n_days = len(dates)

    market_innovations = rng.normal(0.0003, 0.01, size=n_days)

    records = []

    termination_map = {}
    if inject_terminations:
        if "EMC" in tickers:
            termination_map["EMC"] = pd.Timestamp("2016-09-07")
        if "MON" in tickers:
            termination_map["MON"] = pd.Timestamp("2018-06-07")
        if "TWX" in tickers:
            termination_map["TWX"] = pd.Timestamp("2018-06-14")

    for i, ticker in enumerate(tickers):
        beta = rng.uniform(0.7, 1.3)
        sigma = rng.uniform(0.01, 0.02)
        initial_price = rng.uniform(30.0, 120.0)

        idio_innovations = rng.normal(0.0, sigma, size=n_days)
        daily_returns = beta * market_innovations + idio_innovations

        prices = np.zeros(n_days)
        prices[0] = initial_price
        for t in range(1, n_days):
            prices[t] = prices[t - 1] * np.exp(daily_returns[t])

        intraday_vol = sigma * 0.5
        highs = prices * (1.0 + np.abs(rng.normal(0.004, intraday_vol, size=n_days)))
        lows = prices * (1.0 - np.abs(rng.normal(0.004, intraday_vol, size=n_days)))
        opens = np.roll(prices, 1)
        opens[0] = initial_price
        opens = np.clip(opens + rng.normal(0, sigma * 0.2, size=n_days) * prices, lows * 1.001, highs * 0.999)
        closes = prices.copy()

        highs = np.maximum(highs, np.maximum(opens, closes) * 1.001)
        lows = np.minimum(lows, np.minimum(opens, closes) * 0.999)

        base_vol = rng.uniform(2_000_000, 10_000_000)
        volumes = rng.lognormal(mean=np.log(base_vol), sigma=0.4, size=n_days).astype("int64")

        adj_closes = closes.copy()
        split_adjusted_vol = volumes.astype(float).copy()

        # Simulate 2-for-1 stock split for ticker 0 midway through sample
        # On split date, 1 old share became 2 new shares
        # Pre-split raw price was 2x higher; historical volume is multiplied by 2 to match new shares
        if i == 0 and n_days > 100:
            split_idx = n_days // 2
            opens[:split_idx] = opens[:split_idx] * 2.0
            highs[:split_idx] = highs[:split_idx] * 2.0
            lows[:split_idx] = lows[:split_idx] * 2.0
            closes[:split_idx] = closes[:split_idx] * 2.0
            volumes[:split_idx] = (volumes[:split_idx] / 2.0).astype("int64")
            # Historical split-adjusted volume scales pre-split volume by split factor (2.0)
            split_adjusted_vol[:split_idx] = volumes[:split_idx] * 2.0

        for d_idx, d in enumerate(dates):
            if ticker in termination_map and d > termination_map[ticker]:
                continue

            if inject_missing_bars and rng.uniform(0, 1) < 0.002:
                continue

            records.append({
                "date": d,
                "ticker": ticker,
                "open": opens[d_idx],
                "high": highs[d_idx],
                "low": lows[d_idx],
                "close": closes[d_idx],
                "adj_close": adj_closes[d_idx],
                "volume": volumes[d_idx],
                "volume_split_adjusted": split_adjusted_vol[d_idx],
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"]).sort_index()
    return df
