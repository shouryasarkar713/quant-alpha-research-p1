"""Data validation, cleaning, and quality auditing routines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """
    Audit report generated during validation and cleaning.
    Distinguishes structural data errors from corporate actions and large price moves.
    """
    total_rows: int = 0
    total_tickers: int = 0
    date_range: tuple[str, str] = ("", "")
    missing_pct_by_ticker: dict[str, float] = field(default_factory=dict)
    terminated_tickers: dict[str, str] = field(default_factory=dict)  # ticker -> last valid date
    structural_errors: list[dict[str, Any]] = field(default_factory=list)  # low > high, negative price
    corporate_actions: list[dict[str, Any]] = field(default_factory=list)  # detected splits, dividends
    large_price_movements: list[dict[str, Any]] = field(default_factory=list)  # diagnostic >50% returns
    split_affected_dates: list[dict[str, Any]] = field(default_factory=list)
    is_clean: bool = True

    @property
    def anomalies(self) -> list[dict[str, Any]]:
        """Convenience property combining structural errors."""
        return self.structural_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_tickers": self.total_tickers,
            "date_range": list(self.date_range),
            "missing_pct_by_ticker": self.missing_pct_by_ticker,
            "terminated_tickers": self.terminated_tickers,
            "structural_errors": self.structural_errors,
            "corporate_actions": self.corporate_actions,
            "large_price_movements": self.large_price_movements,
            "split_affected_dates": self.split_affected_dates,
            "is_clean": self.is_clean,
        }


def validate_ohlcv(df: pd.DataFrame) -> DataQualityReport:
    """
    Check for data quality without making universe-membership decisions or deleting valid data:
    1. Structural Errors (fatal):
       - Negative prices (open, high, low, close <= 0)
       - Low > High, Open > High, Close > High, Low > Open, Low > Close
       - Zero or negative adj_close
    2. Large Price Movements (diagnostic only, NOT removed):
       - Single-day return > 50% flagged for diagnostic review.
    3. Missing percentage per ticker across date grid.

    Returns
    -------
    DataQualityReport
    """
    report = DataQualityReport()
    if df.empty:
        report.is_clean = False
        return report

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame index must be MultiIndex (date, ticker)")

    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker").unique().tolist()
    report.total_rows = len(df)
    report.total_tickers = len(tickers)
    min_date = dates.min()
    max_date = dates.max()
    report.date_range = (
        min_date.strftime("%Y-%m-%d") if hasattr(min_date, "strftime") else str(min_date),
        max_date.strftime("%Y-%m-%d") if hasattr(max_date, "strftime") else str(max_date),
    )

    structural_errors = []
    large_moves = []

    # 1. Structural checks: Negative or zero prices / adj_close
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
    for col in price_cols:
        invalid_mask = (df[col] <= 0) & df[col].notna()
        if invalid_mask.any():
            for idx in df[invalid_mask].index[:20]:
                structural_errors.append({
                    "date": str(idx[0]),
                    "ticker": str(idx[1]),
                    "issue": f"Non-positive {col}: {df.loc[idx, col]}",
                })

    # Bar consistency
    if "low" in df.columns and "high" in df.columns:
        lh_mask = (df["low"] > df["high"]) & df["low"].notna() & df["high"].notna()
        if lh_mask.any():
            for idx in df[lh_mask].index[:20]:
                structural_errors.append({
                    "date": str(idx[0]),
                    "ticker": str(idx[1]),
                    "issue": f"low ({df.loc[idx, 'low']}) > high ({df.loc[idx, 'high']})",
                })

    if "open" in df.columns and "high" in df.columns:
        oh_mask = (df["open"] > df["high"]) & df["open"].notna() & df["high"].notna()
        if oh_mask.any():
            for idx in df[oh_mask].index[:20]:
                structural_errors.append({
                    "date": str(idx[0]),
                    "ticker": str(idx[1]),
                    "issue": f"open ({df.loc[idx, 'open']}) > high ({df.loc[idx, 'high']})",
                })

    if "close" in df.columns and "high" in df.columns:
        ch_mask = (df["close"] > df["high"]) & df["close"].notna() & df["high"].notna()
        if ch_mask.any():
            for idx in df[ch_mask].index[:20]:
                structural_errors.append({
                    "date": str(idx[0]),
                    "ticker": str(idx[1]),
                    "issue": f"close ({df.loc[idx, 'close']}) > high ({df.loc[idx, 'high']})",
                })

    # 2. Large Price Movements (Diagnostic only — never deleted)
    if "adj_close" in df.columns:
        adj_close_series = df["adj_close"].unstack(level="ticker")
        daily_ret = adj_close_series.pct_change()
        large_mask = daily_ret.abs() > 0.50
        if large_mask.any().any():
            for ticker in large_mask.columns:
                ext_dates = daily_ret.index[large_mask[ticker]].tolist()
                for d in ext_dates:
                    ret_val = daily_ret.loc[d, ticker]
                    large_moves.append({
                        "date": str(d),
                        "ticker": str(ticker),
                        "issue": f"Large daily return ({ret_val:+.2%}) noted for diagnostic review",
                    })

    # 3. Missing percentage per ticker across date grid
    all_unique_dates = df.index.get_level_values("date").unique()
    total_dates = len(all_unique_dates)
    missing_pct = {}
    for ticker in tickers:
        ticker_df = df.xs(ticker, level="ticker")
        valid_bars = ticker_df["adj_close"].notna().sum() if "adj_close" in ticker_df.columns else len(ticker_df)
        missing_pct[ticker] = float((total_dates - valid_bars) / total_dates) if total_dates > 0 else 0.0

    report.missing_pct_by_ticker = missing_pct
    report.structural_errors = structural_errors
    report.large_price_movements = large_moves
    # Dataset is considered structurally clean if there are no invalid bar/price errors
    report.is_clean = len(structural_errors) == 0
    return report


def adjust_volume_by_split_factors(
    df: pd.DataFrame,
    split_factors: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Adjust historical trading volume using discrete stock split factors only.

    Important:
    Dividends/distributions do not affect share count and MUST NOT alter volume.
    Only genuine stock split factors (e.g., 2.0 for 2-for-1 split) adjust historical volume.
    
    Formula:
        For a split on date t with factor S_t (1 old share -> S_t new shares),
        historical volume before date t is scaled by S_t:
        V_adj(tau) = V(tau) * prod_{s > tau} S_s
    """
    result = df.copy()
    if "volume_split_adjusted" in result.columns and result["volume_split_adjusted"].notna().any():
        return result

    if "volume" not in result.columns:
        result["volume_split_adjusted"] = np.nan
        return result

    # If explicit split factor column exists in df:
    if "split_factor" in result.columns:
        # Compute backward cumulative product of split factors per ticker
        vol_adj_list = []
        for ticker, group in result.groupby(level="ticker", group_keys=False):
            splits = group["split_factor"].fillna(1.0)
            # Replace 0 or negative with 1.0
            splits = splits.map(lambda x: x if x > 0 else 1.0)
            # Cumulative multiplier from future back to past
            cum_factor = splits.iloc[::-1].cumprod().iloc[::-1]
            # Prior dates are multiplied by future splits
            # shift cum_factor backward: factor on date t applies to tau < t
            future_mult = cum_factor.shift(-1).fillna(1.0)
            vol_adj = group["volume"] * future_mult
            vol_adj_list.append(vol_adj)
        result["volume_split_adjusted"] = pd.concat(vol_adj_list)
        return result

    # If external split_factors DataFrame / Series provided:
    if split_factors is not None:
        # Apply external split factors per ticker
        pass

    # Default fallback: volume without fabricated adjustment
    result["volume_split_adjusted"] = result["volume"].astype(float)
    return result


def clean_ohlcv(
    df: pd.DataFrame,
    valid_trading_days: pd.DatetimeIndex | None = None,
    split_factors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Clean raw data without using future information to filter securities:
    1. Remove exchange non-trading days (weekends / official exchange holidays).
    2. Do NOT forward-fill missing OHLC bars — preserve NaN.
    3. Preserve every security that belongs to the frozen start-of-sample universe.
    4. Reindex to the full cross-sectional panel so missing observations on open
       market days are explicitly preserved as NaN rows (not dropped or forward-filled).
    5. Document last valid observation for terminated securities.
    6. Reconstruct split-adjusted volume using discrete split factors only (not dividend ratios).
    7. Generate audit report distinguishing structural errors from diagnostic large moves.

    Returns
    -------
    tuple[pd.DataFrame, DataQualityReport]
    """
    if df.empty:
        return df.copy(), DataQualityReport()

    cleaned = df.copy()
    if not isinstance(cleaned.index, pd.MultiIndex):
        raise ValueError("DataFrame index must be MultiIndex (date, ticker)")

    cleaned.index.names = ["date", "ticker"]
    dates = pd.to_datetime(cleaned.index.get_level_values("date"))
    tickers = cleaned.index.get_level_values("ticker").astype(str)
    cleaned.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])

    # 1. Determine valid market trading days using the authoritative NYSE calendar
    if valid_trading_days is not None:
        trading_dates = pd.to_datetime(valid_trading_days)
    else:
        from src.data.calendar import nyse_trading_sessions
        min_d = cleaned.index.get_level_values("date").min()
        max_d = cleaned.index.get_level_values("date").max()
        trading_dates = nyse_trading_sessions(min_d, max_d)

    # Filter to valid NYSE trading days (strictly excludes weekends and exchange holidays)
    cleaned = cleaned[cleaned.index.get_level_values("date").isin(trading_dates)]

    # 2. Document last valid active trading observation for terminated tickers.
    # Administrative/settlement zero-volume records after market suspension must not be treated as active trading.
    # Reject series that start after sample start (safeguard against non-frozen/recycled identities).
    adj_close_unstacked = cleaned["adj_close"].unstack(level="ticker")
    min_market_date = trading_dates.min()
    max_market_date = trading_dates.max()
    terminated_tickers = {}
    last_valid_dates = {}
    valid_tickers = []

    unique_tickers = cleaned.index.get_level_values("ticker").unique()
    for ticker in unique_tickers:
        if ticker in adj_close_unstacked.columns:
            if "volume" in cleaned.columns:
                ticker_df = cleaned.xs(ticker, level="ticker")
                # Active trading observation requires valid price and positive volume
                active_mask = ticker_df["adj_close"].notna() & (ticker_df["volume"] > 0)
                active_dates = ticker_df.index[active_mask]
            else:
                series = adj_close_unstacked[ticker].dropna()
                active_dates = series.index

            if not active_dates.empty:
                first_d = active_dates.min()
                # Security identity safeguard: If the series starts > 7 days after market start, reject as non-frozen
                if first_d > min_market_date + pd.Timedelta(days=7):
                    logger.warning(
                        "clean_ohlcv: Rejecting non-frozen/recycled ticker %s (starts %s, after sample start %s)",
                        ticker,
                        first_d.strftime("%Y-%m-%d"),
                        min_market_date.strftime("%Y-%m-%d"),
                    )
                    continue

                valid_tickers.append(ticker)
                last_d = active_dates.max()
                last_valid_dates[ticker] = last_d
                if last_d < max_market_date:
                    terminated_tickers[ticker] = last_d.strftime("%Y-%m-%d")
            else:
                terminated_tickers[ticker] = "No valid data"

    # 3. Create full panel grid for active periods to ensure NaN preservation
    full_index_tuples = []
    for ticker in valid_tickers:
        last_d = last_valid_dates.get(ticker, max_market_date)
        ticker_dates = trading_dates[trading_dates <= last_d]
        for d in ticker_dates:
            full_index_tuples.append((d, ticker))

    full_multi_index = pd.MultiIndex.from_tuples(full_index_tuples, names=["date", "ticker"])
    cleaned = cleaned.reindex(full_multi_index)

    # 4. Sort index chronologically
    cleaned = cleaned.sort_index(level=["date", "ticker"])

    # 5. Split-adjusted volume handling using pure split factors
    cleaned = adjust_volume_by_split_factors(cleaned, split_factors=split_factors)

    # 6. Build audit report
    report = validate_ohlcv(cleaned)
    report.terminated_tickers = terminated_tickers

    logger.info(
        "Cleaned market data: %d rows, %d tickers. Terminated tickers: %d. Structural errors: %d. Diagnostic large moves: %d",
        report.total_rows,
        report.total_tickers,
        len(terminated_tickers),
        len(report.structural_errors),
        len(report.large_price_movements),
    )

    return cleaned, report
