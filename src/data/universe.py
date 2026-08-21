"""Universe loading and point-in-time universe management."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def load_universe(
    universe_name: str = "sp100_20140101",
    universe_dir: str | Path = "data/universe",
) -> list[str]:
    """
    Load the historically documented start-of-sample universe definition.

    Important:
    This function loads and validates the frozen membership file. It does not
    remove securities because of missingness, future delisting, or future
    liquidity information. Those issues affect point-in-time signal eligibility,
    not historical membership.

    Parameters
    ----------
    universe_name : str
        Base name of the universe file (e.g. 'sp100_20140101').
    universe_dir : str | Path
        Directory containing universe definitions.

    Returns
    -------
    list[str]
        List of ticker symbols frozen at the start-of-sample date.
    """
    directory = Path(universe_dir)
    csv_path = directory / f"{universe_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Universe file not found at {csv_path}. "
            f"Please ensure historical universe definition is present."
        )

    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns:
        raise ValueError(f"Universe CSV {csv_path} must contain a 'ticker' column.")

    tickers = [str(t).strip().upper() for t in df["ticker"].dropna().tolist() if str(t).strip()]
    if not tickers:
        raise ValueError(f"Universe file {csv_path} contains no valid tickers.")

    return sorted(list(set(tickers)))


def load_universe_metadata(
    universe_name: str = "sp100_20140101",
    universe_dir: str | Path = "data/universe",
) -> dict:
    """Load metadata and provenance information for a universe definition."""
    directory = Path(universe_dir)
    meta_path = directory / f"{universe_name}_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
