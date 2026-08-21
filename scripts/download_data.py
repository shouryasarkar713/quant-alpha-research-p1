"""CLI script to download, validate, and clean market data for the research universe."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.calendar import nyse_trading_sessions
from src.data.cleaning import clean_ohlcv
from src.data.loader import DataDownloadError, download_ohlcv, generate_synthetic_ohlcv
from src.data.universe import load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean historical market data.")
    parser.add_argument("--universe", type=str, default="sp100_20140101", help="Universe name")
    parser.add_argument("--start-date", type=str, default="2014-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["research", "synthetic"],
        default="research",
        help="Data ingestion mode: 'research' (genuine historical market data) or 'synthetic' (offline test mode)",
    )
    parser.add_argument("--synthetic", action="store_true", help="Alias for --mode synthetic")
    args = parser.parse_args()

    mode = "synthetic" if args.synthetic else args.mode

    # 1. Load universe
    tickers = load_universe(args.universe)
    logger.info("Loaded universe '%s' with %d tickers (Mode: %s).", args.universe, len(tickers), mode.upper())

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. Ingest data
    if mode == "synthetic":
        logger.info("Generating synthetic research dataset for offline execution (mode=synthetic)...")
        df_raw = generate_synthetic_ohlcv(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            seed=42,
        )
        raw_file = raw_dir / "raw_ohlcv.parquet"
        df_raw.to_parquet(raw_file)
        logger.info("Saved synthetic raw data to %s (%d rows)", raw_file, len(df_raw))
    elif mode == "research":
        logger.info("Executing RESEARCH MODE: Downloading genuine historical market data from yfinance...")
        try:
            df_raw = download_ohlcv(
                tickers=tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                output_dir=raw_dir,
                filename="raw_ohlcv.parquet",
            )
        except DataDownloadError as e:
            logger.error("RESEARCH MODE FATAL ERROR: Live data download failed: %s", e)
            logger.error("Research mode STRICTLY REFUSES to generate synthetic fallback data.")
            sys.exit(1)
        except Exception as e:
            logger.error("RESEARCH MODE FATAL ERROR: Unexpected exception during download: %s", e)
            logger.error("Research mode STRICTLY REFUSES to generate synthetic fallback data.")
            sys.exit(1)
    else:
        raise ValueError(f"Unknown data mode: {mode}")

    # 3. Clean and audit data using authoritative NYSE calendar sessions
    trading_sessions = nyse_trading_sessions(args.start_date, args.end_date)
    logger.info(
        "Cleaning OHLCV against %d authoritative NYSE trading sessions (%s to %s)...",
        len(trading_sessions),
        args.start_date,
        args.end_date,
    )
    df_cleaned, report = clean_ohlcv(df_raw, valid_trading_days=trading_sessions)
    processed_file = processed_dir / "cleaned_ohlcv.parquet"
    df_cleaned.to_parquet(processed_file)
    logger.info("Saved cleaned data to %s (%d rows)", processed_file, len(df_cleaned))

    # Save DataQualityReport with provenance metadata
    cleaned_tickers = sorted(df_cleaned.index.get_level_values("ticker").unique().tolist())
    missing_tickers = sorted(list(set(tickers) - set(cleaned_tickers)))
    terminated_list = sorted(list(report.terminated_tickers.keys()))
    continuous_list = sorted(list(set(cleaned_tickers) - set(terminated_list)))

    report_dict = report.to_dict()
    report_dict["provenance"] = {
        "mode": mode,
        "data_source": "yfinance" if mode == "research" else "synthetic_generator",
        "universe": args.universe,
        "frozen_universe_total_constituents": len(tickers),
        "usable_historical_identities": len(cleaned_tickers),
        "continuous_identities_count": len(continuous_list),
        "continuous_identities_list": continuous_list,
        "terminated_identities_count": len(terminated_list),
        "terminated_identities": report.terminated_tickers,
        "unavailable_legacy_delisted_constituents": missing_tickers,
        "unavailable_legacy_delisted_notes": {
            "APC": "Anadarko Petroleum Corp - Acquired by Occidental Petroleum (OXY) Aug 2019; series purged on public endpoint.",
            "DOW": "The Dow Chemical Company - Merged into DowDuPont Sep 2017; series purged on public endpoint (2019 spin-off rejected).",
            "EMC": "EMC Corporation - Acquired by Dell Technologies Sep 2016; series purged on public endpoint (2023 SPAC rejected).",
            "FOXA": "Twenty-First Century Fox Inc - Acquired by Disney Mar 2019; series purged on public endpoint (2019 spin-off rejected).",
            "MON": "Monsanto Co - Acquired by Bayer AG Jun 2018; series purged on public endpoint.",
            "RTN": "Raytheon Co - Merged with UTX Apr 2020 to form RTX; standalone series purged on public endpoint.",
            "WAG": "Walgreen Co - Reorganized as WBA Dec 2014, taken private Aug 2025; series purged on public endpoint.",
        },
        "fabricated_observations": 0,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "authoritative_trading_sessions": len(trading_sessions),
        "cleaned_total_panel_rows": len(df_cleaned),
        "cleaned_unique_dates": int(df_cleaned.index.get_level_values("date").nunique()),
        "cleaned_unique_tickers": int(df_cleaned.index.get_level_values("ticker").nunique()),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    report_file = processed_dir / "data_quality_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Saved Data Quality Report to %s", report_file)


if __name__ == "__main__":
    main()
