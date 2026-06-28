# preprocess.py

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wrds


def load_local_env() -> None:
    """
    Load local key=value settings from .env if it exists.

    This avoids hard-coding credentials in source files while still allowing
    automatic local login.
    """

    env_path = Path(__file__).resolve().parents[2] / ".env"

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if not os.environ.get(key):
            os.environ[key] = value


def connect_wrds(interactive: bool = True) -> wrds.Connection:
    """
    Connect to WRDS.

    If WRDS_USERNAME and WRDS_PASSWORD are set in the environment, use them.
    Otherwise, fall back to the normal interactive WRDS login flow when
    interactive=True. In long non-interactive jobs, set interactive=False so
    transient connection failures raise instead of blocking on input().
    """

    load_local_env()

    wrds_username = os.environ.get("WRDS_USERNAME")
    wrds_password = os.environ.get("WRDS_PASSWORD")

    if wrds_username and wrds_password and not interactive:
        db = wrds.Connection(
            autoconnect=False,
            wrds_username=wrds_username,
            wrds_password=wrds_password,
        )
        db._Connection__make_sa_engine_conn(raise_err=True)
        db.load_library_list()
        return db

    if wrds_username and wrds_password:
        return wrds.Connection(
            wrds_username=wrds_username,
            wrds_password=wrds_password,
        )

    if not interactive:
        raise RuntimeError(
            "WRDS_USERNAME and WRDS_PASSWORD are required for non-interactive WRDS jobs."
        )

    if wrds_username:
        return wrds.Connection(wrds_username=wrds_username)

    return wrds.Connection()


def download_crsp_daily(
    start_date: str,
    end_date: str,
    output_raw_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Download daily CRSP data from WRDS.

    Uses:
    - crsp.dsf: Daily Stock File
    - crsp.dsenames: names, share codes, exchange codes

    shrcd 10 and 11 usually represent common shares.
    exchcd 1, 2, 3 represent NYSE, AMEX, NASDAQ.
    """

    db = connect_wrds()

    query = f"""
        SELECT
            a.permno,
            a.date,
            a.prc,
            a.openprc,
            a.askhi,
            a.bidlo,
            a.vol,
            a.ret,
            a.shrout,

            b.ticker,
            b.comnam,
            b.shrcd,
            b.exchcd,
            b.siccd

        FROM crsp.dsf AS a

        LEFT JOIN crsp.dsenames AS b
            ON a.permno = b.permno
            AND a.date >= b.namedt
            AND a.date <= b.nameendt

        WHERE a.date >= '{start_date}'
            AND a.date <= '{end_date}'
            AND b.shrcd IN (10, 11)
            AND b.exchcd IN (1, 2, 3)
    """

    df = db.raw_sql(query, date_cols=["date"])

    if output_raw_path is not None:
        output_raw_path = Path(output_raw_path)
        output_raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_raw_path, index=False)

    db.close()
    return df


def clean_crsp_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Clean CRSP daily data and standardize column names."""

    df = df.copy()

    df = df.rename(
        columns={
            "permno": "permno",
            "date": "date",
            "ticker": "symbol",
            "comnam": "company_name",
            "prc": "close",
            "openprc": "open",
            "askhi": "high",
            "bidlo": "low",
            "vol": "volume",
            "ret": "crsp_return",
            "shrout": "shares_outstanding",
            "shrcd": "share_code",
            "exchcd": "exchange_code",
            "siccd": "sic_code",
        }
    )

    df["date"] = pd.to_datetime(df["date"])

    # CRSP prices can be negative because of bid/ask conventions; use abs value.
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["shares_outstanding"] = pd.to_numeric(df["shares_outstanding"], errors="coerce")
    df["crsp_return"] = pd.to_numeric(df["crsp_return"], errors="coerce")

    df = df.dropna(subset=["date", "permno", "close", "volume"])
    df = df[df["close"] > 0]
    df = df[df["volume"] > 0]
    df = df.sort_values(["permno", "date"])

    # CRSP shrout is generally stored in thousands of shares.
    df["market_cap"] = df["close"] * df["shares_outstanding"] * 1000
    df["dollar_volume"] = df["close"] * df["volume"]

    exchange_map = {
        1: "NYSE",
        2: "AMEX",
        3: "NASDAQ",
    }
    df["exchange"] = df["exchange_code"].map(exchange_map)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features known at date T."""

    df = df.copy()
    df = df.sort_values(["permno", "date"])

    group = df.groupby("permno", group_keys=False)

    df["prev_close"] = group["close"].shift(1)
    df["return_1d"] = df["close"] / df["prev_close"] - 1

    df["avg_volume_20d"] = group["volume"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    df["avg_dollar_volume_20d"] = group["dollar_volume"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    df["relative_volume_20d"] = df["volume"] / df["avg_volume_20d"]

    price_range = df["high"] - df["low"]
    df["clv"] = np.where(
        price_range > 0,
        (df["close"] - df["low"]) / price_range,
        np.nan,
    )
    df["upper_shadow_ratio"] = np.where(
        price_range > 0,
        (df["high"] - df["close"]) / price_range,
        np.nan,
    )
    df["lower_shadow_ratio"] = np.where(
        price_range > 0,
        (df["close"] - df["low"]) / price_range,
        np.nan,
    )

    return df


def add_next_day_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate T+1 outcome variables.

    These columns are only for evaluation, not for screening.
    """

    df = df.copy()
    df = df.sort_values(["permno", "date"])

    group = df.groupby("permno", group_keys=False)

    df["next_date"] = group["date"].shift(-1)
    df["next_open"] = group["open"].shift(-1)
    df["next_high"] = group["high"].shift(-1)
    df["next_low"] = group["low"].shift(-1)
    df["next_close"] = group["close"].shift(-1)
    df["next_volume"] = group["volume"].shift(-1)
    df["next_dollar_volume"] = group["dollar_volume"].shift(-1)
    df["next_relative_volume_20d"] = group["relative_volume_20d"].shift(-1)

    df["next_open_return"] = df["next_open"] / df["close"] - 1
    df["next_high_return"] = df["next_high"] / df["close"] - 1
    df["next_low_return"] = df["next_low"] / df["close"] - 1
    df["next_close_return"] = df["next_close"] / df["close"] - 1
    df["next_intraday_return"] = df["next_close"] / df["next_open"] - 1

    df["next_volume_change"] = df["next_volume"] / df["volume"] - 1
    df["next_dollar_volume_change"] = df["next_dollar_volume"] / df["dollar_volume"] - 1

    return df


def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup before screening."""

    df = df.copy()

    df = df[df["return_1d"].between(-0.95, 5.0, inclusive="both") | df["return_1d"].isna()]
    df = df.sort_values(["date", "permno"])

    return df


def build_crsp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build screening features from raw CRSP daily data."""

    df = clean_crsp_daily(df)
    df = add_features(df)
    df = add_next_day_returns(df)
    df = final_clean(df)

    return df


def load_wrds_crsp_features(
    start_date: str,
    end_date: str,
    raw_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Download CRSP daily data from WRDS and build features in memory.

    This does not save processed parquet unless raw_output_path is provided
    for the raw download.
    """

    print(f"Downloading CRSP daily data from WRDS: {start_date} to {end_date}")

    df = download_crsp_daily(
        start_date=start_date,
        end_date=end_date,
        output_raw_path=raw_output_path,
    )

    print(f"Raw rows: {len(df):,}")

    df = build_crsp_features(df)

    print(f"Processed rows: {len(df):,}")

    return df


def preprocess_wrds_crsp(
    start_date: str,
    end_date: str,
    output_path: str | Path | None,
    raw_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Full preprocessing flow."""

    df = load_wrds_crsp_features(
        start_date=start_date,
        end_date=end_date,
        raw_output_path=raw_output_path,
    )

    if output_path is None:
        return df

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=str, required=True, help="Start date, e.g. 2020-01-01")
    parser.add_argument("--end", type=str, required=True, help="End date, e.g. 2025-12-31")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/crsp_daily_features.parquet",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="Optional path to save raw WRDS data.",
    )

    args = parser.parse_args()

    preprocess_wrds_crsp(
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        raw_output_path=args.raw_output,
    )


if __name__ == "__main__":
    main()
