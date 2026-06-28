# preprocess_premarket.py

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.wrds.preprocess import connect_wrds


WINDOWS = [
    ("pre_0400_0700", "04:00:00", "07:00:00"),
    ("pre_0700_0800", "07:00:00", "08:00:00"),
    ("pre_0800_0900", "08:00:00", "09:00:00"),
    ("pre_0900_0915", "09:00:00", "09:15:00"),
    ("pre_0915_0930", "09:15:00", "09:30:00"),
    ("open_0930_0935", "09:30:00", "09:35:00"),
    ("open_0935_1000", "09:35:00", "10:00:00"),
]


def normalize_taq_symbol(symbol: object) -> str | None:
    """Convert CRSP ticker text to a first-pass TAQ sym_root."""

    if pd.isna(symbol):
        return None

    symbol = str(symbol).strip().upper()
    if not symbol:
        return None

    parts = re.split(r"[^A-Z0-9]", symbol)
    return parts[0] if parts and parts[0] else None


def sql_string_list(values: list[str]) -> str:
    """Format a safe SQL string list for simple ticker symbols."""

    clean_values = []
    for value in values:
        value = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]+", value):
            continue
        clean_values.append(value)

    if not clean_values:
        return "''"

    return ", ".join(f"'{value}'" for value in sorted(set(clean_values)))


def load_candidates(path: str | Path) -> pd.DataFrame:
    """Load screen results and keep rows with a usable next_date and symbol."""

    path = Path(path)
    df = pd.read_csv(path)

    if "next_date" not in df.columns:
        raise ValueError(
            "Input candidates must contain next_date. "
            "Rerun tester/run_screen.py after the latest preprocessing changes."
        )

    df["date"] = pd.to_datetime(df["date"])
    df["next_date"] = pd.to_datetime(df["next_date"])
    df["taq_symbol"] = df["symbol"].map(normalize_taq_symbol)

    df = df[df["next_date"].notna()]
    df = df[df["taq_symbol"].notna()]
    df = df[df["next_date"].dt.year >= 2015]

    return df


def query_taq_trades(
    db,
    trade_date: pd.Timestamp,
    symbols: list[str],
) -> pd.DataFrame:
    """Query TAQ millisecond trades for one date and a small symbol list."""

    year = trade_date.year
    ymd = trade_date.strftime("%Y%m%d")
    schema = f"taqm_{year}"
    table = f"ctm_{ymd}"
    symbol_sql = sql_string_list(symbols)

    query = f"""
        SELECT
            date,
            time_m,
            sym_root,
            sym_suffix,
            tr_scond,
            size,
            price,
            tr_corr
        FROM {schema}.{table}
        WHERE time_m >= '04:00:00'
            AND time_m < '10:00:00'
            AND sym_root IN ({symbol_sql})
            AND price > 0
            AND size > 0
            AND (tr_corr IS NULL OR tr_corr = '00')
    """

    return db.raw_sql(query, date_cols=["date"])


def aggregate_trade_windows(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw TAQ trades into configured time windows."""

    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades["time_m"] = pd.to_timedelta(trades["time_m"].astype(str))
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["size"] = pd.to_numeric(trades["size"], errors="coerce")
    trades = trades.dropna(subset=["time_m", "sym_root", "price", "size"])
    trades = trades.sort_values(["sym_root", "time_m"])
    trades["dollar_volume"] = trades["price"] * trades["size"]

    rows = []
    for window_name, start, end in WINDOWS:
        start_td = pd.to_timedelta(start)
        end_td = pd.to_timedelta(end)
        window = trades[(trades["time_m"] >= start_td) & (trades["time_m"] < end_td)]

        if window.empty:
            continue

        grouped = window.groupby("sym_root", observed=True)
        summary = grouped.agg(
            window_open=("price", "first"),
            window_high=("price", "max"),
            window_low=("price", "min"),
            window_close=("price", "last"),
            window_volume=("size", "sum"),
            window_dollar_volume=("dollar_volume", "sum"),
            window_trade_count=("price", "count"),
            window_first_time=("time_m", "first"),
            window_last_time=("time_m", "last"),
        )
        summary = summary.reset_index()
        summary["window"] = window_name
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["window_return"] = result["window_close"] / result["window_open"] - 1

    return result


def pivot_window_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert one-row-per-window summary to one row per TAQ symbol."""

    if summary.empty:
        return pd.DataFrame()

    value_cols = [
        "window_open",
        "window_high",
        "window_low",
        "window_close",
        "window_volume",
        "window_dollar_volume",
        "window_trade_count",
        "window_return",
    ]

    wide = summary.pivot(index="sym_root", columns="window", values=value_cols)
    wide.columns = [f"{window}_{metric.replace('window_', '')}" for metric, window in wide.columns]
    wide = wide.reset_index().rename(columns={"sym_root": "taq_symbol"})

    return wide


def add_premarket_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add research-only signal columns from window summaries."""

    df = df.copy()

    for window_name, _, _ in WINDOWS:
        close_col = f"{window_name}_close"
        high_col = f"{window_name}_high"
        low_col = f"{window_name}_low"

        if close_col in df.columns and "close" in df.columns:
            df[f"{window_name}_return_from_t_close"] = df[close_col] / df["close"] - 1
        if close_col in df.columns and "next_open" in df.columns:
            df[f"{window_name}_to_next_open_return"] = df["next_open"] / df[close_col] - 1
        if close_col in df.columns and "next_close" in df.columns:
            df[f"{window_name}_to_next_close_return"] = df["next_close"] / df[close_col] - 1
        if close_col in df.columns and "next_high" in df.columns:
            df[f"{window_name}_to_next_high_return"] = df["next_high"] / df[close_col] - 1
        if close_col in df.columns and "next_low" in df.columns:
            df[f"{window_name}_to_next_low_return"] = df["next_low"] / df[close_col] - 1
        if high_col in df.columns and "close" in df.columns:
            df[f"{window_name}_high_from_t_close"] = df[high_col] / df["close"] - 1
        if low_col in df.columns and "close" in df.columns:
            df[f"{window_name}_low_from_t_close"] = df[low_col] / df["close"] - 1

    return df


def summarize_signal_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize correlations between premarket windows and next-day outcomes."""

    outcomes = [
        "next_open_return",
        "next_close_return",
        "next_high_return",
        "next_low_return",
    ]

    rows = []
    for window_name, _, _ in WINDOWS:
        signal_cols = [
            f"{window_name}_return",
            f"{window_name}_return_from_t_close",
            f"{window_name}_dollar_volume",
            f"{window_name}_trade_count",
        ]

        for signal_col in signal_cols:
            if signal_col not in df.columns:
                continue

            forward_outcomes = outcomes + [
                f"{window_name}_to_next_open_return",
                f"{window_name}_to_next_close_return",
                f"{window_name}_to_next_high_return",
                f"{window_name}_to_next_low_return",
            ]

            for outcome_col in forward_outcomes:
                if outcome_col not in df.columns:
                    continue

                valid = df[[signal_col, outcome_col]].dropna()
                if len(valid) < 2:
                    continue

                rows.append(
                    {
                        "window": window_name,
                        "signal": signal_col,
                        "outcome": outcome_col,
                        "count": len(valid),
                        "pearson_corr": valid[signal_col].corr(valid[outcome_col]),
                        "rank_corr": valid[signal_col].rank().corr(valid[outcome_col].rank()),
                    }
                )

    return pd.DataFrame(rows)


def summarize_by_signal_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket key premarket signals and summarize next-day outcomes."""

    rows = []
    bucket_specs = []

    for window_name, _, _ in WINDOWS:
        ret_col = f"{window_name}_return_from_t_close"
        dv_col = f"{window_name}_dollar_volume"

        if ret_col in df.columns:
            bucket_specs.append(
                (
                    window_name,
                    ret_col,
                    [-float("inf"), -0.02, 0, 0.02, 0.05, 0.10, float("inf")],
                    ["<-2%", "-2%-0%", "0%-2%", "2%-5%", "5%-10%", ">10%"],
                )
            )
        if dv_col in df.columns:
            bucket_specs.append(
                (
                    window_name,
                    dv_col,
                    [0, 50_000, 250_000, 1_000_000, 5_000_000, 20_000_000, float("inf")],
                    ["<50K", "50K-250K", "250K-1M", "1M-5M", "5M-20M", ">20M"],
                )
            )

    for window_name, col, bins, labels in bucket_specs:
        forward_cols = [
            f"{window_name}_to_next_open_return",
            f"{window_name}_to_next_close_return",
            f"{window_name}_to_next_high_return",
            f"{window_name}_to_next_low_return",
        ]
        keep_cols = [
            col,
            "next_open_return",
            "next_close_return",
            "next_high_return",
            "next_low_return",
            *[forward_col for forward_col in forward_cols if forward_col in df.columns],
        ]

        temp = df[keep_cols].copy()
        temp["bucket"] = pd.cut(temp[col], bins=bins, labels=labels, right=False)
        temp = temp.dropna(subset=["bucket"])

        if temp.empty:
            continue

        summary = (
            temp.groupby("bucket", observed=True)
            .agg(
                count=(col, "count"),
                avg_signal=(col, "mean"),
                avg_next_open_return=("next_open_return", "mean"),
                win_rate_next_open=("next_open_return", lambda x: (x > 0).mean()),
                avg_next_close_return=("next_close_return", "mean"),
                win_rate_next_close=("next_close_return", lambda x: (x > 0).mean()),
                avg_next_high_return=("next_high_return", "mean"),
                avg_next_low_return=("next_low_return", "mean"),
            )
            .reset_index()
        )

        for forward_col in forward_cols:
            if forward_col not in temp.columns:
                continue

            forward_summary = (
                temp.groupby("bucket", observed=True)
                .agg(
                    **{
                        f"avg_{forward_col}": (forward_col, "mean"),
                        f"win_rate_{forward_col}": (forward_col, lambda x: (x > 0).mean()),
                    }
                )
                .reset_index()
            )
            summary = summary.merge(forward_summary, on="bucket", how="left")

        summary.insert(0, "signal", col)
        summary.insert(0, "window", window_name)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def enrich_candidates_with_taq(
    candidates_path: str | Path,
    output_dir: str | Path | None = None,
    max_dates: int | None = None,
) -> pd.DataFrame:
    """Add TAQ premarket/open-window summaries to screen candidates."""

    candidates = load_candidates(candidates_path)
    if candidates.empty:
        raise ValueError("No usable candidates found for TAQ enrichment.")

    candidates_path = Path(candidates_path)
    output_dir = Path(output_dir) if output_dir is not None else candidates_path.parent / "premarket"
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_dir = output_dir / "daily_taq_windows"
    daily_dir.mkdir(parents=True, exist_ok=True)

    db = connect_wrds()
    rows = []

    grouped = list(candidates.groupby(candidates["next_date"].dt.date, sort=True))
    if max_dates is not None:
        grouped = grouped[:max_dates]

    try:
        for i, (trade_date, daily_candidates) in enumerate(grouped, start=1):
            trade_date_ts = pd.Timestamp(trade_date)
            daily_path = daily_dir / f"premarket_{trade_date_ts.strftime('%Y%m%d')}.csv"

            if daily_path.exists():
                enriched = pd.read_csv(daily_path, parse_dates=["date", "next_date"])
                enriched = add_premarket_signal_columns(enriched)
                enriched.to_csv(daily_path, index=False)
                rows.append(enriched)
                print(f"[{i}/{len(grouped)}] Loaded cached {trade_date_ts.date()}: {len(enriched):,}")
                continue

            symbols = sorted(daily_candidates["taq_symbol"].dropna().unique().tolist())
            if not symbols:
                continue

            try:
                trades = query_taq_trades(db, trade_date_ts, symbols)
            except Exception as exc:
                print(f"[{i}/{len(grouped)}] Skipped {trade_date_ts.date()} due to TAQ query error: {exc}")
                continue

            summary = aggregate_trade_windows(trades)
            wide = pivot_window_summary(summary)

            enriched = daily_candidates.merge(wide, on="taq_symbol", how="left")
            enriched = add_premarket_signal_columns(enriched)
            enriched.to_csv(daily_path, index=False)
            rows.append(enriched)

            print(
                f"[{i}/{len(grouped)}] {trade_date_ts.date()} "
                f"candidates={len(daily_candidates):,} trades={len(trades):,} enriched={len(enriched):,}"
            )
    finally:
        db.close()

    if not rows:
        raise ValueError("No TAQ rows were produced.")

    result = pd.concat(rows, ignore_index=True)

    result_path = output_dir / "premarket_enriched_results.csv"
    result.to_csv(result_path, index=False)
    print(f"Saved enriched results to: {result_path}")

    corr = summarize_signal_relationships(result)
    if not corr.empty:
        corr_path = output_dir / "premarket_signal_relationship_summary.csv"
        corr.to_csv(corr_path, index=False)
        print(f"Saved signal relationship summary to: {corr_path}")

    bucket_summary = summarize_by_signal_bucket(result)
    if not bucket_summary.empty:
        bucket_path = output_dir / "premarket_signal_bucket_summary.csv"
        bucket_summary.to_csv(bucket_path, index=False)
        print(f"Saved signal bucket summary to: {bucket_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidates", type=str, required=True, help="Path to all_screen_results.csv.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-dates", type=int, default=None, help="Optional small test limit.")

    args = parser.parse_args()

    enrich_candidates_with_taq(
        candidates_path=args.candidates,
        output_dir=args.output_dir,
        max_dates=args.max_dates,
    )


if __name__ == "__main__":
    main()
