from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.wrds.preprocess import connect_wrds
from data.wrds.preprocess_premarket import normalize_taq_symbol, sql_string_list


PULLBACKS = [0.00, 0.01, 0.02, 0.03, 0.04]
ENTRY_DEADLINES = ["08:30:00", "09:00:00", "09:15:00", "09:30:00"]
MIN_TOUCH_DOLLAR_VOLUMES = [0, 5_000, 25_000]
ENTRY_START = "08:01:00"
RETURN_BANDS = [("2%-5%", 0.02, 0.05), ("5%-10%", 0.05, 0.10)]
MIN_DOLLAR_VOLUMES = [250_000, 500_000, 1_000_000]


def load_candidates(path: str | Path) -> pd.DataFrame:
    usecols = [
        "date",
        "next_date",
        "symbol",
        "permno",
        "market_cap",
        "close",
        "return_1d",
        "dollar_volume",
        "entry_time",
        "confirm_return",
        "cum_dollar_volume",
        "cum_trade_count",
        "pre_0700_0800_close",
    ]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["date", "next_date"])
    df = df[
        (df["entry_time"] == "08:00")
        & df["confirm_return"].between(0.02, 0.10, inclusive="left")
        & df["cum_dollar_volume"].ge(250_000)
        & df["cum_trade_count"].ge(25)
        & df["market_cap"].between(300_000_000, 5_000_000_000, inclusive="both")
    ].copy()
    df["taq_symbol"] = df["symbol"].map(normalize_taq_symbol)
    return df[df["taq_symbol"].notna() & df["pre_0700_0800_close"].gt(0)]


def query_trades(db, trade_date: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
    schema = f"taqm_{trade_date.year}"
    table = f"ctm_{trade_date.strftime('%Y%m%d')}"
    query = f"""
        SELECT date, time_m, sym_root, sym_suffix, size, price, tr_corr, tr_seqnum
        FROM {schema}.{table}
        WHERE time_m >= '08:00:00'
          AND time_m < '10:00:01'
          AND sym_root IN ({sql_string_list(symbols)})
          AND COALESCE(TRIM(sym_suffix), '') = ''
          AND price > 0
          AND size > 0
          AND (tr_corr IS NULL OR tr_corr = '00')
    """
    return db.raw_sql(query, date_cols=["date"])


def prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["time_m"] = pd.to_timedelta(df["time_m"].astype(str))
    for column in ["price", "size", "tr_seqnum"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["time_m", "sym_root", "price", "size"])
    return df.sort_values(["sym_root", "time_m", "tr_seqnum"], na_position="last")


def load_or_query(
    db,
    cache_dir: Path,
    trade_date: pd.Timestamp,
    symbols: list[str],
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"taq_common_0800_1000_{trade_date.strftime('%Y%m%d')}.csv.gz"
    if path.exists():
        return prepare_trades(pd.read_csv(path, parse_dates=["date"]))
    df = query_trades(db, trade_date, symbols)
    df.to_csv(path, index=False, compression="gzip")
    return prepare_trades(df)


def last_price_at(trades: pd.DataFrame, end: str) -> float | None:
    eligible = trades[trades["time_m"] <= pd.to_timedelta(end)]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1]["price"])


def simulate_rule(
    candidate: pd.Series,
    trades: pd.DataFrame,
    pullback: float,
    deadline: str,
    min_touch_dollar_volume: float,
    slippage_bps: float,
) -> dict[str, object] | None:
    reference_price = float(candidate["pre_0700_0800_close"])
    deadline_td = pd.to_timedelta(deadline)
    eligible = trades[
        (trades["time_m"] >= pd.to_timedelta(ENTRY_START))
        & (trades["time_m"] <= deadline_td)
    ]
    if eligible.empty:
        return None

    limit_price = reference_price * (1 - pullback)
    if pullback == 0:
        trigger = eligible.iloc[0]
        entry_price = float(trigger["price"]) * (1 + slippage_bps / 10_000)
    else:
        fills = eligible[eligible["price"] <= limit_price].copy()
        if fills.empty:
            return None
        fills["touch_dollar_volume"] = (fills["price"] * fills["size"]).cumsum()
        confirmed = fills[
            fills["touch_dollar_volume"].ge(min_touch_dollar_volume)
        ]
        if confirmed.empty:
            return None
        trigger = confirmed.iloc[0]
        # Conservative limit-fill assumption: pay the submitted limit, not a better print.
        entry_price = limit_price

    entry_time = trigger["time_m"]
    after_entry = trades[trades["time_m"] > entry_time]
    if after_entry.empty:
        return None

    row: dict[str, object] = {
        "date": candidate["date"],
        "trade_date": candidate["next_date"],
        "symbol": candidate["symbol"],
        "permno": candidate["permno"],
        "market_cap": candidate["market_cap"],
        "close": candidate["close"],
        "return_1d": candidate["return_1d"],
        "t_day_dollar_volume": candidate["dollar_volume"],
        "confirm_return": candidate["confirm_return"],
        "cum_dollar_volume": candidate["cum_dollar_volume"],
        "cum_trade_count": candidate["cum_trade_count"],
        "reference_price_0800": reference_price,
        "entry_start": ENTRY_START,
        "pullback": pullback,
        "entry_deadline": deadline,
        "min_touch_dollar_volume": min_touch_dollar_volume,
        "entry_time": entry_time,
        "entry_price": entry_price,
    }

    for horizon, end in [
        ("0930", "09:30:00"),
        ("0935", "09:35:00"),
        ("1000", "10:00:00"),
    ]:
        path = after_entry[after_entry["time_m"] <= pd.to_timedelta(end)]
        if path.empty:
            row[f"return_{horizon}"] = np.nan
            row[f"mfe_{horizon}"] = np.nan
            row[f"mae_{horizon}"] = np.nan
            continue
        exit_price = float(path.iloc[-1]["price"])
        row[f"return_{horizon}"] = exit_price / entry_price - 1
        row[f"mfe_{horizon}"] = float(path["price"].max()) / entry_price - 1
        row[f"mae_{horizon}"] = float(path["price"].min()) / entry_price - 1

    return row


def run_candidate(candidate: pd.Series, trades: pd.DataFrame, slippage_bps: float) -> list[dict[str, object]]:
    symbol_trades = trades[trades["sym_root"] == candidate["taq_symbol"]]
    if symbol_trades.empty:
        return []
    rows = []
    for pullback in PULLBACKS:
        for deadline in ENTRY_DEADLINES:
            touch_thresholds = [0] if pullback == 0 else MIN_TOUCH_DOLLAR_VOLUMES
            for min_touch_dollar_volume in touch_thresholds:
                result = simulate_rule(
                    candidate,
                    symbol_trades,
                    pullback,
                    deadline,
                    min_touch_dollar_volume,
                    slippage_bps,
                )
                if result is not None:
                    rows.append(result)
    return rows


def trimmed_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return clean.clip(clean.quantile(0.05), clean.quantile(0.95)).mean()


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period_name, period in [
        ("train_2015_2021", results[results["trade_date"].dt.year <= 2021]),
        ("validation_2022_2024", results[results["trade_date"].dt.year >= 2022]),
        ("all_2015_2024", results),
    ]:
        for band_name, lower, upper in RETURN_BANDS:
            for min_dv in MIN_DOLLAR_VOLUMES:
                selected = period[
                    period["confirm_return"].between(lower, upper, inclusive="left")
                    & period["cum_dollar_volume"].ge(min_dv)
                ]
                grouped = selected.groupby(
                    ["pullback", "entry_deadline", "min_touch_dollar_volume"],
                    observed=True,
                )
                for (pullback, deadline, min_touch_dollar_volume), group in grouped:
                    if group.empty:
                        continue
                    rows.append(
                        {
                            "period": period_name,
                            "return_band": band_name,
                            "min_cum_dollar_volume": min_dv,
                            "pullback": pullback,
                            "entry_deadline": deadline,
                            "min_touch_dollar_volume": min_touch_dollar_volume,
                            "count": len(group),
                            "fill_candidates": selected[
                                ["trade_date", "symbol"]
                            ].drop_duplicates().shape[0],
                            "trimmed_return_0930": trimmed_mean(group["return_0930"]),
                            "median_return_0930": group["return_0930"].median(),
                            "win_rate_0930": group["return_0930"].gt(0).mean(),
                            "median_mfe_0930": group["mfe_0930"].median(),
                            "median_mae_0930": group["mae_0930"].median(),
                            "mae_25pct_0930": group["mae_0930"].quantile(0.25),
                            "trimmed_return_0935": trimmed_mean(group["return_0935"]),
                            "trimmed_return_1000": trimmed_mean(group["return_1000"]),
                        }
                    )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["fill_rate"] = summary["count"] / summary["fill_candidates"].replace(0, np.nan)
    return summary.sort_values(
        ["period", "trimmed_return_0930", "median_mae_0930"],
        ascending=[True, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    if args.start_date:
        candidates = candidates[candidates["next_date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        candidates = candidates[candidates["next_date"] <= pd.Timestamp(args.end_date)]

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "raw_trade_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = list(candidates.groupby(candidates["next_date"].dt.date, sort=True))
    if args.max_dates:
        grouped = grouped[: args.max_dates]

    db = connect_wrds()
    rows: list[dict[str, object]] = []
    try:
        for index, (trade_date, daily) in enumerate(grouped, start=1):
            trade_date_ts = pd.Timestamp(trade_date)
            symbols = sorted(daily["taq_symbol"].unique().tolist())
            trades = None
            for attempt in range(1, 4):
                try:
                    trades = load_or_query(db, cache_dir, trade_date_ts, symbols)
                    break
                except Exception as exc:
                    print(
                        f"[{index}/{len(grouped)}] {trade_date} "
                        f"attempt {attempt}/3 failed: {exc}",
                        flush=True,
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = connect_wrds()
            if trades is None:
                print(f"[{index}/{len(grouped)}] skip {trade_date} after 3 attempts", flush=True)
                continue
            for _, candidate in daily.iterrows():
                rows.extend(run_candidate(candidate, trades, args.slippage_bps))
            if index == 1 or index % 20 == 0 or index == len(grouped):
                print(
                    f"[{index}/{len(grouped)}] {trade_date} "
                    f"symbols={len(symbols)} trades={len(trades):,} results={len(rows):,}",
                    flush=True,
                )
    finally:
        db.close()

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "strict_entry_results.csv", index=False)
    if not results.empty:
        results["trade_date"] = pd.to_datetime(results["trade_date"])
    summary = summarize(results)
    summary.to_csv(output_dir / "strict_entry_summary.csv", index=False)
    print(f"Saved strict results to: {output_dir}")


if __name__ == "__main__":
    main()
