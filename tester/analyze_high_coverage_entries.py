from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.wrds.preprocess_premarket import normalize_taq_symbol


FIXED_TIMES = ["08:01:00", "08:05:00", "08:15:00", "08:30:00", "09:00:00"]
PULLBACKS = [0.01, 0.02, 0.03]
FALLBACK_TIMES = ["08:15:00", "08:30:00", "09:00:00"]
MIN_TOUCH_DOLLAR_VOLUME = 5_000


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
        & df["cum_dollar_volume"].ge(1_000_000)
        & df["cum_trade_count"].ge(100)
        & df["market_cap"].between(300_000_000, 5_000_000_000, inclusive="both")
        & df["pre_0700_0800_close"].gt(0)
    ].copy()
    df["taq_symbol"] = df["symbol"].map(normalize_taq_symbol)
    return df[df["taq_symbol"].notna()]


def build_cache_map(root: str | Path) -> dict[str, Path]:
    files = Path(root).glob("strict_entry_timing*/raw_trade_cache/taq_common_0800_1000_*.csv.gz")
    cache_map: dict[str, Path] = {}
    for path in files:
        date_key = path.stem.split("_")[-1].split(".")[0]
        cache_map[date_key] = path
    return cache_map


def prepare_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["time_m"] = pd.to_timedelta(df["time_m"].astype(str))
    for column in ["price", "size", "tr_seqnum"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return (
        df.dropna(subset=["time_m", "sym_root", "price", "size"])
        .sort_values(["sym_root", "time_m", "tr_seqnum"], na_position="last")
    )


def first_trade_at_or_after(trades: pd.DataFrame, time_text: str) -> pd.Series | None:
    eligible = trades[trades["time_m"] >= pd.to_timedelta(time_text)]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def fixed_entry(
    trades: pd.DataFrame,
    time_text: str,
    slippage_bps: float,
) -> tuple[pd.Series, float, str] | None:
    trade = first_trade_at_or_after(trades, time_text)
    if trade is None:
        return None
    fill = float(trade["price"]) * (1 + slippage_bps / 10_000)
    return trade, fill, "market"


def pullback_with_fallback(
    trades: pd.DataFrame,
    reference_price: float,
    pullback: float,
    fallback_time: str,
    slippage_bps: float,
) -> tuple[pd.Series, float, str] | None:
    start = pd.to_timedelta("08:01:00")
    fallback_td = pd.to_timedelta(fallback_time)
    before_fallback = trades[
        (trades["time_m"] >= start) & (trades["time_m"] < fallback_td)
    ].copy()
    limit_price = reference_price * (1 - pullback)
    touches = before_fallback[before_fallback["price"] <= limit_price].copy()
    if not touches.empty:
        touches["touch_dollar_volume"] = (touches["price"] * touches["size"]).cumsum()
        confirmed = touches[
            touches["touch_dollar_volume"].ge(MIN_TOUCH_DOLLAR_VOLUME)
        ]
        if not confirmed.empty:
            return confirmed.iloc[0], limit_price, "limit"

    fallback = first_trade_at_or_after(trades, fallback_time)
    if fallback is None:
        return None
    fill = float(fallback["price"]) * (1 + slippage_bps / 10_000)
    return fallback, fill, "fallback_market"


def outcome(
    candidate: pd.Series,
    trades: pd.DataFrame,
    method: str,
    entry: tuple[pd.Series, float, str],
) -> dict[str, object] | None:
    trigger, entry_price, fill_type = entry
    entry_time = trigger["time_m"]
    after = trades[trades["time_m"] > entry_time]
    if after.empty:
        return None

    row: dict[str, object] = {
        "date": candidate["date"],
        "trade_date": candidate["next_date"],
        "symbol": candidate["symbol"],
        "permno": candidate["permno"],
        "market_cap": candidate["market_cap"],
        "t_day_dollar_volume": candidate["dollar_volume"],
        "return_1d": candidate["return_1d"],
        "confirm_return": candidate["confirm_return"],
        "cum_dollar_volume": candidate["cum_dollar_volume"],
        "cum_trade_count": candidate["cum_trade_count"],
        "reference_price_0800": candidate["pre_0700_0800_close"],
        "method": method,
        "fill_type": fill_type,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "entry_premium": entry_price / candidate["close"] - 1,
    }

    for horizon, end in [
        ("0930", "09:30:00"),
        ("0935", "09:35:00"),
        ("1000", "10:00:00"),
    ]:
        path = after[after["time_m"] <= pd.to_timedelta(end)]
        if path.empty:
            row[f"return_{horizon}"] = np.nan
            row[f"mfe_{horizon}"] = np.nan
            row[f"mae_{horizon}"] = np.nan
            continue
        high_index = path["price"].idxmax()
        row[f"return_{horizon}"] = float(path.iloc[-1]["price"]) / entry_price - 1
        row[f"mfe_{horizon}"] = float(path["price"].max()) / entry_price - 1
        row[f"mae_{horizon}"] = float(path["price"].min()) / entry_price - 1
        row[f"time_to_high_{horizon}_min"] = (
            path.loc[high_index, "time_m"] - entry_time
        ).total_seconds() / 60
        if horizon == "0930":
            returns = path["price"] / entry_price - 1
            for threshold in [0.02, 0.03, 0.05]:
                up = path[returns >= threshold]
                row[f"hit_up_{int(threshold * 100)}pct"] = not up.empty
                row[f"time_up_{int(threshold * 100)}pct_min"] = (
                    (up.iloc[0]["time_m"] - entry_time).total_seconds() / 60
                    if not up.empty
                    else np.nan
                )
            down = path[returns <= -0.02]
            row["hit_down_2pct"] = not down.empty
            first_down_time = down.iloc[0]["time_m"] if not down.empty else None
            for threshold in [0.02, 0.03, 0.05]:
                up = path[returns >= threshold]
                first_up_time = up.iloc[0]["time_m"] if not up.empty else None
                row[f"up_{int(threshold * 100)}pct_before_down_2pct"] = (
                    first_up_time is not None
                    and (first_down_time is None or first_up_time < first_down_time)
                )
    return row


def run_candidate(
    candidate: pd.Series,
    trades: pd.DataFrame,
    slippage_bps: float,
) -> list[dict[str, object]]:
    symbol_trades = trades[trades["sym_root"] == candidate["taq_symbol"]]
    if symbol_trades.empty:
        return []

    entries: list[tuple[str, tuple[pd.Series, float, str] | None]] = []
    for time_text in FIXED_TIMES:
        entries.append(
            (
                f"fixed_{time_text.replace(':', '')}",
                fixed_entry(symbol_trades, time_text, slippage_bps),
            )
        )
    for pullback in PULLBACKS:
        for fallback_time in FALLBACK_TIMES:
            entries.append(
                (
                    f"pb{int(pullback * 100)}_fallback_{fallback_time.replace(':', '')}",
                    pullback_with_fallback(
                        symbol_trades,
                        float(candidate["pre_0700_0800_close"]),
                        pullback,
                        fallback_time,
                        slippage_bps,
                    ),
                )
            )

    rows = []
    for method, entry in entries:
        if entry is None:
            continue
        result = outcome(candidate, symbol_trades, method, entry)
        if result is not None:
            rows.append(result)
    return rows


def trimmed_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return clean.clip(clean.quantile(0.05), clean.quantile(0.95)).mean()


def summarize(results: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = [
        (
            "train_2015_2021",
            results[results["trade_date"].dt.year <= 2021],
            candidates[candidates["next_date"].dt.year <= 2021],
        ),
        (
            "validation_2022_2024",
            results[results["trade_date"].dt.year >= 2022],
            candidates[candidates["next_date"].dt.year >= 2022],
        ),
        ("all_2015_2024", results, candidates),
    ]
    for period_name, period, period_candidates in periods:
        denominator = period_candidates[["next_date", "symbol"]].drop_duplicates().shape[0]
        for method, group in period.groupby("method", observed=True):
            rows.append(
                {
                    "period": period_name,
                    "method": method,
                    "candidate_count": denominator,
                    "entry_count": len(group),
                    "coverage": len(group) / denominator if denominator else np.nan,
                    "limit_fill_rate": group["fill_type"].eq("limit").mean(),
                    "median_entry_time": group["entry_time"].median(),
                    "avg_entry_premium": group["entry_premium"].mean(),
                    "trimmed_return_0930": trimmed_mean(group["return_0930"]),
                    "median_return_0930": group["return_0930"].median(),
                    "win_rate_0930": group["return_0930"].gt(0).mean(),
                    "median_mfe_0930": group["mfe_0930"].median(),
                    "median_mae_0930": group["mae_0930"].median(),
                    "mae_25pct_0930": group["mae_0930"].quantile(0.25),
                    "median_time_to_high_0930_min": group[
                        "time_to_high_0930_min"
                    ].median(),
                    "p_hit_up_2pct": group["hit_up_2pct"].mean(),
                    "p_hit_up_3pct": group["hit_up_3pct"].mean(),
                    "p_hit_up_5pct": group["hit_up_5pct"].mean(),
                    "p_up_2_before_down_2": group[
                        "up_2pct_before_down_2pct"
                    ].mean(),
                    "p_up_3_before_down_2": group[
                        "up_3pct_before_down_2pct"
                    ].mean(),
                    "median_time_up_3pct_min": group["time_up_3pct_min"].median(),
                    "trimmed_return_0935": trimmed_mean(group["return_0935"]),
                    "trimmed_return_1000": trimmed_mean(group["return_1000"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    cache_map = build_cache_map(args.cache_root)
    rows: list[dict[str, object]] = []
    grouped = list(candidates.groupby(candidates["next_date"].dt.date, sort=True))
    for index, (trade_date, daily) in enumerate(grouped, start=1):
        date_key = pd.Timestamp(trade_date).strftime("%Y%m%d")
        path = cache_map.get(date_key)
        if path is None:
            print(f"Missing cache for {trade_date}")
            continue
        trades = prepare_trades(path)
        for _, candidate in daily.iterrows():
            rows.extend(run_candidate(candidate, trades, args.slippage_bps))
        if index == 1 or index % 50 == 0 or index == len(grouped):
            print(f"[{index}/{len(grouped)}] results={len(rows):,}", flush=True)

    results = pd.DataFrame(rows)
    results["trade_date"] = pd.to_datetime(results["trade_date"])
    summary = summarize(results, candidates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "high_coverage_entry_results.csv", index=False)
    summary.to_csv(output_dir / "high_coverage_entry_summary.csv", index=False)
    print(f"Candidates: {len(candidates):,}")
    print(f"Saved high-coverage analysis to: {output_dir}")


if __name__ == "__main__":
    main()
