from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.wrds.preprocess_premarket import normalize_taq_symbol


ENTRY_START = pd.to_timedelta("08:01:00")
ENTRY_END = pd.to_timedelta("09:00:00")
MIN_TOUCH_DOLLAR_VOLUME = 5_000


@dataclass(frozen=True)
class StagedRule:
    name: str
    anchor: str
    offsets: tuple[float, ...]
    weights: tuple[float, ...]


RULES = [
    StagedRule("reference_equal_4", "reference", (0.00, -0.01, -0.02, -0.03), (0.25, 0.25, 0.25, 0.25)),
    StagedRule("reference_weighted_4", "reference", (0.00, -0.01, -0.02, -0.03), (0.10, 0.20, 0.30, 0.40)),
    StagedRule("reference_wide_4", "reference", (0.00, -0.02, -0.04, -0.06), (0.10, 0.20, 0.30, 0.40)),
    StagedRule("starter_equal_4", "starter", (0.00, -0.01, -0.02, -0.03), (0.25, 0.25, 0.25, 0.25)),
    StagedRule("starter_weighted_4", "starter", (0.00, -0.01, -0.02, -0.03), (0.10, 0.20, 0.30, 0.40)),
    StagedRule("starter_wide_4", "starter", (0.00, -0.02, -0.04, -0.06), (0.10, 0.20, 0.30, 0.40)),
    StagedRule("starter_weighted_3", "starter", (0.00, -0.015, -0.03), (0.20, 0.30, 0.50)),
]


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
    cache_map: dict[str, Path] = {}
    for path in Path(root).glob(
        "strict_entry_timing*/raw_trade_cache/taq_common_0800_1000_*.csv.gz"
    ):
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


def find_limit_fill(
    trades: pd.DataFrame,
    level: float,
    after_time: pd.Timedelta,
) -> pd.Series | None:
    eligible = trades[
        (trades["time_m"] >= after_time)
        & (trades["time_m"] <= ENTRY_END)
        & (trades["price"] <= level)
    ].copy()
    if eligible.empty:
        return None
    eligible["touch_dollar_volume"] = (eligible["price"] * eligible["size"]).cumsum()
    confirmed = eligible[
        eligible["touch_dollar_volume"].ge(MIN_TOUCH_DOLLAR_VOLUME)
    ]
    if confirmed.empty:
        return None
    return confirmed.iloc[0]


def build_fills(
    trades: pd.DataFrame,
    reference_price: float,
    rule: StagedRule,
    slippage_bps: float,
) -> list[dict[str, float | pd.Timedelta]]:
    window = trades[
        (trades["time_m"] >= ENTRY_START) & (trades["time_m"] <= ENTRY_END)
    ]
    if window.empty:
        return []

    fills: list[dict[str, float | pd.Timedelta]] = []
    next_time = ENTRY_START

    if rule.anchor == "starter":
        starter = window.iloc[0]
        anchor_price = float(starter["price"]) * (1 + slippage_bps / 10_000)
        fills.append(
            {
                "time": starter["time_m"],
                "price": anchor_price,
                "weight": rule.weights[0],
                "offset": rule.offsets[0],
            }
        )
        next_time = starter["time_m"]
        start_index = 1
    else:
        anchor_price = reference_price
        start_index = 0

    for index in range(start_index, len(rule.offsets)):
        level = anchor_price * (1 + rule.offsets[index])
        fill = find_limit_fill(trades, level, next_time)
        if fill is None:
            continue
        fills.append(
            {
                "time": fill["time_m"],
                "price": level,
                "weight": rule.weights[index],
                "offset": rule.offsets[index],
            }
        )
        next_time = fill["time_m"]

    return fills


def position_snapshot(
    fills: list[dict[str, float | pd.Timedelta]],
    price: float,
) -> tuple[float, float, float]:
    invested = sum(float(fill["weight"]) for fill in fills)
    if invested == 0:
        return np.nan, 0.0, 0.0
    shares = sum(float(fill["weight"]) / float(fill["price"]) for fill in fills)
    average_price = invested / shares
    pnl_on_invested = price / average_price - 1
    pnl_on_total_capital = shares * price - invested
    return average_price, pnl_on_invested, pnl_on_total_capital


def simulate(
    candidate: pd.Series,
    trades: pd.DataFrame,
    rule: StagedRule,
    slippage_bps: float,
) -> dict[str, object] | None:
    fills = build_fills(
        trades,
        float(candidate["pre_0700_0800_close"]),
        rule,
        slippage_bps,
    )
    if not fills:
        return None

    final_fill_time = max(fill["time"] for fill in fills)
    invested_fraction = sum(float(fill["weight"]) for fill in fills)
    shares = sum(float(fill["weight"]) / float(fill["price"]) for fill in fills)
    average_price = invested_fraction / shares
    after = trades[trades["time_m"] > final_fill_time]
    if after.empty:
        return None

    row: dict[str, object] = {
        "date": candidate["date"],
        "trade_date": candidate["next_date"],
        "symbol": candidate["symbol"],
        "market_cap": candidate["market_cap"],
        "confirm_return": candidate["confirm_return"],
        "cum_dollar_volume": candidate["cum_dollar_volume"],
        "cum_trade_count": candidate["cum_trade_count"],
        "rule": rule.name,
        "anchor": rule.anchor,
        "tranche_count": len(fills),
        "invested_fraction": invested_fraction,
        "average_entry_price": average_price,
        "first_fill_time": min(fill["time"] for fill in fills),
        "final_fill_time": final_fill_time,
        "last_fill_offset": min(float(fill["offset"]) for fill in fills),
    }

    for horizon, end in [
        ("0930", "09:30:00"),
        ("0935", "09:35:00"),
        ("1000", "10:00:00"),
    ]:
        path = after[after["time_m"] <= pd.to_timedelta(end)]
        if path.empty:
            continue
        exit_price = float(path.iloc[-1]["price"])
        _, invested_return, capital_return = position_snapshot(fills, exit_price)
        row[f"return_on_invested_{horizon}"] = invested_return
        row[f"return_on_capital_{horizon}"] = capital_return
        row[f"mfe_{horizon}"] = float(path["price"].max()) / average_price - 1
        row[f"mae_{horizon}"] = float(path["price"].min()) / average_price - 1
    return row


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
        denominator = len(period_candidates)
        for rule, group in period.groupby("rule", observed=True):
            rows.append(
                {
                    "period": period_name,
                    "rule": rule,
                    "candidate_count": denominator,
                    "entry_count": len(group),
                    "coverage": len(group) / denominator,
                    "avg_invested_fraction": group["invested_fraction"].mean(),
                    "median_invested_fraction": group["invested_fraction"].median(),
                    "full_position_rate": group["invested_fraction"].ge(0.999).mean(),
                    "avg_tranche_count": group["tranche_count"].mean(),
                    "median_average_entry_price": group["average_entry_price"].median(),
                    "trimmed_invested_return_0930": trimmed_mean(
                        group["return_on_invested_0930"]
                    ),
                    "median_invested_return_0930": group[
                        "return_on_invested_0930"
                    ].median(),
                    "trimmed_capital_return_0930": trimmed_mean(
                        group["return_on_capital_0930"]
                    ),
                    "median_mfe_0930": group["mfe_0930"].median(),
                    "median_mae_0930": group["mae_0930"].median(),
                    "mae_25pct_0930": group["mae_0930"].quantile(0.25),
                    "trimmed_capital_return_0935": trimmed_mean(
                        group["return_on_capital_0935"]
                    ),
                    "trimmed_capital_return_1000": trimmed_mean(
                        group["return_on_capital_1000"]
                    ),
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
        trades = prepare_trades(cache_map[date_key])
        for _, candidate in daily.iterrows():
            symbol_trades = trades[trades["sym_root"] == candidate["taq_symbol"]]
            for rule in RULES:
                result = simulate(candidate, symbol_trades, rule, args.slippage_bps)
                if result is not None:
                    rows.append(result)
        if index == 1 or index % 50 == 0 or index == len(grouped):
            print(f"[{index}/{len(grouped)}] results={len(rows):,}", flush=True)

    results = pd.DataFrame(rows)
    results["trade_date"] = pd.to_datetime(results["trade_date"])
    summary = summarize(results, candidates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "staged_entry_results.csv", index=False)
    summary.to_csv(output_dir / "staged_entry_summary.csv", index=False)
    print(f"Saved staged-entry analysis to: {output_dir}")


if __name__ == "__main__":
    main()
