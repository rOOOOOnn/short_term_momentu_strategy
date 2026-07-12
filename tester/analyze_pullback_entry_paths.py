from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("screen_results/20150102_20241231_gainers_return_1d_top50/premarket")
DEFAULT_TRIGGER_DIR = ROOT / "full_premarket_dynamic_triggers"
DEFAULT_OUTPUT_DIR = ROOT / "pullback_entry_paths"

PULLBACKS = [0.01, 0.02, 0.03, 0.05]
TAKE_PROFITS = [0.03, 0.05]
STOP_LOSSES = [-0.02, -0.03]


def parse_time_delta(value: object) -> pd.Timedelta:
    text = str(value)
    if text.startswith("0 days "):
        text = text.replace("0 days ", "", 1)
    return pd.to_timedelta(text)


def prepare_trades(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    usecols = [col for col in ["date", "time_m", "sym_root", "price", "tr_seqnum"] if col in header]
    trades = pd.read_csv(path, usecols=usecols, parse_dates=["date"])
    if trades.empty:
        return trades
    trades["time_m"] = pd.to_timedelta(trades["time_m"].astype(str))
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    if "tr_seqnum" in trades.columns:
        trades["tr_seqnum"] = pd.to_numeric(trades["tr_seqnum"], errors="coerce")
    trades = trades.dropna(subset=["sym_root", "time_m", "price"])
    sort_cols = ["sym_root", "time_m"]
    if "tr_seqnum" in trades.columns:
        sort_cols.append("tr_seqnum")
    return trades.sort_values(sort_cols).reset_index(drop=True)


def first_hit(path: pd.DataFrame, entry_price: float, threshold: float, direction: str) -> pd.Timedelta | None:
    returns = path["price"] / entry_price - 1
    hit = path[returns.ge(threshold)] if direction == "up" else path[returns.le(threshold)]
    if hit.empty:
        return None
    return hit.iloc[0]["time_m"]


def simulate_pullback(event: pd.Series, symbol_trades: pd.DataFrame, pullback: float) -> dict[str, object] | None:
    trigger_time = parse_time_delta(event["trigger_time"])
    trigger_price = float(event["trigger_price"])
    after_trigger = symbol_trades[symbol_trades["time_m"].gt(trigger_time)].copy()
    if after_trigger.empty:
        return None

    entry_candidates = after_trigger[after_trigger["price"].le(trigger_price * (1 - pullback))]
    if entry_candidates.empty:
        return None
    entry = entry_candidates.iloc[0]
    entry_time = entry["time_m"]
    entry_price = float(entry["price"])
    path = after_trigger[after_trigger["time_m"].ge(entry_time)].copy()
    if path.empty or entry_price <= 0:
        return None

    returns = path["price"] / entry_price - 1
    max_idx = returns.idxmax()
    min_idx = returns.idxmin()
    row: dict[str, object] = {
        "date": event["date"],
        "trade_date": event["trade_date"],
        "symbol": event["symbol"],
        "pullback": pullback,
        "trigger_time": trigger_time,
        "trigger_price": trigger_price,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "entry_discount_from_trigger": entry_price / trigger_price - 1,
        "mfe_to_0930": float(returns.loc[max_idx]),
        "mfe_time": path.loc[max_idx, "time_m"],
        "mae_to_0930": float(returns.loc[min_idx]),
        "mae_time": path.loc[min_idx, "time_m"],
        "last_return": float(path["price"].iloc[-1]) / entry_price - 1,
        "mfe_before_mae": path.loc[max_idx, "time_m"] <= path.loc[min_idx, "time_m"],
    }
    for tp in TAKE_PROFITS:
        tp_time = first_hit(path, entry_price, tp, "up")
        row[f"hit_tp_{int(tp * 100)}"] = tp_time is not None
        for sl in STOP_LOSSES:
            sl_time = first_hit(path, entry_price, sl, "down")
            sl_abs = int(abs(sl) * 100)
            if tp_time is None and sl_time is None:
                outcome = "neither"
            elif tp_time is not None and (sl_time is None or tp_time <= sl_time):
                outcome = "tp_first"
            else:
                outcome = "sl_first"
            row[f"tp{int(tp * 100)}_vs_sl{sl_abs}_outcome"] = outcome
    return row


def summarize(results: pd.DataFrame, trigger_counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(["period", "pullback"], sort=True):
        period, pullback = keys
        denominator = trigger_counts[period]
        row: dict[str, object] = {
            "period": period,
            "pullback": pullback,
            "trigger_count": denominator,
            "entry_count": len(group),
            "entry_coverage": len(group) / denominator,
            "median_entry_time": group["entry_time"].median(),
            "median_entry_discount_from_trigger": group["entry_discount_from_trigger"].median(),
            "median_mfe_to_0930": group["mfe_to_0930"].median(),
            "median_mae_to_0930": group["mae_to_0930"].median(),
            "p_hit_tp3": group["hit_tp_3"].mean(),
            "p_hit_tp5": group["hit_tp_5"].mean(),
            "median_last_return": group["last_return"].median(),
            "mfe_before_mae_rate": group["mfe_before_mae"].mean(),
        }
        for tp in TAKE_PROFITS:
            for sl in STOP_LOSSES:
                sl_abs = int(abs(sl) * 100)
                col = f"tp{int(tp * 100)}_vs_sl{sl_abs}_outcome"
                counts = group[col].value_counts(normalize=True)
                row[f"p_tp{int(tp * 100)}_before_sl{sl_abs}"] = counts.get("tp_first", 0.0)
                row[f"p_sl{sl_abs}_before_tp{int(tp * 100)}"] = counts.get("sl_first", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test pullback entries after the premarket dynamic trigger.")
    parser.add_argument("--trigger-dir", type=Path, default=DEFAULT_TRIGGER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trigger-floor", type=float, default=0.05)
    parser.add_argument("--min-dollar-volume", type=float, default=1_000_000)
    parser.add_argument("--min-trade-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    triggers = pd.read_csv(
        args.trigger_dir / "full_premarket_dynamic_trigger_results.csv",
        parse_dates=["date", "trade_date"],
    )
    triggers = triggers[
        triggers["trigger_floor"].eq(args.trigger_floor)
        & triggers["min_cum_dollar_volume"].eq(args.min_dollar_volume)
        & triggers["min_cum_trade_count"].eq(args.min_trade_count)
    ].copy()

    cache_dir = args.trigger_dir / "raw_trade_cache"
    rows: list[dict[str, object]] = []
    grouped = list(triggers.groupby("trade_date"))
    for i, (trade_date, day_events) in enumerate(grouped, start=1):
        if i == 1 or i % 250 == 0:
            print(f"processing {i}/{len(grouped)} {pd.Timestamp(trade_date).date()}", flush=True)
        cache_path = cache_dir / f"taq_common_0400_0930_{pd.Timestamp(trade_date).strftime('%Y%m%d')}.csv.gz"
        if not cache_path.exists():
            continue
        trades = prepare_trades(cache_path)
        trades_by_symbol = {symbol: group.reset_index(drop=True) for symbol, group in trades.groupby("sym_root", sort=False)}
        for _, event in day_events.iterrows():
            symbol_trades = trades_by_symbol.get(str(event["symbol"]))
            if symbol_trades is None:
                continue
            for pullback in PULLBACKS:
                simulated = simulate_pullback(event, symbol_trades, pullback)
                if simulated is not None:
                    rows.append(simulated)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No pullback entries found.")
    trigger_years = pd.to_datetime(triggers["trade_date"]).dt.year
    trigger_counts = {
        "all": len(triggers),
        "train_2015_2021": int((trigger_years <= 2021).sum()),
        "validation_2022_2024": int((trigger_years >= 2022).sum()),
    }
    years = pd.to_datetime(results["trade_date"]).dt.year
    results["period"] = np.where(years <= 2021, "train_2015_2021", "validation_2022_2024")
    all_results = results.copy()
    all_results["period"] = "all"
    combined = pd.concat([results, all_results], ignore_index=True)
    combined.to_csv(args.output_dir / "pullback_entry_path_events.csv", index=False)

    summary = summarize(combined, trigger_counts)
    summary.to_csv(args.output_dir / "pullback_entry_path_summary.csv", index=False)
    print(f"triggers={len(triggers)} pullback_entries={len(results)}")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
