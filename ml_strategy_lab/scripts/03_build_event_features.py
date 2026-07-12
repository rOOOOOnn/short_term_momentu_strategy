from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DEFAULT_TRIGGER_DIR,
    LAB_ROOT,
    ensure_dir,
    load_baseline_triggers,
    parse_trigger_time,
    read_cached_trades,
)


DEFAULT_CACHE_DIR = DEFAULT_TRIGGER_DIR / "raw_trade_cache"
DEFAULT_OUTPUT = LAB_ROOT / "data" / "event_features.csv"


ROLLING_WINDOWS_MIN = [5, 10, 30, 60]


def cache_path(cache_dir: Path, trade_date: pd.Timestamp) -> Path:
    return cache_dir / f"taq_common_0400_0930_{trade_date.strftime('%Y%m%d')}.csv.gz"


def safe_return(last: float, first: float) -> float:
    if pd.isna(last) or pd.isna(first) or first <= 0:
        return np.nan
    return last / first - 1


def max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return np.nan
    running_high = prices.cummax()
    dd = prices / running_high - 1
    return float(dd.min())


def max_runup(prices: pd.Series) -> float:
    if prices.empty:
        return np.nan
    running_low = prices.cummin()
    ru = prices / running_low - 1
    return float(ru.max())


def summarize_window(prefix: str, trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            f"{prefix}_trade_count": 0,
            f"{prefix}_volume": 0.0,
            f"{prefix}_dollar_volume": 0.0,
            f"{prefix}_return": np.nan,
            f"{prefix}_high_return": np.nan,
            f"{prefix}_low_return": np.nan,
        }
    first = float(trades["price"].iloc[0])
    last = float(trades["price"].iloc[-1])
    high = float(trades["price"].max())
    low = float(trades["price"].min())
    return {
        f"{prefix}_trade_count": len(trades),
        f"{prefix}_volume": float(trades["size"].sum()) if "size" in trades.columns else np.nan,
        f"{prefix}_dollar_volume": float(trades["dollar_volume"].sum()) if "dollar_volume" in trades.columns else np.nan,
        f"{prefix}_return": safe_return(last, first),
        f"{prefix}_high_return": safe_return(high, first),
        f"{prefix}_low_return": safe_return(low, first),
    }


def build_features_for_event(event: pd.Series, symbol_trades: pd.DataFrame) -> dict[str, object]:
    trigger_time = parse_trigger_time(event["trigger_time"])
    asof = symbol_trades[symbol_trades["time_m"].le(trigger_time)].copy()
    row: dict[str, object] = {
        "event_id": event["event_id"],
        "date": event["date"],
        "trade_date": event["trade_date"],
        "symbol": event["symbol"],
        "permno": event["permno"],
        "year": pd.Timestamp(event["trade_date"]).year,
        "market_cap": event["market_cap"],
        "log_market_cap": np.log(float(event["market_cap"])) if float(event["market_cap"]) > 0 else np.nan,
        "return_1d": event["return_1d"],
        "t_day_dollar_volume": event["t_day_dollar_volume"],
        "log_t_day_dollar_volume": np.log(float(event["t_day_dollar_volume"])) if float(event["t_day_dollar_volume"]) > 0 else np.nan,
        "trigger_time": trigger_time,
        "trigger_minutes_from_0400": trigger_time.total_seconds() / 60 - 240,
        "trigger_price": event["trigger_price"],
        "trigger_return": event["trigger_return"],
        "trigger_cum_dollar_volume": event["cum_dollar_volume"],
        "trigger_cum_trade_count": event["cum_trade_count"],
    }

    if asof.empty:
        row.update(
            {
                "asof_trade_count": 0,
                "asof_volume": 0.0,
                "asof_dollar_volume": 0.0,
                "asof_vwap": np.nan,
                "trigger_price_vs_vwap": np.nan,
                "asof_return": np.nan,
                "asof_high_from_t_close": np.nan,
                "asof_low_from_t_close": np.nan,
                "asof_max_drawdown": np.nan,
                "asof_max_runup": np.nan,
                "asof_price_slope_per_min": np.nan,
            }
        )
        return row

    prices = asof["price"].astype(float)
    dollar_volume = asof["dollar_volume"].sum() if "dollar_volume" in asof.columns else np.nan
    volume = asof["size"].sum() if "size" in asof.columns else np.nan
    vwap = dollar_volume / volume if volume and volume > 0 else np.nan
    first_price = float(prices.iloc[0])
    last_price = float(prices.iloc[-1])
    t_close = float(event["trigger_price"]) / (1 + float(event["trigger_return"]))
    elapsed_min = max((asof["time_m"].iloc[-1] - asof["time_m"].iloc[0]).total_seconds() / 60, 1e-9)

    row.update(
        {
            "asof_trade_count": len(asof),
            "asof_volume": float(volume) if not pd.isna(volume) else np.nan,
            "asof_dollar_volume": float(dollar_volume) if not pd.isna(dollar_volume) else np.nan,
            "asof_vwap": vwap,
            "trigger_price_vs_vwap": float(event["trigger_price"]) / vwap - 1 if vwap and vwap > 0 else np.nan,
            "asof_return": safe_return(last_price, first_price),
            "asof_high_from_t_close": float(prices.max()) / t_close - 1 if t_close > 0 else np.nan,
            "asof_low_from_t_close": float(prices.min()) / t_close - 1 if t_close > 0 else np.nan,
            "asof_max_drawdown": max_drawdown(prices),
            "asof_max_runup": max_runup(prices),
            "asof_price_slope_per_min": (last_price / first_price - 1) / elapsed_min,
        }
    )

    for minutes in ROLLING_WINDOWS_MIN:
        start = trigger_time - pd.Timedelta(minutes=minutes)
        window = asof[asof["time_m"].ge(start)]
        row.update(summarize_window(f"last_{minutes}m", window))

    completed_windows = [
        ("w_0400_0700", "04:00:00", "07:00:00"),
        ("w_0700_0800", "07:00:00", "08:00:00"),
        ("w_0800_0900", "08:00:00", "09:00:00"),
        ("w_0900_0930", "09:00:00", "09:30:00"),
    ]
    for name, start, end in completed_windows:
        start_td = pd.to_timedelta(start)
        end_td = min(pd.to_timedelta(end), trigger_time)
        available = trigger_time >= start_td
        row[f"{name}_available"] = available
        if not available:
            row.update(summarize_window(name, asof.iloc[0:0]))
            continue
        row.update(summarize_window(name, asof[(asof["time_m"].ge(start_td)) & (asof["time_m"].le(end_td))]))

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trigger-time as-of event features.")
    parser.add_argument("--trigger-dir", type=Path, default=DEFAULT_TRIGGER_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-dates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    triggers = load_baseline_triggers(args.trigger_dir)
    if args.start_date:
        triggers = triggers[triggers["trade_date"].ge(pd.Timestamp(args.start_date))]
    if args.end_date:
        triggers = triggers[triggers["trade_date"].le(pd.Timestamp(args.end_date))]
    grouped = list(triggers.groupby("trade_date", sort=True))
    if args.max_dates:
        grouped = grouped[: args.max_dates]

    rows: list[dict[str, object]] = []
    for index, (trade_date, day_events) in enumerate(grouped, start=1):
        path = cache_path(args.cache_dir, pd.Timestamp(trade_date))
        if not path.exists():
            print(f"[{index}/{len(grouped)}] missing cache {path.name}", flush=True)
            continue
        trades = read_cached_trades(path)
        trades_by_symbol = {
            symbol: group.reset_index(drop=True)
            for symbol, group in trades.groupby("sym_root", sort=False)
        }
        for _, event in day_events.iterrows():
            symbol_trades = trades_by_symbol.get(str(event["symbol"]).upper())
            if symbol_trades is None:
                continue
            rows.append(build_features_for_event(event, symbol_trades))
        if index == 1 or index % 100 == 0 or index == len(grouped):
            print(f"[{index}/{len(grouped)}] {pd.Timestamp(trade_date).date()} rows={len(rows):,}", flush=True)

    output = Path(args.output)
    ensure_dir(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output.resolve())


if __name__ == "__main__":
    main()
