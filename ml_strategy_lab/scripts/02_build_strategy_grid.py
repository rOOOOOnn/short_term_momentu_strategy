from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DEFAULT_TRIGGER_DIR,
    LAB_ROOT,
    ensure_dir,
    load_baseline_triggers,
    load_json,
    parse_time,
    parse_trigger_time,
    read_cached_trades,
)


DEFAULT_CONFIG = LAB_ROOT / "configs" / "strategy_grid.json"
DEFAULT_CACHE_DIR = DEFAULT_TRIGGER_DIR / "raw_trade_cache"
DEFAULT_OUTPUT = LAB_ROOT / "data" / "strategy_grid_results.csv"


@dataclass
class Position:
    units: float = 0.0
    cost: float = 0.0
    realized_pnl: float = 0.0
    invested_fraction: float = 0.0
    entry_units_total: float = 0.0
    entry_cost_total: float = 0.0
    exited: bool = False
    partial_taken: bool = False
    high_water: float = 0.0

    @property
    def avg_cost(self) -> float:
        return self.cost / self.units if self.units > 0 else np.nan


def cache_path(cache_dir: Path, trade_date: pd.Timestamp) -> Path:
    return cache_dir / f"taq_common_0400_0930_{trade_date.strftime('%Y%m%d')}.csv.gz"


def make_strategy_templates(config: dict) -> list[dict[str, object]]:
    strategies: list[dict[str, object]] = [
        {
            "strategy_id": "SKIP",
            "entry_type": "skip",
            "weight_scheme": "none",
            "exit_name": "none",
            "exit_type": "none",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "trail": 0.0,
        }
    ]
    for weight_name, weights in config["entry_weight_schemes"].items():
        for exit_template in config["exit_templates"]:
            strategy = {
                "strategy_id": f"PB235_{weight_name}_{exit_template['name']}",
                "entry_type": "staged_pb235",
                "weight_scheme": weight_name,
                "weights": weights,
                "exit_name": exit_template["name"],
                "exit_type": exit_template["type"],
                "stop_loss": float(exit_template["stop_loss"]),
                "take_profit": float(exit_template["take_profit"]),
                "trail": float(exit_template.get("trail", 0.0)),
                "take_profit_fraction": float(exit_template.get("take_profit_fraction", 0.0)),
            }
            strategies.append(strategy)
    return strategies


def base_event_fields(event: pd.Series) -> dict[str, object]:
    return {
        "event_id": event["event_id"],
        "date": event["date"],
        "trade_date": event["trade_date"],
        "symbol": event["symbol"],
        "permno": event["permno"],
        "market_cap": event["market_cap"],
        "return_1d": event["return_1d"],
        "t_day_dollar_volume": event["t_day_dollar_volume"],
        "trigger_time": parse_trigger_time(event["trigger_time"]),
        "trigger_price": event["trigger_price"],
        "trigger_return": event["trigger_return"],
        "cum_dollar_volume": event["cum_dollar_volume"],
        "cum_trade_count": event["cum_trade_count"],
    }


def skip_result(event: pd.Series) -> dict[str, object]:
    row = base_event_fields(event)
    row.update(
        {
            "strategy_id": "SKIP",
            "entry_type": "skip",
            "weight_scheme": "none",
            "exit_name": "none",
            "exit_type": "none",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "trail": 0.0,
            "planned_entry_count": 0,
            "filled_entry_count": 0,
            "first_entry_time": pd.NaT,
            "last_entry_time": pd.NaT,
            "avg_entry_price": np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "exit_reason": "skip",
            "invested_fraction": 0.0,
            "position_return": 0.0,
            "capital_return": 0.0,
            "max_runup": 0.0,
            "max_drawdown": 0.0,
            "hit_stop": False,
            "hit_take_profit": False,
            "no_entry": False,
        }
    )
    return row


def close_position(position: Position, price: float, fraction: float) -> None:
    fraction = min(max(fraction, 0.0), 1.0)
    sell_units = position.units * fraction
    if sell_units <= 0:
        return
    avg_cost = position.avg_cost
    position.realized_pnl += sell_units * (price - avg_cost)
    position.units -= sell_units
    position.cost -= sell_units * avg_cost
    if position.units <= 1e-12:
        position.units = 0.0
        position.cost = 0.0
        position.exited = True


def add_position(position: Position, price: float, weight: float) -> None:
    if weight <= 0:
        return
    units = weight / price
    position.units += units
    position.cost += weight
    position.invested_fraction += weight
    position.entry_units_total += units
    position.entry_cost_total += weight
    position.high_water = max(position.high_water, price)


def current_capital_return(position: Position, price: float) -> float:
    unrealized = position.units * (price - position.avg_cost) if position.units > 0 else 0.0
    return position.realized_pnl + unrealized


def current_position_return(position: Position, price: float) -> float:
    if position.invested_fraction <= 0:
        return 0.0
    return current_capital_return(position, price) / position.invested_fraction


def simulate_strategy(
    event: pd.Series,
    symbol_trades: pd.DataFrame,
    strategy: dict[str, object],
    config: dict,
) -> dict[str, object]:
    if strategy["strategy_id"] == "SKIP":
        return skip_result(event)

    row = base_event_fields(event)
    trigger_time = row["trigger_time"]
    trigger_price = float(row["trigger_price"])
    entry_deadline = parse_time(config["entry_deadline"])
    forced_exit_time = parse_time(config["forced_exit_time"])
    entry_slippage = float(config.get("entry_slippage_bps", 0.0)) / 10_000
    exit_slippage = float(config.get("exit_slippage_bps", 0.0)) / 10_000
    levels = [float(x) for x in config["staged_entry_levels"]]
    weights = [float(x) for x in strategy["weights"]]

    path = symbol_trades[
        (symbol_trades["time_m"].gt(trigger_time))
        & (symbol_trades["time_m"].le(forced_exit_time))
    ].copy()
    position = Position()
    filled_levels: set[int] = set()
    first_entry_time = pd.NaT
    last_entry_time = pd.NaT
    exit_time = pd.NaT
    exit_price = np.nan
    exit_reason = "forced_exit"
    max_runup = 0.0
    max_drawdown = 0.0
    hit_stop = False
    hit_take_profit = False

    if path.empty:
        exit_reason = "no_path"
    else:
        for _, trade in path.iterrows():
            time_m = trade["time_m"]
            raw_price = float(trade["price"])

            if time_m <= entry_deadline and not position.exited:
                for idx, (level, weight) in enumerate(zip(levels, weights)):
                    if idx in filled_levels:
                        continue
                    limit_price = trigger_price * (1 - level)
                    if raw_price <= limit_price:
                        fill_price = limit_price * (1 + entry_slippage)
                        add_position(position, fill_price, weight)
                        filled_levels.add(idx)
                        if pd.isna(first_entry_time):
                            first_entry_time = time_m
                        last_entry_time = time_m

            if position.units <= 0 or position.exited:
                continue

            mark_return = current_position_return(position, raw_price)
            max_runup = max(max_runup, mark_return)
            max_drawdown = min(max_drawdown, mark_return)
            avg_cost = position.avg_cost
            position.high_water = max(position.high_water, raw_price)

            stop_price = avg_cost * (1 - float(strategy["stop_loss"]))
            take_profit_price = avg_cost * (1 + float(strategy["take_profit"]))

            if raw_price <= stop_price:
                fill_price = raw_price * (1 - exit_slippage)
                close_position(position, fill_price, 1.0)
                exit_time = time_m
                exit_price = fill_price
                exit_reason = "stop_loss"
                hit_stop = True
                break

            if strategy["exit_type"] == "fixed":
                if raw_price >= take_profit_price:
                    fill_price = raw_price * (1 - exit_slippage)
                    close_position(position, fill_price, 1.0)
                    exit_time = time_m
                    exit_price = fill_price
                    exit_reason = "take_profit"
                    hit_take_profit = True
                    break
            elif strategy["exit_type"] == "half_tp_trailing":
                if not position.partial_taken and raw_price >= take_profit_price:
                    fill_price = raw_price * (1 - exit_slippage)
                    close_position(position, fill_price, float(strategy["take_profit_fraction"]))
                    position.partial_taken = True
                    position.high_water = raw_price
                    hit_take_profit = True
                if position.partial_taken and position.units > 0:
                    position.high_water = max(position.high_water, raw_price)
                    trail_price = position.high_water * (1 - float(strategy["trail"]))
                    if raw_price <= trail_price:
                        fill_price = raw_price * (1 - exit_slippage)
                        close_position(position, fill_price, 1.0)
                        exit_time = time_m
                        exit_price = fill_price
                        exit_reason = "trailing_stop"
                        break

        if position.units > 0 and not position.exited:
            eligible_exit = path[path["time_m"].le(forced_exit_time)]
            if not eligible_exit.empty:
                final = eligible_exit.iloc[-1]
                fill_price = float(final["price"]) * (1 - exit_slippage)
                close_position(position, fill_price, 1.0)
                exit_time = final["time_m"]
                exit_price = fill_price

    no_entry = position.invested_fraction == 0
    capital_return = position.realized_pnl
    position_return = (
        position.realized_pnl / position.invested_fraction
        if position.invested_fraction > 0
        else 0.0
    )
    avg_entry_price = (
        position.entry_cost_total / position.entry_units_total
        if position.entry_units_total > 0
        else np.nan
    )

    row.update(
        {
            "strategy_id": strategy["strategy_id"],
            "entry_type": strategy["entry_type"],
            "weight_scheme": strategy["weight_scheme"],
            "exit_name": strategy["exit_name"],
            "exit_type": strategy["exit_type"],
            "stop_loss": strategy["stop_loss"],
            "take_profit": strategy["take_profit"],
            "trail": strategy["trail"],
            "planned_entry_count": len(levels),
            "filled_entry_count": len(filled_levels),
            "first_entry_time": first_entry_time,
            "last_entry_time": last_entry_time,
            "avg_entry_price": avg_entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": "no_entry" if no_entry else exit_reason,
            "invested_fraction": position.invested_fraction,
            "position_return": position_return,
            "capital_return": capital_return,
            "max_runup": max_runup,
            "max_drawdown": max_drawdown,
            "hit_stop": hit_stop,
            "hit_take_profit": hit_take_profit,
            "no_entry": no_entry,
        }
    )
    return row


def simulate_event(
    event: pd.Series,
    trades_by_symbol: dict[str, pd.DataFrame],
    strategies: list[dict[str, object]],
    config: dict,
) -> list[dict[str, object]]:
    symbol_trades = trades_by_symbol.get(str(event["symbol"]).upper())
    if symbol_trades is None:
        return [skip_result(event)]
    return [simulate_strategy(event, symbol_trades, strategy, config) for strategy in strategies]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build realized returns for each event and strategy template.")
    parser.add_argument("--trigger-dir", type=Path, default=DEFAULT_TRIGGER_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-dates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    strategies = make_strategy_templates(config)
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
        trade_date = pd.Timestamp(trade_date)
        path = cache_path(args.cache_dir, trade_date)
        if not path.exists():
            print(f"[{index}/{len(grouped)}] missing cache {path.name}", flush=True)
            continue
        trades = read_cached_trades(path)
        trades_by_symbol = {
            symbol: group.reset_index(drop=True)
            for symbol, group in trades.groupby("sym_root", sort=False)
        }
        for _, event in day_events.iterrows():
            rows.extend(simulate_event(event, trades_by_symbol, strategies, config))
        if index == 1 or index % 100 == 0 or index == len(grouped):
            print(f"[{index}/{len(grouped)}] {trade_date.date()} rows={len(rows):,}", flush=True)

    output = Path(args.output)
    ensure_dir(output.parent)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    print(output.resolve())


if __name__ == "__main__":
    main()
