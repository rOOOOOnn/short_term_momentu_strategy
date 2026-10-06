from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = PROJECT_ROOT / "ml_strategy_lab"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LAB_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB_ROOT / "scripts"))

from common import load_baseline_triggers  # noqa: E402
from importlib import import_module  # noqa: E402


DEFAULT_GRID = LAB_ROOT / "data" / "strategy_grid_intraday_results.csv"
DEFAULT_DATASET = LAB_ROOT / "data" / "model_dataset_intraday.csv"
DEFAULT_CACHE_DIR = LAB_ROOT / "data" / "taq_0400_1600_cache"
DEFAULT_OUTPUT = LAB_ROOT / "reports" / "intraday" / "pipeline_validation.json"
STRATEGY_CONFIG = LAB_ROOT / "configs" / "strategy_grid_intraday.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit intraday grid formulas and dataset integrity.")
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def timedeltas(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce")


def add_check(checks: list[dict[str, object]], name: str, failures: int, total: int, detail: str) -> None:
    checks.append(
        {
            "name": name,
            "failures": int(failures),
            "total": int(total),
            "passed": int(failures) == 0,
            "detail": detail,
        }
    )


def main() -> None:
    args = parse_args()
    config = json.loads(STRATEGY_CONFIG.read_text(encoding="utf-8"))
    grid = pd.read_csv(args.grid, parse_dates=["date", "trade_date"])
    dataset = pd.read_csv(args.dataset, parse_dates=["date", "trade_date"])
    triggers = load_baseline_triggers()
    checks: list[dict[str, object]] = []

    expected_dates = sorted(pd.Timestamp(value) for value in triggers["trade_date"].unique())
    missing_cache = [
        value
        for value in expected_dates
        if not (args.cache_dir / f"taq_path_0400_1600_{value.strftime('%Y%m%d')}.csv.gz").is_file()
    ]
    add_check(
        checks,
        "cache_date_coverage",
        len(missing_cache),
        len(expected_dates),
        f"expected_dates={len(expected_dates)} missing_dates={len(missing_cache)}",
    )

    expected_rows = len(triggers) * 9
    add_check(
        checks,
        "grid_row_count",
        abs(len(grid) - expected_rows),
        expected_rows,
        f"expected_rows={expected_rows} actual_rows={len(grid)}",
    )
    duplicate_count = int(grid.duplicated(["event_id", "strategy_id"]).sum())
    add_check(
        checks,
        "event_strategy_uniqueness",
        duplicate_count,
        len(grid),
        f"duplicate_event_strategy_rows={duplicate_count}",
    )
    strategy_counts = grid.groupby("event_id")["strategy_id"].nunique()
    bad_strategy_counts = int(strategy_counts.ne(9).sum())
    add_check(
        checks,
        "nine_strategies_per_event",
        bad_strategy_counts,
        len(strategy_counts),
        f"events_with_non_nine_strategy_count={bad_strategy_counts}",
    )

    trigger_td = timedeltas(grid["trigger_time"])
    first_entry_td = timedeltas(grid["first_entry_time"])
    last_entry_td = timedeltas(grid["last_entry_time"])
    exit_td = timedeltas(grid["exit_time"])
    entered = ~grid["no_entry"].astype(bool) & ~grid["strategy_id"].eq("SKIP")
    entry_deadline = pd.to_timedelta(config["entry_deadline"])
    forced_exit = pd.to_timedelta(config["forced_exit_time"])

    bad_entry_time = entered & (
        first_entry_td.isna()
        | first_entry_td.le(trigger_td)
        | first_entry_td.gt(entry_deadline)
        | last_entry_td.lt(first_entry_td)
        | last_entry_td.gt(entry_deadline)
    )
    add_check(
        checks,
        "entry_time_bounds",
        int(bad_entry_time.sum()),
        int(entered.sum()),
        "Entries must be after the trigger, ordered, and no later than the configured deadline.",
    )
    bad_exit_time = entered & (exit_td.isna() | exit_td.lt(first_entry_td) | exit_td.gt(forced_exit))
    add_check(
        checks,
        "exit_time_bounds",
        int(bad_exit_time.sum()),
        int(entered.sum()),
        "Entered strategies must exit no earlier than entry and no later than the forced exit.",
    )

    weight_map = {
        "equal": np.array(config["entry_weight_schemes"]["equal"], dtype=float),
        "back_weighted": np.array(config["entry_weight_schemes"]["back_weighted"], dtype=float),
    }
    levels = np.array(config["staged_entry_levels"], dtype=float)
    entry_slippage = float(config["entry_slippage_bps"]) / 10_000
    expected_invested = []
    expected_avg_entry = []
    for row in grid.itertuples(index=False):
        if row.strategy_id == "SKIP" or int(row.filled_entry_count) == 0:
            expected_invested.append(0.0)
            expected_avg_entry.append(np.nan)
            continue
        weights = weight_map[row.weight_scheme][: int(row.filled_entry_count)]
        fill_prices = float(row.trigger_price) * (1 - levels[: len(weights)]) * (1 + entry_slippage)
        expected_invested.append(float(weights.sum()))
        expected_avg_entry.append(float(weights.sum() / np.sum(weights / fill_prices)))
    expected_invested_series = pd.Series(expected_invested, index=grid.index)
    expected_avg_entry_series = pd.Series(expected_avg_entry, index=grid.index)
    bad_invested = ~np.isclose(
        grid["invested_fraction"].astype(float),
        expected_invested_series,
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    )
    add_check(
        checks,
        "invested_fraction_formula",
        int(bad_invested.sum()),
        len(grid),
        "Invested capital must equal the cumulative configured stage weights.",
    )
    avg_entry_actual = pd.to_numeric(grid["avg_entry_price"], errors="coerce")
    bad_avg_entry = entered & ~np.isclose(
        avg_entry_actual,
        expected_avg_entry_series,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
    add_check(
        checks,
        "average_entry_price_formula",
        int(bad_avg_entry.sum()),
        int(entered.sum()),
        "Average entry price is independently recomputed from staged weights and slipped fill prices.",
    )

    fixed_entered = entered & grid["exit_type"].eq("fixed")
    expected_fixed_return = grid["invested_fraction"] * (
        grid["exit_price"] / grid["avg_entry_price"] - 1
    )
    bad_fixed_return = fixed_entered & ~np.isclose(
        grid["capital_return"],
        expected_fixed_return,
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True,
    )
    add_check(
        checks,
        "fixed_exit_capital_return_formula",
        int(bad_fixed_return.sum()),
        int(fixed_entered.sum()),
        "Fixed-exit capital return equals invested fraction times the slipped exit-to-entry return.",
    )

    exit_slippage = float(config["exit_slippage_bps"]) / 10_000
    raw_exit = grid["exit_price"] / (1 - exit_slippage)
    stop_rows = entered & grid["exit_reason"].eq("stop_loss")
    bad_stops = stop_rows & raw_exit.gt(grid["avg_entry_price"] * (1 - grid["stop_loss"]) + 1e-10)
    add_check(
        checks,
        "stop_trigger_threshold",
        int(bad_stops.sum()),
        int(stop_rows.sum()),
        "The observed trade price before exit slippage must be at or below the stop threshold.",
    )
    take_profit_rows = entered & grid["exit_reason"].eq("take_profit")
    bad_take_profits = take_profit_rows & raw_exit.lt(
        grid["avg_entry_price"] * (1 + grid["take_profit"]) - 1e-10
    )
    add_check(
        checks,
        "take_profit_trigger_threshold",
        int(bad_take_profits.sum()),
        int(take_profit_rows.sum()),
        "The observed trade price before exit slippage must be at or above the target threshold.",
    )

    no_entry = grid["no_entry"].astype(bool) & ~grid["strategy_id"].eq("SKIP")
    bad_no_entry = no_entry & (
        grid["capital_return"].ne(0)
        | grid["position_return"].ne(0)
        | grid["invested_fraction"].ne(0)
        | grid["first_entry_time"].notna()
        | grid["exit_time"].notna()
    )
    add_check(
        checks,
        "no_entry_zero_state",
        int(bad_no_entry.sum()),
        int(no_entry.sum()),
        "Unfilled strategies must have zero capital exposure and no entry or exit timestamps.",
    )

    dataset_duplicates = int(dataset.duplicated(["event_id", "strategy_id"]).sum())
    add_check(
        checks,
        "model_dataset_uniqueness",
        dataset_duplicates,
        len(dataset),
        f"duplicate_event_strategy_rows={dataset_duplicates}",
    )
    missing_event_features = int(dataset["asof_trade_count"].isna().sum())
    add_check(
        checks,
        "model_feature_join_coverage",
        missing_event_features,
        len(dataset),
        f"rows_missing_asof_trade_count={missing_event_features}",
    )

    train_module = import_module("05_train_lightgbm")
    selected_features = train_module.numeric_feature_columns(dataset)
    leakage_overlap = sorted(set(selected_features) & train_module.LEAKAGE_COLUMNS)
    add_check(
        checks,
        "declared_leakage_exclusion",
        len(leakage_overlap),
        len(selected_features),
        f"leakage_columns_selected={leakage_overlap}",
    )

    simulator_module = import_module("02_build_strategy_grid")
    synthetic_position = simulator_module.Position()
    simulator_module.add_position(synthetic_position, price=10.0, weight=1.0)
    simulator_module.close_position(synthetic_position, price=12.0, fraction=0.5)
    simulator_module.close_position(synthetic_position, price=11.0, fraction=1.0)
    expected_partial_pnl = 0.15
    partial_math_failed = not (
        np.isclose(synthetic_position.realized_pnl, expected_partial_pnl, atol=1e-12)
        and np.isclose(synthetic_position.units, 0.0, atol=1e-12)
        and np.isclose(synthetic_position.cost, 0.0, atol=1e-12)
    )
    add_check(
        checks,
        "partial_exit_accounting_unit_case",
        int(partial_math_failed),
        1,
        "One unit of capital at $10, half sold at $12 and the remainder at $11 must realize $0.15.",
    )

    result = {
        "summary": {
            "cache_dates": len(expected_dates) - len(missing_cache),
            "expected_cache_dates": len(expected_dates),
            "events": int(grid["event_id"].nunique()),
            "strategies": int(grid["strategy_id"].nunique()),
            "grid_rows": len(grid),
            "entered_strategy_rows": int(entered.sum()),
            "validation_events": int(dataset.loc[dataset["year"].ge(2022), "event_id"].nunique()),
            "selected_numeric_features": len(selected_features),
        },
        "checks": checks,
        "limitations": {
            "partial_exit_reconstruction": int(
                (entered & grid["exit_type"].eq("half_tp_trailing") & grid["hit_take_profit"].astype(bool)).sum()
            ),
            "description": "Rows with a partial take-profit cannot be independently reconstructed from the grid alone because the partial fill price and time are not persisted.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if any(not check["passed"] for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
