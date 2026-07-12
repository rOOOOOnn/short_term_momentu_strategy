from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import LAB_ROOT, ensure_dir, load_json


DEFAULT_GRID = LAB_ROOT / "data" / "strategy_grid_results.csv"
DEFAULT_FEATURES = LAB_ROOT / "data" / "event_features.csv"
DEFAULT_CONFIG = LAB_ROOT / "configs" / "lightgbm_baseline.json"
DEFAULT_OUTPUT = LAB_ROOT / "data" / "model_dataset.csv"


def add_strategy_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_skip_strategy"] = df["strategy_id"].eq("SKIP").astype(int)
    df["is_trailing_exit"] = df["exit_type"].eq("half_tp_trailing").astype(int)
    df["is_fixed_exit"] = df["exit_type"].eq("fixed").astype(int)
    df["is_back_weighted"] = df["weight_scheme"].eq("back_weighted").astype(int)
    df["is_equal_weighted"] = df["weight_scheme"].eq("equal").astype(int)
    df["planned_entry_count"] = pd.to_numeric(df["planned_entry_count"], errors="coerce").fillna(0)
    for col in ["stop_loss", "take_profit", "trail"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join event features with strategy grid labels.")
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    grid = pd.read_csv(args.grid, parse_dates=["date", "trade_date"])
    features = pd.read_csv(args.features, parse_dates=["date", "trade_date"])

    duplicated_feature_cols = [
        col
        for col in [
            "date",
            "trade_date",
            "symbol",
            "permno",
            "market_cap",
            "return_1d",
            "t_day_dollar_volume",
            "trigger_time",
            "trigger_price",
            "trigger_return",
            "cum_dollar_volume",
            "cum_trade_count",
        ]
        if col in features.columns
    ]
    features = features.drop(columns=duplicated_feature_cols)
    dataset = grid.merge(features, on="event_id", how="inner", validate="many_to_one")
    dataset = add_strategy_numeric_features(dataset)

    large_loss_threshold = float(config.get("large_loss_threshold", -0.02))
    dataset["large_loss"] = dataset["capital_return"].le(large_loss_threshold).astype(int)
    dataset["trade_entered"] = (~dataset["no_entry"].astype(bool) & ~dataset["strategy_id"].eq("SKIP")).astype(int)
    dataset["year"] = pd.to_datetime(dataset["trade_date"]).dt.year
    dataset["validation_period"] = np.where(dataset["year"].ge(int(config["validation_start_year"])), "validation", "train")

    output = Path(args.output)
    ensure_dir(output.parent)
    dataset.to_csv(output, index=False)
    print(f"rows={len(dataset):,} events={dataset['event_id'].nunique():,} strategies={dataset['strategy_id'].nunique():,}")
    print(output.resolve())


if __name__ == "__main__":
    main()
