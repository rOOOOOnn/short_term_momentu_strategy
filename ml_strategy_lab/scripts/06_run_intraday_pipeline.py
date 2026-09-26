from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import DEFAULT_TRIGGER_DIR, LAB_ROOT, load_baseline_triggers


DEFAULT_CACHE_DIR = LAB_ROOT / "data" / "taq_0400_1600_cache"
DEFAULT_CACHE_PATTERN = "taq_path_0400_1600_{date}.csv.gz"
DEFAULT_GRID = LAB_ROOT / "data" / "strategy_grid_intraday_results.csv"
DEFAULT_FEATURES = LAB_ROOT / "data" / "event_features.csv"
DEFAULT_DATASET = LAB_ROOT / "data" / "model_dataset_intraday.csv"
DEFAULT_STRATEGY_CONFIG = LAB_ROOT / "configs" / "strategy_grid_intraday.json"
DEFAULT_MODEL_CONFIG = LAB_ROOT / "configs" / "lightgbm_intraday.json"
DEFAULT_REPORT = LAB_ROOT / "reports" / "intraday" / "lightgbm_intraday_report.md"
DEFAULT_MODEL_DIR = LAB_ROOT / "models" / "intraday"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate complete intraday cache coverage, then build and train the intraday model."
    )
    parser.add_argument("--trigger-dir", type=Path, default=DEFAULT_TRIGGER_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-pattern", default=DEFAULT_CACHE_PATTERN)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--strategy-config", type=Path, default=DEFAULT_STRATEGY_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Run with available cache dates. Intended only for explicit smoke tests.",
    )
    return parser.parse_args()


def cache_coverage(
    trigger_dir: Path,
    cache_dir: Path,
    cache_pattern: str,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    triggers = load_baseline_triggers(trigger_dir)
    expected = sorted(pd.Timestamp(value) for value in triggers["trade_date"].unique())
    missing = [
        trade_date
        for trade_date in expected
        if not (
            cache_dir
            / cache_pattern.format(date=trade_date.strftime("%Y%m%d"))
        ).is_file()
    ]
    return expected, missing


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    expected, missing = cache_coverage(args.trigger_dir, args.cache_dir, args.cache_pattern)
    available_count = len(expected) - len(missing)
    print(
        f"cache_dates={available_count:,}/{len(expected):,} missing={len(missing):,}",
        flush=True,
    )
    if missing and not args.allow_partial:
        preview = ", ".join(value.strftime("%Y-%m-%d") for value in missing[:20])
        raise RuntimeError(
            f"Intraday cache is incomplete; refusing to train. First missing dates: {preview}"
        )

    grid_command = [
        sys.executable,
        str(SCRIPT_DIR / "02_build_strategy_grid.py"),
        "--trigger-dir",
        str(args.trigger_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--cache-pattern",
        args.cache_pattern,
        "--config",
        str(args.strategy_config),
        "--output",
        str(args.grid),
    ]
    if missing and args.allow_partial:
        print("WARNING: partial mode uses available cache files only.", flush=True)
    run(grid_command)

    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "04_build_model_dataset.py"),
            "--grid",
            str(args.grid),
            "--features",
            str(args.features),
            "--config",
            str(args.model_config),
            "--output",
            str(args.dataset),
        ]
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "05_train_lightgbm.py"),
            "--dataset",
            str(args.dataset),
            "--config",
            str(args.model_config),
            "--report",
            str(args.report),
            "--model-dir",
            str(args.model_dir),
        ]
    )


if __name__ == "__main__":
    main()
