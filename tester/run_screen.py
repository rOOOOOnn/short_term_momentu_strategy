# run_screen.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE_UNIVERSE,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOP_N,
    GAINERS_FILTER,
)
from filters import (
    add_all_buckets,
    apply_default_strategy_filter,
    base_universe_filter,
    yesterday_gainer_filter,
)


def rank_candidates(df: pd.DataFrame, rank_by: str = "score") -> pd.DataFrame:
    """
    Score candidates using only information known at date T.
    Do not use any next_* columns here.
    """

    df = df.copy()

    if df.empty:
        return df

    score = pd.Series(0.0, index=df.index)

    score += df["return_1d"].rank(pct=True) * 0.25
    score += df["dollar_volume"].rank(pct=True) * 0.25
    score += df["relative_volume_20d"].rank(pct=True) * 0.25
    score += df["clv"].rank(pct=True) * 0.25

    df["score"] = score

    if rank_by == "score":
        sort_cols = ["score", "dollar_volume", "return_1d"]
    elif rank_by == "return_1d":
        sort_cols = ["return_1d", "dollar_volume", "relative_volume_20d"]
    elif rank_by == "dollar_volume":
        sort_cols = ["dollar_volume", "return_1d", "relative_volume_20d"]
    elif rank_by == "relative_volume_20d":
        sort_cols = ["relative_volume_20d", "return_1d", "dollar_volume"]
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")

    df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    df["screen_rank"] = range(1, len(df) + 1)

    return df


def screen_one_day(
    df: pd.DataFrame,
    date: pd.Timestamp,
    top_n: int = DEFAULT_TOP_N,
    mode: str = DEFAULT_MODE,
    rank_by: str = "score",
) -> pd.DataFrame:
    """Screen one T date and return the top candidates."""

    daily = df[df["date"] == date].copy()

    if daily.empty:
        return daily

    if mode == "default":
        selected = apply_default_strategy_filter(daily)
    elif mode == "gainers":
        selected = base_universe_filter(daily, **BASE_UNIVERSE)
        selected = yesterday_gainer_filter(selected, **GAINERS_FILTER)
    else:
        raise ValueError(f"Unknown screening mode: {mode}")

    if selected.empty:
        return selected

    selected = rank_candidates(selected, rank_by=rank_by)
    return selected.head(top_n)


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns used only for output analysis."""

    df = df.copy()

    df["year"] = pd.to_datetime(df["date"]).dt.year

    if "score" in df.columns and df["score"].notna().any():
        score_rank = df["score"].rank(method="first", pct=True)
        score_decile = (score_rank * 10).apply(lambda x: min(10, max(1, int(x + 0.999999))))
        df["score_bucket"] = score_decile.map(lambda x: f"D{x:02d}")

    return df


def get_output_columns(df: pd.DataFrame) -> list[str]:
    """Columns to keep in CSV outputs."""

    cols = [
        "date",
        "next_date",
        "year",
        "permno",
        "symbol",
        "company_name",
        "exchange",
        "sic_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "dollar_volume",
        "avg_dollar_volume_20d",
        "relative_volume_20d",
        "market_cap",
        "clv",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "score",
        "screen_rank",
        "score_bucket",
        "market_cap_bucket",
        "dollar_volume_bucket",
        "return_bucket",
        "relative_volume_bucket",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "next_volume",
        "next_dollar_volume",
        "next_relative_volume_20d",
        "next_open_return",
        "next_high_return",
        "next_low_return",
        "next_close_return",
        "next_intraday_return",
        "next_volume_change",
        "next_dollar_volume_change",
    ]

    return [col for col in cols if col in df.columns]


def summarize_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize next-day price and volume behavior."""

    if df.empty:
        return pd.DataFrame()

    metrics = {
        "trade_count": len(df),
        "avg_score": df["score"].mean(),
        "median_score": df["score"].median(),
        "avg_next_open_return": df["next_open_return"].mean(),
        "median_next_open_return": df["next_open_return"].median(),
        "win_rate_next_open": (df["next_open_return"] > 0).mean(),
        "avg_next_close_return": df["next_close_return"].mean(),
        "median_next_close_return": df["next_close_return"].median(),
        "win_rate_next_close": (df["next_close_return"] > 0).mean(),
        "avg_next_high_return": df["next_high_return"].mean(),
        "median_next_high_return": df["next_high_return"].median(),
        "avg_next_low_return": df["next_low_return"].mean(),
        "median_next_low_return": df["next_low_return"].median(),
        "avg_next_volume": df["next_volume"].mean(),
        "median_next_volume": df["next_volume"].median(),
        "avg_next_dollar_volume": df["next_dollar_volume"].mean(),
        "median_next_dollar_volume": df["next_dollar_volume"].median(),
        "avg_next_volume_change": df["next_volume_change"].mean(),
        "median_next_volume_change": df["next_volume_change"].median(),
        "avg_next_dollar_volume_change": df["next_dollar_volume_change"].mean(),
        "median_next_dollar_volume_change": df["next_dollar_volume_change"].median(),
    }

    return pd.DataFrame([metrics])


def group_summary(
    df: pd.DataFrame,
    group_cols: str | list[str],
) -> pd.DataFrame:
    """Group-level summary for next-day behavior."""

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    if df.empty or any(col not in df.columns for col in group_cols):
        return pd.DataFrame()

    summary = (
        df.groupby(group_cols, observed=True)
        .agg(
            trade_count=("symbol", "count"),
            avg_score=("score", "mean"),
            median_score=("score", "median"),
            avg_next_open_return=("next_open_return", "mean"),
            median_next_open_return=("next_open_return", "median"),
            win_rate_next_open=("next_open_return", lambda x: (x > 0).mean()),
            avg_next_close_return=("next_close_return", "mean"),
            median_next_close_return=("next_close_return", "median"),
            win_rate_next_close=("next_close_return", lambda x: (x > 0).mean()),
            avg_next_high_return=("next_high_return", "mean"),
            avg_next_low_return=("next_low_return", "mean"),
            avg_next_volume=("next_volume", "mean"),
            median_next_volume=("next_volume", "median"),
            avg_next_dollar_volume=("next_dollar_volume", "mean"),
            median_next_dollar_volume=("next_dollar_volume", "median"),
            avg_next_volume_change=("next_volume_change", "mean"),
            median_next_volume_change=("next_volume_change", "median"),
            avg_next_dollar_volume_change=("next_dollar_volume_change", "mean"),
            median_next_dollar_volume_change=("next_dollar_volume_change", "median"),
        )
        .reset_index()
    )

    return summary


def score_relationship_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between the composite score and next-day outcomes."""

    if df.empty or "score" not in df.columns:
        return pd.DataFrame()

    outcome_cols = [
        "next_open_return",
        "next_close_return",
        "next_high_return",
        "next_low_return",
        "next_intraday_return",
        "next_volume_change",
        "next_dollar_volume_change",
    ]

    rows = []
    for col in outcome_cols:
        if col not in df.columns:
            continue

        valid = df[["score", col]].dropna()
        if len(valid) < 2:
            continue

        rows.append(
            {
                "outcome": col,
                "count": len(valid),
                "pearson_corr": valid["score"].corr(valid[col], method="pearson"),
                "rank_corr": valid["score"].rank().corr(valid[col].rank()),
            }
        )

    return pd.DataFrame(rows)


def factor_relationship_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between each score component and next-day outcomes."""

    if df.empty:
        return pd.DataFrame()

    factor_cols = [
        "return_1d",
        "dollar_volume",
        "relative_volume_20d",
        "clv",
        "score",
    ]
    outcome_cols = [
        "next_open_return",
        "next_close_return",
        "next_high_return",
        "next_low_return",
        "next_intraday_return",
        "next_volume_change",
        "next_dollar_volume_change",
    ]

    rows = []
    for factor_col in factor_cols:
        if factor_col not in df.columns:
            continue

        for outcome_col in outcome_cols:
            if outcome_col not in df.columns:
                continue

            valid = df[[factor_col, outcome_col]].dropna()
            if len(valid) < 2:
                continue

            rows.append(
                {
                    "factor": factor_col,
                    "outcome": outcome_col,
                    "count": len(valid),
                    "pearson_corr": valid[factor_col].corr(valid[outcome_col], method="pearson"),
                    "rank_corr": valid[factor_col].rank().corr(valid[outcome_col].rank()),
                }
            )

    return pd.DataFrame(rows)


def make_run_output_dir(
    base_output_dir: str | Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    mode: str,
    top_n: int,
    target_date: str | pd.Timestamp | None = None,
) -> Path:
    """Create one clean output folder per run."""

    base_output_dir = Path(base_output_dir)

    if target_date is not None:
        date_label = f"target_{pd.Timestamp(target_date).strftime('%Y%m%d')}"
    else:
        date_label = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    output_dir = base_output_dir / f"{date_label}_{mode}_top{top_n}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_analysis_outputs(
    result: pd.DataFrame,
    output_dir: Path,
    save_yearly_files: bool = True,
) -> None:
    """Save all aggregate analysis files."""

    all_result_path = output_dir / "all_screen_results.csv"
    result.to_csv(all_result_path, index=False)
    print(f"Saved all results to: {all_result_path}")

    overall_summary = summarize_returns(result)
    overall_summary_path = output_dir / "overall_summary.csv"
    overall_summary.to_csv(overall_summary_path, index=False)
    print(f"Saved overall summary to: {overall_summary_path}")

    score_corr = score_relationship_summary(result)
    if not score_corr.empty:
        score_corr_path = output_dir / "score_relationship_summary.csv"
        score_corr.to_csv(score_corr_path, index=False)
        print(f"Saved score relationship summary to: {score_corr_path}")

    factor_corr = factor_relationship_summary(result)
    if not factor_corr.empty:
        factor_corr_path = output_dir / "factor_relationship_summary.csv"
        factor_corr.to_csv(factor_corr_path, index=False)
        print(f"Saved factor relationship summary to: {factor_corr_path}")

    group_outputs = {
        "year": ["year"],
        "score_bucket": ["score_bucket"],
        "year_score_bucket": ["year", "score_bucket"],
        "market_cap_bucket": ["market_cap_bucket"],
        "dollar_volume_bucket": ["dollar_volume_bucket"],
        "return_bucket": ["return_bucket"],
        "relative_volume_bucket": ["relative_volume_bucket"],
        "exchange": ["exchange"],
        "year_market_cap_bucket": ["year", "market_cap_bucket"],
        "year_dollar_volume_bucket": ["year", "dollar_volume_bucket"],
        "year_return_bucket": ["year", "return_bucket"],
        "year_relative_volume_bucket": ["year", "relative_volume_bucket"],
    }

    for name, cols in group_outputs.items():
        summary = group_summary(result, cols)
        if summary.empty:
            continue

        summary_path = output_dir / f"summary_by_{name}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved group summary to: {summary_path}")

    if save_yearly_files and "year" in result.columns:
        yearly_dir = output_dir / "yearly"
        yearly_dir.mkdir(parents=True, exist_ok=True)

        for year, yearly_result in result.groupby("year"):
            yearly_path = yearly_dir / f"screen_results_{int(year)}.csv"
            yearly_result.to_csv(yearly_path, index=False)

        print(f"Saved yearly result files to: {yearly_dir}")


def run_screen_df(
    df: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = DEFAULT_TOP_N,
    mode: str = DEFAULT_MODE,
    rank_by: str = "score",
    save_daily_files: bool = True,
    target_date: str | pd.Timestamp | None = None,
    screen_start: str | pd.Timestamp | None = None,
    screen_end: str | pd.Timestamp | None = None,
    create_run_subdir: bool = True,
) -> pd.DataFrame:
    """Run screening on an in-memory feature DataFrame."""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_all_buckets(df)

    dates = sorted(df["date"].dropna().unique())
    if target_date is not None:
        target_date_ts = pd.Timestamp(target_date)
        dates = [date for date in dates if pd.Timestamp(date) == target_date_ts]
    else:
        if screen_start is not None:
            screen_start_ts = pd.Timestamp(screen_start)
            dates = [date for date in dates if pd.Timestamp(date) >= screen_start_ts]
        if screen_end is not None:
            screen_end_ts = pd.Timestamp(screen_end)
            dates = [date for date in dates if pd.Timestamp(date) <= screen_end_ts]

    if not dates:
        print("No dates to screen.")
        return pd.DataFrame()

    start_date = pd.Timestamp(min(dates))
    end_date = pd.Timestamp(max(dates))

    if create_run_subdir:
        output_dir = make_run_output_dir(
            output_dir,
            start_date,
            end_date,
            f"{mode}_{rank_by}",
            top_n,
            target_date,
        )
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    daily_output_dir = output_dir / "daily"
    if save_daily_files:
        daily_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {output_dir}")
    print(f"Total dates: {len(dates):,}")
    print(f"Screening mode: {mode}")
    print(f"Rank by: {rank_by}")

    all_results = []

    for date in dates:
        selected = screen_one_day(
            df=df,
            date=date,
            top_n=top_n,
            mode=mode,
            rank_by=rank_by,
        )

        if selected.empty:
            continue

        selected = add_all_buckets(selected)
        selected = add_analysis_columns(selected)
        selected = selected[get_output_columns(selected)]

        all_results.append(selected)

        if save_daily_files:
            file_name = f"{pd.Timestamp(date).strftime('%Y-%m-%d')}.csv"
            selected.to_csv(daily_output_dir / file_name, index=False)

    if not all_results:
        print("No selected stocks.")
        return pd.DataFrame()

    result = pd.concat(all_results, ignore_index=True)
    result = add_analysis_columns(result)
    result = result[get_output_columns(result)]

    print(f"Selected rows: {len(result):,}")
    save_analysis_outputs(result, output_dir)

    return result


def run_screen(
    input_path: str | Path,
    output_dir: str | Path,
    top_n: int = DEFAULT_TOP_N,
    mode: str = DEFAULT_MODE,
    rank_by: str = "score",
    save_daily_files: bool = True,
    target_date: str | pd.Timestamp | None = None,
    screen_start: str | pd.Timestamp | None = None,
    screen_end: str | pd.Timestamp | None = None,
    create_run_subdir: bool = True,
) -> pd.DataFrame:
    """Read a local parquet feature file and run screening."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}. "
            "Pass --input to an existing parquet, or use --wrds-start and --wrds-end."
        )

    df = pd.read_parquet(input_path)

    return run_screen_df(
        df=df,
        output_dir=output_dir,
        top_n=top_n,
        mode=mode,
        rank_by=rank_by,
        save_daily_files=save_daily_files,
        target_date=target_date,
        screen_start=screen_start,
        screen_end=screen_end,
        create_run_subdir=create_run_subdir,
    )


def run_screen_from_wrds(
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    top_n: int = DEFAULT_TOP_N,
    mode: str = DEFAULT_MODE,
    rank_by: str = "score",
    save_daily_files: bool = True,
    raw_output_path: str | Path | None = None,
    target_date: str | pd.Timestamp | None = None,
    screen_start: str | pd.Timestamp | None = None,
    screen_end: str | pd.Timestamp | None = None,
    create_run_subdir: bool = True,
) -> pd.DataFrame:
    """Download a WRDS date window, build features in memory, and run screening."""

    from data.wrds.preprocess import load_wrds_crsp_features

    df = load_wrds_crsp_features(
        start_date=start_date,
        end_date=end_date,
        raw_output_path=raw_output_path,
    )

    return run_screen_df(
        df=df,
        output_dir=output_dir,
        top_n=top_n,
        mode=mode,
        rank_by=rank_by,
        save_daily_files=save_daily_files,
        target_date=target_date,
        screen_start=screen_start,
        screen_end=screen_end,
        create_run_subdir=create_run_subdir,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to processed parquet. Ignored when --wrds-start and --wrds-end are set.",
    )
    parser.add_argument(
        "--wrds-start",
        type=str,
        default=None,
        help="Start date for direct WRDS screening, e.g. 2000-01-01.",
    )
    parser.add_argument(
        "--wrds-end",
        type=str,
        default=None,
        help="End date for direct WRDS screening, e.g. 2025-12-31.",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="Optional path to save raw WRDS data. Processed parquet is not saved.",
    )
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Only screen this T date. The input window still needs enough history and T+1 data.",
    )
    parser.add_argument(
        "--screen-start",
        type=str,
        default=None,
        help="First T date to screen. Useful when WRDS start includes warmup history.",
    )
    parser.add_argument(
        "--screen-end",
        type=str,
        default=None,
        help="Last T date to screen. WRDS end should still include the next trading day.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory. A date-range subfolder is created under it by default.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="score",
        choices=["score", "return_1d", "dollar_volume", "relative_volume_20d"],
        help="Ranking rule after filters. Use return_1d for top gainers.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["gainers", "default"],
        help="gainers: simple yesterday-gainer universe; default: stricter baseline filter.",
    )
    parser.add_argument(
        "--no-daily-files",
        action="store_true",
        help="Do not save daily CSV files.",
    )
    parser.add_argument(
        "--no-run-subdir",
        action="store_true",
        help="Write directly to --output-dir instead of creating a date-range subfolder.",
    )

    args = parser.parse_args()

    if args.wrds_start or args.wrds_end:
        if not args.wrds_start or not args.wrds_end:
            parser.error("--wrds-start and --wrds-end must be used together.")

        run_screen_from_wrds(
            start_date=args.wrds_start,
            end_date=args.wrds_end,
            output_dir=args.output_dir,
            top_n=args.top_n,
            mode=args.mode,
            rank_by=args.rank_by,
            save_daily_files=not args.no_daily_files,
            raw_output_path=args.raw_output,
            target_date=args.target_date,
            screen_start=args.screen_start,
            screen_end=args.screen_end,
            create_run_subdir=not args.no_run_subdir,
        )
    else:
        run_screen(
            input_path=args.input,
            output_dir=args.output_dir,
            top_n=args.top_n,
            mode=args.mode,
            rank_by=args.rank_by,
            save_daily_files=not args.no_daily_files,
            target_date=args.target_date,
            screen_start=args.screen_start,
            screen_end=args.screen_end,
            create_run_subdir=not args.no_run_subdir,
        )


if __name__ == "__main__":
    main()
