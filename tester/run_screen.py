# run_screen.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    给候选股票打分。

    注意：
    这里只能使用 T 日已经知道的变量。
    不能使用 next_open_return / next_close_return 等未来结果变量。
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

    df = df.sort_values(
        ["score", "dollar_volume", "return_1d"],
        ascending=[False, False, False],
    )

    return df


def screen_one_day(
    df: pd.DataFrame,
    date: pd.Timestamp,
    top_n: int = DEFAULT_TOP_N,
    mode: str = DEFAULT_MODE,
) -> pd.DataFrame:
    """
    对某一天进行筛选。
    date 是 T 日，也就是当天收盘后能看到的数据。
    """

    daily = df[df["date"] == date].copy()

    if daily.empty:
        return daily

    if mode == "default":
        selected = apply_default_strategy_filter(daily)

    elif mode == "gainers":
        selected = base_universe_filter(
            daily,
            **BASE_UNIVERSE,
        )
        selected = yesterday_gainer_filter(
            selected,
            **GAINERS_FILTER,
        )

    else:
        raise ValueError(f"Unknown screening mode: {mode}")

    if selected.empty:
        return selected

    selected = rank_candidates(selected)
    selected = selected.head(top_n)

    return selected


def get_output_columns(df: pd.DataFrame) -> list[str]:
    """
    输出时保留的列。
    """

    cols = [
        # identity
        "date",
        "permno",
        "symbol",
        "company_name",
        "exchange",
        "sic_code",
        # price / volume
        "open",
        "high",
        "low",
        "close",
        "volume",
        # features
        "return_1d",
        "dollar_volume",
        "avg_dollar_volume_20d",
        "relative_volume_20d",
        "market_cap",
        "clv",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "score",
        # buckets
        "market_cap_bucket",
        "dollar_volume_bucket",
        "return_bucket",
        "relative_volume_bucket",
        # next day evaluation
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "next_open_return",
        "next_high_return",
        "next_low_return",
        "next_close_return",
        "next_intraday_return",
    ]

    return [col for col in cols if col in df.columns]


def summarize_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总筛选结果的第二天表现。
    """

    if df.empty:
        return pd.DataFrame()

    metrics = {
        "trade_count": len(df),
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
    }

    return pd.DataFrame([metrics])


def group_summary(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    按某个 bucket 分组统计 T+1 表现。
    """

    if df.empty or group_col not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby(group_col, observed=True)
        .agg(
            trade_count=("symbol", "count"),
            avg_next_open_return=("next_open_return", "mean"),
            median_next_open_return=("next_open_return", "median"),
            win_rate_next_open=("next_open_return", lambda x: (x > 0).mean()),
            avg_next_close_return=("next_close_return", "mean"),
            median_next_close_return=("next_close_return", "median"),
            win_rate_next_close=("next_close_return", lambda x: (x > 0).mean()),
            avg_next_high_return=("next_high_return", "mean"),
            avg_next_low_return=("next_low_return", "mean"),
        )
        .reset_index()
    )

    return summary


def run_screen(
    input_path: str | Path,
    output_dir: str | Path,
    top_n: int = DEFAULT_TOP_N,
    mode: str = DEFAULT_MODE,
    save_daily_files: bool = True,
) -> pd.DataFrame:
    """
    主筛选程序。
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_output_dir = output_dir / "daily"
    if save_daily_files:
        daily_output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])

    # 添加分组标签。
    df = add_all_buckets(df)

    all_results = []

    dates = sorted(df["date"].dropna().unique())

    print(f"Total dates: {len(dates):,}")
    print(f"Screening mode: {mode}")

    for date in dates:
        selected = screen_one_day(
            df=df,
            date=date,
            top_n=top_n,
            mode=mode,
        )

        if selected.empty:
            continue

        selected = add_all_buckets(selected)
        selected = selected[get_output_columns(selected)]

        all_results.append(selected)

        if save_daily_files:
            file_name = f"{pd.Timestamp(date).strftime('%Y-%m-%d')}.csv"
            selected.to_csv(daily_output_dir / file_name, index=False)

    if not all_results:
        print("No selected stocks.")
        return pd.DataFrame()

    result = pd.concat(all_results, ignore_index=True)

    all_result_path = output_dir / "all_screen_results.csv"
    result.to_csv(all_result_path, index=False)

    print(f"Selected rows: {len(result):,}")
    print(f"Saved all results to: {all_result_path}")

    # 总体统计。
    overall_summary = summarize_returns(result)
    overall_summary_path = output_dir / "overall_summary.csv"
    overall_summary.to_csv(overall_summary_path, index=False)

    print(f"Saved overall summary to: {overall_summary_path}")

    # 分组统计。
    group_cols = [
        "market_cap_bucket",
        "dollar_volume_bucket",
        "return_bucket",
        "relative_volume_bucket",
        "exchange",
    ]

    for group_col in group_cols:
        if group_col not in result.columns:
            continue

        summary = group_summary(result, group_col=group_col)

        if summary.empty:
            continue

        summary_path = output_dir / f"summary_by_{group_col}.csv"
        summary.to_csv(summary_path, index=False)

        print(f"Saved group summary to: {summary_path}")

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["gainers", "default"],
        help="""
        gainers: 只筛昨日上涨股票，适合做分层研究
        default: 使用第一版完整策略条件
        """,
    )

    parser.add_argument(
        "--no-daily-files",
        action="store_true",
        help="Do not save daily CSV files.",
    )

    args = parser.parse_args()

    run_screen(
        input_path=args.input,
        output_dir=args.output_dir,
        top_n=args.top_n,
        mode=args.mode,
        save_daily_files=not args.no_daily_files,
    )


if __name__ == "__main__":
    main()
