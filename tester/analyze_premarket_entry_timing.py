from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = [
    "pre_0400_0700",
    "pre_0700_0800",
    "pre_0800_0900",
    "pre_0900_0915",
    "pre_0915_0930",
    "open_0930_0935",
    "open_0935_1000",
]

ENTRY_SETUPS = [
    ("07:00", ["pre_0400_0700"], "pre_0700_0800"),
    ("08:00", ["pre_0400_0700", "pre_0700_0800"], "pre_0800_0900"),
    ("09:00", ["pre_0400_0700", "pre_0700_0800", "pre_0800_0900"], "pre_0900_0915"),
    (
        "09:15",
        ["pre_0400_0700", "pre_0700_0800", "pre_0800_0900", "pre_0900_0915"],
        "pre_0915_0930",
    ),
]

RETURN_BANDS = [
    ("0%-2%", 0.00, 0.02),
    ("2%-5%", 0.02, 0.05),
    ("5%-10%", 0.05, 0.10),
]


def numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "next_date"], low_memory=False)
    numeric_columns = ["close", "market_cap", "next_open"]
    for window in WINDOWS:
        numeric_columns.extend(
            [
                f"{window}_open",
                f"{window}_high",
                f"{window}_low",
                f"{window}_close",
                f"{window}_dollar_volume",
                f"{window}_trade_count",
            ]
        )
    numeric(df, numeric_columns)
    return df[
        df["market_cap"].between(300_000_000, 5_000_000_000, inclusive="both")
        & df["close"].gt(0)
    ].copy()


def row_max(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    return df[available].max(axis=1, skipna=True)


def row_min(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    return df[available].min(axis=1, skipna=True)


def build_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for entry_time, confirm_windows, entry_window in ENTRY_SETUPS:
        confirm_close = f"{confirm_windows[-1]}_close"
        entry_open = f"{entry_window}_open"
        required = [confirm_close, entry_open]
        if not all(column in df.columns for column in required):
            continue

        event = df.copy()
        event["entry_time"] = entry_time
        event["entry_window"] = entry_window
        event["confirm_return"] = event[confirm_close] / event["close"] - 1
        event["cum_dollar_volume"] = event[
            [f"{window}_dollar_volume" for window in confirm_windows]
        ].fillna(0).sum(axis=1)
        event["cum_trade_count"] = event[
            [f"{window}_trade_count" for window in confirm_windows]
        ].fillna(0).sum(axis=1)
        event["entry_price"] = event[entry_open]
        event["entry_premium"] = event["entry_price"] / event["close"] - 1

        entry_index = WINDOWS.index(entry_window)
        through_0930 = WINDOWS[entry_index : WINDOWS.index("open_0930_0935")]
        through_0935 = WINDOWS[entry_index : WINDOWS.index("open_0935_1000")]
        through_1000 = WINDOWS[entry_index:]

        event["high_to_0930"] = row_max(
            event, [f"{window}_high" for window in through_0930]
        )
        event["low_to_0930"] = row_min(
            event, [f"{window}_low" for window in through_0930]
        )
        event["high_to_0935"] = row_max(
            event, [f"{window}_high" for window in through_0935]
        )
        event["low_to_0935"] = row_min(
            event, [f"{window}_low" for window in through_0935]
        )
        event["high_to_1000"] = row_max(
            event, [f"{window}_high" for window in through_1000]
        )
        event["low_to_1000"] = row_min(
            event, [f"{window}_low" for window in through_1000]
        )

        event["return_0930"] = event["next_open"] / event["entry_price"] - 1
        event["return_0935"] = (
            event["open_0930_0935_close"] / event["entry_price"] - 1
        )
        event["return_1000"] = (
            event["open_0935_1000_close"] / event["entry_price"] - 1
        )

        for horizon in ["0930", "0935", "1000"]:
            event[f"mfe_{horizon}"] = (
                event[f"high_to_{horizon}"] / event["entry_price"] - 1
            )
            event[f"mae_{horizon}"] = (
                event[f"low_to_{horizon}"] / event["entry_price"] - 1
            )

        event["future_low_discount_0930"] = (
            event["low_to_0930"] / event["entry_price"] - 1
        )
        event = event[
            event["entry_price"].gt(0)
            & event["confirm_return"].notna()
            & event["cum_dollar_volume"].gt(0)
            & (event["entry_price"] / event["close"]).between(0.5, 2.0)
            & (event["next_open"] / event["close"]).between(0.5, 2.0)
        ]
        rows.append(event)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def trimmed_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    lower = clean.quantile(0.05)
    upper = clean.quantile(0.95)
    return clean.clip(lower, upper).mean()


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band_name, lower, upper in RETURN_BANDS:
        band = events["confirm_return"].between(lower, upper, inclusive="left")
        for min_dv in [250_000, 500_000, 1_000_000]:
            selected = events[
                band
                & events["cum_dollar_volume"].ge(min_dv)
                & events["cum_trade_count"].ge(25)
            ]
            for entry_time, group in selected.groupby("entry_time", sort=False):
                if group.empty:
                    continue
                rows.append(
                    {
                        "entry_time": entry_time,
                        "return_band": band_name,
                        "min_cum_dollar_volume": min_dv,
                        "min_cum_trades": 25,
                        "count": len(group),
                        "avg_entry_premium": group["entry_premium"].mean(),
                        "median_entry_premium": group["entry_premium"].median(),
                        "avg_return_0930": group["return_0930"].mean(),
                        "trimmed_avg_return_0930": trimmed_mean(group["return_0930"]),
                        "median_return_0930": group["return_0930"].median(),
                        "win_rate_0930": group["return_0930"].gt(0).mean(),
                        "avg_return_0935": group["return_0935"].mean(),
                        "median_return_0935": group["return_0935"].median(),
                        "win_rate_0935": group["return_0935"].gt(0).mean(),
                        "avg_return_1000": group["return_1000"].mean(),
                        "median_return_1000": group["return_1000"].median(),
                        "win_rate_1000": group["return_1000"].gt(0).mean(),
                        "avg_mfe_0930": group["mfe_0930"].mean(),
                        "median_mfe_0930": group["mfe_0930"].median(),
                        "avg_mae_0930": group["mae_0930"].mean(),
                        "median_mae_0930": group["mae_0930"].median(),
                        "mae_25pct_0930": group["mae_0930"].quantile(0.25),
                        "avg_mfe_1000": group["mfe_1000"].mean(),
                        "avg_mae_1000": group["mae_1000"].mean(),
                        "p_future_drop_1pct": group["future_low_discount_0930"]
                        .le(-0.01)
                        .mean(),
                        "p_future_drop_2pct": group["future_low_discount_0930"]
                        .le(-0.02)
                        .mean(),
                        "p_mfe_3pct_0930": group["mfe_0930"].ge(0.03).mean(),
                        "p_mfe_5pct_1000": group["mfe_1000"].ge(0.05).mean(),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["return_band", "min_cum_dollar_volume", "entry_time"]
    )


def summarize_yearly(events: pd.DataFrame) -> pd.DataFrame:
    selected = events[
        events["confirm_return"].between(0.02, 0.05, inclusive="left")
        & events["cum_dollar_volume"].ge(500_000)
        & events["cum_trade_count"].ge(25)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["year"] = selected["next_date"].dt.year
    return (
        selected.groupby(["year", "entry_time"], observed=True)
        .agg(
            count=("return_0930", "count"),
            avg_entry_premium=("entry_premium", "mean"),
            avg_return_0930=("return_0930", "mean"),
            avg_return_0935=("return_0935", "mean"),
            avg_return_1000=("return_1000", "mean"),
            avg_mfe_0930=("mfe_0930", "mean"),
            avg_mae_0930=("mae_0930", "mean"),
        )
        .reset_index()
    )


def summarize_pullback_opportunity(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band_name, lower, upper in RETURN_BANDS:
        for min_dv in [250_000, 500_000, 1_000_000]:
            selected = events[
                events["confirm_return"].between(lower, upper, inclusive="left")
                & events["cum_dollar_volume"].ge(min_dv)
                & events["cum_trade_count"].ge(25)
            ]
            for entry_time, group in selected.groupby("entry_time", sort=False):
                if group.empty:
                    continue
                discount = group["future_low_discount_0930"]
                rows.append(
                    {
                        "entry_time": entry_time,
                        "return_band": band_name,
                        "min_cum_dollar_volume": min_dv,
                        "count": len(group),
                        "median_future_low_discount_0930": discount.median(),
                        "p_pullback_1pct": discount.le(-0.01).mean(),
                        "p_pullback_2pct": discount.le(-0.02).mean(),
                        "p_pullback_3pct": discount.le(-0.03).mean(),
                        "p_pullback_4pct": discount.le(-0.04).mean(),
                        "p_pullback_5pct": discount.le(-0.05).mean(),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    pullbacks: pd.DataFrame,
    output_path: Path,
) -> None:
    eligible = summary[summary["count"].ge(200)].copy()
    ranked = eligible.sort_values(
        ["trimmed_avg_return_0930", "median_mae_0930", "count"],
        ascending=[False, False, False],
    ).head(12)

    lines = [
        "# 盘前入场时机分析",
        "",
        "## 口径",
        "",
        "- 股票池：T 日涨幅前 50，市值 300M-5B。",
        "- 信号只使用入场时点之前的数据：相对昨收涨幅、累计盘前成交额、累计成交笔数。",
        "- 入场价使用下一时间窗口的第一笔成交价代理，不使用未来最低价。",
        "- 本阶段不设置止盈止损，只观察到 09:30、09:35、10:00 的价格路径。",
        "- MFE/MAE 基于分时窗口高低价，不代表实际可按极值成交。",
        "",
        "## 样本数不少于 200 的较优组合",
        "",
    ]

    if ranked.empty:
        lines.append("没有满足样本数要求的组合。")
    else:
        lines.append(
            "|入场|确认涨幅|最低累计成交额|样本|入场溢价|截尾到09:30|中位数到09:30|中位数MFE|中位数MAE|之后回落>=1%|"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in ranked.iterrows():
            lines.append(
                f"|{row['entry_time']}|{row['return_band']}|"
                f"{row['min_cum_dollar_volume'] / 1_000_000:.2f}M|"
                f"{int(row['count'])}|{row['avg_entry_premium']:.2%}|"
                f"{row['trimmed_avg_return_0930']:.2%}|"
                f"{row['median_return_0930']:.2%}|"
                f"{row['median_mfe_0930']:.2%}|{row['median_mae_0930']:.2%}|"
                f"{row['p_future_drop_1pct']:.1%}|"
            )

    lines.extend(
        [
            "",
            "## 解读原则",
            "",
            "- 入场越早通常价格越低，但 04:00-07:00 的流动性和异常成交风险更高。",
            "- 如果入场后经常再次跌破 1%-2%，直接追突破不是最低成本的入场方式，应研究回踩限价或 VWAP 回踩。",
            "- 07:00 的历史均值容易被 2020 年极端行情和早期稀疏样本扭曲，应优先看截尾均值和逐年中位数。",
            "- 平均收益不能单独决定规则；需要同时看中位数、胜率、MFE、MAE 和逐年稳定性。",
            "- 该结果用于选择下一步要开发的入场逻辑，不是完整策略收益回测。",
            "",
            "## 逐年稳定性",
            "",
            "逐年表使用统一条件：确认涨幅 2%-5%、累计成交额至少 500K、累计成交笔数至少 25。",
        ]
    )

    if not yearly.empty:
        stable = (
            yearly.groupby("entry_time")
            .agg(
                years=("year", "nunique"),
                positive_years=("avg_return_0930", lambda x: x.gt(0).sum()),
                median_year_return_0930=("avg_return_0930", "median"),
            )
            .reset_index()
        )
        lines.extend(["", "|入场|覆盖年份|正收益年份|年度中位数到09:30收益|", "|---|---:|---:|---:|"])
        for _, row in stable.iterrows():
            lines.append(
                f"|{row['entry_time']}|{int(row['years'])}|"
                f"{int(row['positive_years'])}|{row['median_year_return_0930']:.2%}|"
            )

    pullback_focus = pullbacks[
        (pullbacks["entry_time"] == "08:00")
        & (pullbacks["return_band"] == "5%-10%")
        & (pullbacks["min_cum_dollar_volume"] == 250_000)
    ]
    lines.extend(
        [
            "",
            "## 当前可行方向",
            "",
            "1. 不采用 07:00 直接市价追入：早期样本稀疏，稳健收益为负，回撤也明显更大。",
            "2. 主候选信号：08:00 时相对昨收上涨 5%-10%，04:00-08:00 累计成交额至少 250K，累计成交至少 25 笔。",
            "3. 信号确认后不立即追价，下一步测试相对 08:00 参考价回踩 2%-3% 的限价入场。",
            "4. 09:00 和 09:15 才入场虽然回撤较小，但剩余上涨空间明显下降。",
            "5. 回踩概率只说明限价可能成交，尚不能证明成交后有利润；需要逐笔数据验证价格先后顺序。",
        ]
    )
    if not pullback_focus.empty:
        row = pullback_focus.iloc[0]
        lines.extend(
            [
                "",
                "### 08:00 主候选的回踩机会",
                "",
                f"- 样本数：{int(row['count'])}。",
                f"- 后续回踩至少 1%：{row['p_pullback_1pct']:.1%}。",
                f"- 后续回踩至少 2%：{row['p_pullback_2pct']:.1%}。",
                f"- 后续回踩至少 3%：{row['p_pullback_3pct']:.1%}。",
                f"- 后续回踩至少 4%：{row['p_pullback_4pct']:.1%}。",
                f"- 后续回踩至少 5%：{row['p_pullback_5pct']:.1%}。",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to premarket_enriched_results.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / "entry_timing_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    events = build_event_rows(df)
    if events.empty:
        raise ValueError("No entry events could be built from the input data.")

    summary = summarize(events)
    yearly = summarize_yearly(events)
    pullbacks = summarize_pullback_opportunity(events)

    events.to_csv(output_dir / "entry_timing_events.csv", index=False)
    summary.to_csv(output_dir / "entry_timing_summary.csv", index=False)
    yearly.to_csv(output_dir / "entry_timing_yearly.csv", index=False)
    pullbacks.to_csv(output_dir / "pullback_opportunity_summary.csv", index=False)
    write_report(
        summary,
        yearly,
        pullbacks,
        output_dir / "entry_timing_report_zh.md",
    )

    print(f"Filtered candidates: {len(df):,}")
    print(f"Entry events: {len(events):,}")
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
