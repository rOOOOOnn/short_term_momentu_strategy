from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MILESTONES = [
    ("07:00", "pre_0400_0700"),
    ("08:00", "pre_0700_0800"),
    ("09:00", "pre_0800_0900"),
    ("09:15", "pre_0900_0915"),
    ("09:30", "pre_0915_0930"),
]

RETURN_THRESHOLDS = [0.02, 0.03, 0.05, 0.10]
RETURN_CEILINGS = [0.10, 0.15, 0.20, np.inf]
DOLLAR_VOLUME_THRESHOLDS = [250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
TRADE_COUNT_THRESHOLDS = [50, 100, 200]


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:+.2f}%"


def trimmed_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return clean.clip(clean.quantile(0.05), clean.quantile(0.95)).mean()


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = df[columns].copy()
    if limit is not None:
        view = view.head(limit)

    pct_cols = [
        "median_trigger_return",
        "avg_mfe_to_day_high",
        "median_mfe_to_day_high",
        "p_mfe_3",
        "p_mfe_5",
        "p_mfe_10",
        "median_mae_to_day_low",
        "p_mae_3",
        "p_mae_5",
        "trimmed_to_close",
        "median_to_close",
        "median_edge",
    ]
    for col in pct_cols:
        if col in view.columns:
            view[col] = view[col].map(pct)
    for col in ["min_cum_dollar_volume"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: f"{int(float(x)):,}")
    for col in ["trigger_floor", "trigger_ceiling"]:
        if col in view.columns:
            view[col] = view[col].map(
                lambda x: "no cap" if np.isinf(float(x)) else pct(float(x))
            )

    view = view.fillna("")
    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def load_input(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "next_date"])
    df = df[
        df["market_cap"].between(300_000_000, 5_000_000_000, inclusive="both")
        & df["next_date"].notna()
    ].copy()
    return df


def build_milestone_events(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    cumulative_dv = pd.Series(0.0, index=df.index)
    cumulative_trades = pd.Series(0.0, index=df.index)

    for trigger_time, window in MILESTONES:
        close_col = f"{window}_close"
        dv_col = f"{window}_dollar_volume"
        trades_col = f"{window}_trade_count"
        to_high_col = f"{window}_to_next_high_return"
        to_low_col = f"{window}_to_next_low_return"
        to_close_col = f"{window}_to_next_close_return"

        required = [close_col, dv_col, trades_col, to_high_col, to_low_col, to_close_col]
        if not all(col in df.columns for col in required):
            continue

        cumulative_dv = cumulative_dv + df[dv_col].fillna(0)
        cumulative_trades = cumulative_trades + df[trades_col].fillna(0)

        event = df[
            [
                "date",
                "next_date",
                "symbol",
                "permno",
                "market_cap",
                "close",
                "return_1d",
                "dollar_volume",
            ]
        ].copy()
        event["trigger_time"] = trigger_time
        event["trigger_window"] = window
        event["trigger_price"] = df[close_col]
        event["trigger_return"] = event["trigger_price"] / event["close"] - 1
        event["cum_dollar_volume"] = cumulative_dv
        event["cum_trade_count"] = cumulative_trades
        event["mfe_to_day_high"] = df[to_high_col]
        event["mae_to_day_low"] = df[to_low_col]
        event["to_close"] = df[to_close_col]
        event = event[event["trigger_price"].gt(0) & event["trigger_return"].notna()]
        rows.append(event)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def first_trigger_for_rule(
    events: pd.DataFrame,
    trigger_floor: float,
    trigger_ceiling: float,
    min_cum_dollar_volume: float,
    min_cum_trade_count: int,
) -> pd.DataFrame:
    eligible = events[
        events["trigger_return"].between(trigger_floor, trigger_ceiling, inclusive="left")
        & events["cum_dollar_volume"].ge(min_cum_dollar_volume)
        & events["cum_trade_count"].ge(min_cum_trade_count)
    ].copy()
    if eligible.empty:
        return eligible
    eligible["trigger_order"] = pd.Categorical(
        eligible["trigger_time"],
        categories=[time for time, _ in MILESTONES],
        ordered=True,
    )
    return (
        eligible.sort_values(["next_date", "symbol", "trigger_order"])
        .drop_duplicates(["next_date", "symbol"], keep="first")
        .drop(columns=["trigger_order"])
    )


def summarize(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []

    for floor in RETURN_THRESHOLDS:
        for ceiling in RETURN_CEILINGS:
            if ceiling <= floor:
                continue
            for min_dv in DOLLAR_VOLUME_THRESHOLDS:
                for min_trades in TRADE_COUNT_THRESHOLDS:
                    selected = first_trigger_for_rule(events, floor, ceiling, min_dv, min_trades)
                    if len(selected) < 100:
                        continue

                    rule = {
                        "trigger_floor": floor,
                        "trigger_ceiling": ceiling,
                        "min_cum_dollar_volume": min_dv,
                        "min_cum_trade_count": min_trades,
                    }
                    row = {
                        **rule,
                        "n": len(selected),
                        "median_trigger_return": selected["trigger_return"].median(),
                        "median_trigger_time": selected["trigger_time"].median()
                        if False
                        else selected["trigger_time"].mode().iat[0],
                        "avg_mfe_to_day_high": selected["mfe_to_day_high"].mean(),
                        "median_mfe_to_day_high": selected["mfe_to_day_high"].median(),
                        "p_mfe_3": selected["mfe_to_day_high"].ge(0.03).mean(),
                        "p_mfe_5": selected["mfe_to_day_high"].ge(0.05).mean(),
                        "p_mfe_10": selected["mfe_to_day_high"].ge(0.10).mean(),
                        "median_mae_to_day_low": selected["mae_to_day_low"].median(),
                        "p_mae_3": selected["mae_to_day_low"].le(-0.03).mean(),
                        "p_mae_5": selected["mae_to_day_low"].le(-0.05).mean(),
                        "trimmed_to_close": trimmed_mean(selected["to_close"]),
                        "median_to_close": selected["to_close"].median(),
                        "median_edge": selected["mfe_to_day_high"].median()
                        + selected["mae_to_day_low"].median(),
                    }
                    rows.append(row)

                    with_period = selected.copy()
                    with_period["period"] = np.where(
                        with_period["next_date"].dt.year <= 2021,
                        "train_2015_2021",
                        "validation_2022_2024",
                    )
                    for period, group in with_period.groupby("period", observed=True):
                        if len(group) < 20:
                            continue
                        period_rows.append(
                            {
                                **rule,
                                "period": period,
                                "n": len(group),
                                "median_trigger_time": group["trigger_time"].mode().iat[0],
                                "median_trigger_return": group["trigger_return"].median(),
                                "median_mfe_to_day_high": group["mfe_to_day_high"].median(),
                                "p_mfe_3": group["mfe_to_day_high"].ge(0.03).mean(),
                                "p_mfe_5": group["mfe_to_day_high"].ge(0.05).mean(),
                                "p_mfe_10": group["mfe_to_day_high"].ge(0.10).mean(),
                                "median_mae_to_day_low": group["mae_to_day_low"].median(),
                                "p_mae_3": group["mae_to_day_low"].le(-0.03).mean(),
                                "p_mae_5": group["mae_to_day_low"].le(-0.05).mean(),
                                "trimmed_to_close": trimmed_mean(group["to_close"]),
                            }
                        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, pd.DataFrame(period_rows)

    summary["rank_score"] = (
        summary["median_mfe_to_day_high"]
        + 0.35 * summary["p_mfe_5"]
        - 0.35 * summary["p_mae_5"]
        + summary["median_edge"]
    )
    summary = summary.sort_values(
        ["rank_score", "median_mfe_to_day_high", "n"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary.insert(0, "rule_id", np.arange(1, len(summary) + 1))

    period_summary = pd.DataFrame(period_rows)
    if not period_summary.empty:
        period_summary = period_summary.merge(
            summary[
                [
                    "rule_id",
                    "trigger_floor",
                    "trigger_ceiling",
                    "min_cum_dollar_volume",
                    "min_cum_trade_count",
                ]
            ],
            on=[
                "trigger_floor",
                "trigger_ceiling",
                "min_cum_dollar_volume",
                "min_cum_trade_count",
            ],
            how="left",
        ).sort_values(["rule_id", "period"])

    return summary, period_summary


def write_report(output_dir: Path, summary: pd.DataFrame, period_summary: pd.DataFrame) -> None:
    top = summary.head(20)
    lines = [
        "# 动态盘前触发信号分析",
        "",
        "这份分析不固定入场窗口，而是从 04:00 到 09:30 顺序检查。",
        "第一次满足盘前涨幅、累计成交额、累计成交笔数的时间，就是信号触发时间。",
        "",
        "注意：当前版本基于已有窗口聚合数据，所以触发时间只能落在 07:00、08:00、09:00、09:15、09:30 五个节点。下一步逐笔 TAQ 版本可以精确到任意成交时间。",
        "",
        "## 股票池",
        "",
        "- 市值：300M-5B。",
        "- T 日涨幅前 50。",
        "- MFE：触发窗口收盘价到当天最高价。",
        "- MAE：触发窗口收盘价到当天最低价。",
        "",
        "## 排名前 20 的动态触发规则",
        "",
        markdown_table(
            top,
            [
                "rule_id",
                "trigger_floor",
                "trigger_ceiling",
                "min_cum_dollar_volume",
                "min_cum_trade_count",
                "n",
                "median_trigger_time",
                "median_trigger_return",
                "median_mfe_to_day_high",
                "p_mfe_3",
                "p_mfe_5",
                "p_mfe_10",
                "median_mae_to_day_low",
                "p_mae_3",
                "p_mae_5",
                "trimmed_to_close",
                "median_edge",
            ],
        ),
        "",
        "## 训练期/验证期拆分",
        "",
    ]

    if period_summary.empty:
        lines.append("没有足够样本生成拆分。")
    else:
        lines.append(
            markdown_table(
                period_summary,
                [
                    "rule_id",
                    "period",
                    "n",
                    "median_trigger_time",
                    "median_trigger_return",
                    "median_mfe_to_day_high",
                    "p_mfe_3",
                    "p_mfe_5",
                    "p_mfe_10",
                    "median_mae_to_day_low",
                    "p_mae_3",
                    "p_mae_5",
                    "trimmed_to_close",
                ],
                limit=40,
            )
        )

    lines.extend(
        [
            "",
            "## 解释",
            "",
            "如果只看当天最高点，这类信号确实有明显空间；但 MAE 同样很深。",
            "因此下一步不能只优化筛选条件，而要用逐笔数据测试触发后如何分批买入，以及 +3%/+5% 止盈和 -2%/-3% 止损谁先发生。",
        ]
    )

    (output_dir / "dynamic_premarket_trigger_report_zh.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    output = Path(output_dir) if output_dir else input_path.parent / "dynamic_premarket_triggers"
    output.mkdir(parents=True, exist_ok=True)

    df = load_input(input_path)
    events = build_milestone_events(df)
    summary, period_summary = summarize(events)
    if summary.empty:
        raise RuntimeError("No dynamic trigger rules met the minimum sample size.")

    events.to_csv(output / "dynamic_premarket_trigger_events.csv", index=False)
    summary.to_csv(output / "dynamic_premarket_trigger_summary.csv", index=False)
    period_summary.to_csv(output / "dynamic_premarket_trigger_period_summary.csv", index=False)
    write_report(output, summary, period_summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output = run(args.input, args.output_dir)
    print(f"Saved dynamic premarket trigger analysis to: {output}")


if __name__ == "__main__":
    main()
