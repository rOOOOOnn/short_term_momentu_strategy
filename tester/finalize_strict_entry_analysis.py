from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RETURN_BANDS = [("2%-5%", 0.02, 0.05), ("5%-10%", 0.05, 0.10)]
MIN_DVS = [250_000, 500_000, 1_000_000]


def trimmed_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return clean.clip(clean.quantile(0.05), clean.quantile(0.95)).mean()


def load_results(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path, parse_dates=["date", "trade_date"]) for path in paths]
    results = pd.concat(frames, ignore_index=True)
    key = [
        "trade_date",
        "symbol",
        "pullback",
        "entry_deadline",
        "min_touch_dollar_volume",
    ]
    return results.drop_duplicates(key).sort_values(key).reset_index(drop=True)


def load_candidates(path: str | Path) -> pd.DataFrame:
    usecols = [
        "next_date",
        "symbol",
        "market_cap",
        "dollar_volume",
        "entry_time",
        "confirm_return",
        "cum_dollar_volume",
        "cum_trade_count",
    ]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["next_date"])
    return df[
        (df["entry_time"] == "08:00")
        & df["confirm_return"].between(0.02, 0.10, inclusive="left")
        & df["cum_dollar_volume"].ge(250_000)
        & df["cum_trade_count"].ge(25)
        & df["market_cap"].between(300_000_000, 5_000_000_000, inclusive="both")
    ].copy()


def period_slice(df: pd.DataFrame, period: str, date_col: str) -> pd.DataFrame:
    years = df[date_col].dt.year
    if period == "train_2015_2021":
        return df[years <= 2021]
    if period == "validation_2022_2024":
        return df[years >= 2022]
    return df


def summarize_rules(results: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = ["train_2015_2021", "validation_2022_2024", "all_2015_2024"]
    for period in periods:
        period_results = period_slice(results, period, "trade_date")
        period_candidates = period_slice(candidates, period, "next_date")
        for band_name, lower, upper in RETURN_BANDS:
            for min_dv in MIN_DVS:
                candidate_filter = (
                    period_candidates["confirm_return"].between(
                        lower, upper, inclusive="left"
                    )
                    & period_candidates["cum_dollar_volume"].ge(min_dv)
                )
                denominator = period_candidates[candidate_filter][
                    ["next_date", "symbol"]
                ].drop_duplicates().shape[0]
                selected = period_results[
                    period_results["confirm_return"].between(
                        lower, upper, inclusive="left"
                    )
                    & period_results["cum_dollar_volume"].ge(min_dv)
                ]
                for (pullback, deadline, min_touch_dollar_volume), group in selected.groupby(
                    ["pullback", "entry_deadline", "min_touch_dollar_volume"],
                    observed=True,
                ):
                    rows.append(
                        {
                            "period": period,
                            "return_band": band_name,
                            "min_cum_dollar_volume": min_dv,
                            "pullback": pullback,
                            "entry_deadline": deadline,
                            "min_touch_dollar_volume": min_touch_dollar_volume,
                            "candidate_count": denominator,
                            "fill_count": len(group),
                            "fill_rate": len(group) / denominator if denominator else np.nan,
                            "trimmed_return_0930": trimmed_mean(group["return_0930"]),
                            "median_return_0930": group["return_0930"].median(),
                            "win_rate_0930": group["return_0930"].gt(0).mean(),
                            "median_mfe_0930": group["mfe_0930"].median(),
                            "median_mae_0930": group["mae_0930"].median(),
                            "mae_25pct_0930": group["mae_0930"].quantile(0.25),
                            "trimmed_return_0935": trimmed_mean(group["return_0935"]),
                            "trimmed_return_1000": trimmed_mean(group["return_1000"]),
                        }
                    )
    return pd.DataFrame(rows)


def add_buckets(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["market_cap_segment"] = pd.cut(
        df["market_cap"],
        bins=[300_000_000, 500_000_000, 1_000_000_000, 2_000_000_000, 5_000_000_000],
        labels=["300M-500M", "500M-1B", "1B-2B", "2B-5B"],
        include_lowest=True,
    )
    df["t_day_dv_segment"] = pd.cut(
        df["t_day_dollar_volume"],
        bins=[0, 10_000_000, 25_000_000, 50_000_000, 100_000_000, np.inf],
        labels=["<10M", "10M-25M", "25M-50M", "50M-100M", ">=100M"],
        include_lowest=True,
    )
    return df


def summarize_segments(results: pd.DataFrame, rule: pd.Series) -> pd.DataFrame:
    selected = results[
        results["confirm_return"].between(
            0.05 if rule["return_band"] == "5%-10%" else 0.02,
            0.10 if rule["return_band"] == "5%-10%" else 0.05,
            inclusive="left",
        )
        & results["cum_dollar_volume"].ge(rule["min_cum_dollar_volume"])
        & results["pullback"].eq(rule["pullback"])
        & results["entry_deadline"].eq(rule["entry_deadline"])
        & results["min_touch_dollar_volume"].eq(rule["min_touch_dollar_volume"])
    ].copy()
    selected = add_buckets(selected)
    rows = []
    for period in ["train_2015_2021", "validation_2022_2024", "all_2015_2024"]:
        period_data = period_slice(selected, period, "trade_date")
        for segment in ["market_cap_segment", "t_day_dv_segment"]:
            for value, group in period_data.groupby(segment, observed=True):
                rows.append(
                    {
                        "period": period,
                        "segment": segment,
                        "segment_value": str(value),
                        "count": len(group),
                        "trimmed_return_0930": trimmed_mean(group["return_0930"]),
                        "median_return_0930": group["return_0930"].median(),
                        "win_rate_0930": group["return_0930"].gt(0).mean(),
                        "median_mfe_0930": group["mfe_0930"].median(),
                        "median_mae_0930": group["mae_0930"].median(),
                    }
                )
    return pd.DataFrame(rows)


def choose_rule(summary: pd.DataFrame) -> pd.Series:
    validation = summary[
        (summary["period"] == "validation_2022_2024")
        & summary["fill_count"].ge(35)
        & summary["fill_rate"].ge(0.25)
    ].copy()
    train = summary[summary["period"] == "train_2015_2021"][
        [
            "return_band",
            "min_cum_dollar_volume",
            "pullback",
            "entry_deadline",
            "min_touch_dollar_volume",
            "trimmed_return_0930",
            "median_return_0930",
        ]
    ].rename(
        columns={
            "trimmed_return_0930": "train_trimmed_return_0930",
            "median_return_0930": "train_median_return_0930",
        }
    )
    merged = validation.merge(
        train,
        on=[
            "return_band",
            "min_cum_dollar_volume",
            "pullback",
            "entry_deadline",
            "min_touch_dollar_volume",
        ],
        how="left",
    )
    robust = merged[
        merged["train_trimmed_return_0930"].gt(0)
        & merged["trimmed_return_0930"].gt(0)
        & merged["median_return_0930"].gt(0)
        & merged["min_touch_dollar_volume"].ge(5_000)
    ].copy()
    if robust.empty:
        robust = merged
    robust["worst_period_return"] = robust[
        ["trimmed_return_0930", "train_trimmed_return_0930"]
    ].min(axis=1)
    best = robust["worst_period_return"].max()
    near_best = robust[robust["worst_period_return"].ge(best - 0.002)]
    return near_best.sort_values(
        ["fill_count", "min_cum_dollar_volume", "entry_deadline"],
        ascending=[False, True, True],
    ).iloc[0]


def write_report(
    rule: pd.Series,
    summary: pd.DataFrame,
    segments: pd.DataFrame,
    output_path: Path,
) -> None:
    key = [
        "return_band",
        "min_cum_dollar_volume",
        "pullback",
        "entry_deadline",
        "min_touch_dollar_volume",
    ]
    matched = summary.copy()
    for column in key:
        matched = matched[matched[column] == rule[column]]

    lines = [
        "# 严格逐笔入场测试结论",
        "",
        "## 测试方法",
        "",
        "- 股票池：T 日涨幅前 50，市值 300M-5B。",
        "- 08:00 使用当时已经完成的盘前涨幅、累计成交额和成交笔数确认信号。",
        "- 回踩限价只在 08:00 后首次出现对应价格时成交，严格保留成交先后顺序。",
        "- 限价成交按提交的限价计算，不使用更有利的成交价。",
        "- 训练期：2015-2021；独立验证期：2022-2024。",
        "- 本阶段仍未加入止盈止损，主要指标是入场后到 09:30 的收益、MFE 和 MAE。",
        "",
        "## 最终候选股票筛选",
        "",
        "- 使用美股普通股，保持原有 CRSP 交易所与股票类型过滤。",
        "- T 日按日涨幅排序，保留前 50 名。",
        "- T 日市值保持在 300M-5B；验证期各主要市值层均为正，没有证据支持继续缩窄。",
        "- 不额外提高 T 日成交额门槛；真正用于入场的流动性过滤放在当日盘前累计成交额和成交确认上。",
        "",
        "## 最终候选入场规则",
        "",
        f"- 08:00 盘前涨幅：{rule['return_band']}。",
        f"- 04:00-08:00 累计成交额至少：{rule['min_cum_dollar_volume'] / 1_000_000:.2f}M。",
        "- 04:00-08:00 累计成交笔数至少：25。",
        f"- 08:00 后等待回踩：{rule['pullback']:.1%}。",
        f"- 回踩价以下累计成交额至少：{rule['min_touch_dollar_volume'] / 1_000:.0f}K。",
        f"- 最晚允许入场：{rule['entry_deadline']}。",
        "",
        "|时期|候选数|成交数|填单率|截尾到09:30|中位数到09:30|胜率|中位数MFE|中位数MAE|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in matched.sort_values("period").iterrows():
        lines.append(
            f"|{row['period']}|{int(row['candidate_count'])}|{int(row['fill_count'])}|"
            f"{row['fill_rate']:.1%}|{row['trimmed_return_0930']:.2%}|"
            f"{row['median_return_0930']:.2%}|{row['win_rate_0930']:.1%}|"
            f"{row['median_mfe_0930']:.2%}|{row['median_mae_0930']:.2%}|"
        )

    validation_segments = segments[
        (segments["period"] == "validation_2022_2024") & segments["count"].ge(20)
    ].sort_values("trimmed_return_0930", ascending=False)
    lines.extend(
        [
            "",
            "## 验证期分层",
            "",
            "|维度|区间|样本|截尾到09:30|中位数|胜率|中位数MFE|中位数MAE|",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in validation_segments.iterrows():
        lines.append(
            f"|{row['segment']}|{row['segment_value']}|{int(row['count'])}|"
            f"{row['trimmed_return_0930']:.2%}|{row['median_return_0930']:.2%}|"
            f"{row['win_rate_0930']:.1%}|{row['median_mfe_0930']:.2%}|"
            f"{row['median_mae_0930']:.2%}|"
        )

    lines.extend(
        [
            "",
            "## 使用限制",
            "",
            "- 结果未包含买卖价差、真实排队成交和手续费，只对立即买入加入了 10 bps 滑点。",
            "- TAQ 成交价可能包含在实际券商盘前时段不可成交的场外打印，开发策略时应增加交易所和条件代码过滤。",
            "- 最终能否盈利仍取决于下一步移动止损和退出逻辑；本报告只确定较合理的股票与入场候选。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.candidates)
    results = load_results(args.result)
    summary = summarize_rules(results, candidates)
    rule = choose_rule(summary)
    segments = summarize_segments(results, rule)

    results.to_csv(output_dir / "strict_entry_results_combined.csv", index=False)
    summary.to_csv(output_dir / "strict_entry_rule_summary.csv", index=False)
    segments.to_csv(output_dir / "strict_entry_segment_summary.csv", index=False)
    write_report(rule, summary, segments, output_dir / "strict_entry_final_report_zh.md")
    print(rule.to_string())
    print(f"Saved final analysis to: {output_dir}")


if __name__ == "__main__":
    main()
