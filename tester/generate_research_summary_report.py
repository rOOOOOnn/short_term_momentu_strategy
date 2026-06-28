from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("screen_results/20150102_20241231_gainers_return_1d_top50/premarket")


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def signed_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:+.2f}%"


def money(value: float) -> str:
    if pd.isna(value):
        return ""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    return f"{value / 1_000:.0f}K"


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = df[columns].copy()
    if limit:
        view = view.head(limit)
    for col in view.columns:
        if col in {
            "trigger_floor",
            "coverage",
            "median_trigger_return",
            "median_mfe_to_day_high",
            "p_mfe_3",
            "p_mfe_5",
            "p_mfe_10",
            "median_mae_to_day_low",
            "p_mae_3",
            "p_mae_5",
            "trimmed_to_close",
            "trimmed_return_0930",
            "median_edge",
            "median_mfe_0930",
            "median_mae_0930",
            "trimmed_capital_return_0930",
            "avg_invested_fraction",
            "full_position_rate",
            "p_hit_up_3pct",
            "p_hit_up_5pct",
        }:
            view[col] = view[col].map(signed_pct if col in {"median_edge", "trimmed_to_close", "trimmed_return_0930", "trimmed_capital_return_0930", "median_mfe_to_day_high", "median_mae_to_day_low", "median_trigger_return", "median_mfe_0930", "median_mae_0930"} else pct)
        elif col == "min_cum_dollar_volume":
            view[col] = view[col].map(lambda x: money(float(x)))
        elif col == "median_trigger_time":
            view[col] = view[col].astype(str).str.replace("0 days ", "", regex=False).str.slice(0, 8)

    view = view.fillna("")
    headers = [str(col) for col in view.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(rows)


def save_bar_chart(
    path: Path,
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    percent_axis: bool = True,
) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, np.array(values) * 100 if percent_axis else values, width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_scatter(path: Path, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = 40 + df["trigger_count"] / df["trigger_count"].max() * 260
    colors = df["coverage"] * 100
    scatter = ax.scatter(
        df["median_mae_to_day_low"] * 100,
        df["median_mfe_to_day_high"] * 100,
        s=sizes,
        c=colors,
        cmap="viridis",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Dynamic trigger: MFE vs MAE")
    ax.set_xlabel("Median MAE to day low (%)")
    ax.set_ylabel("Median MFE to day high (%)")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Coverage (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_trigger_time_hist(path: Path, results: pd.DataFrame) -> None:
    focus = results[
        (results["trigger_floor"] == 0.05)
        & (results["min_cum_dollar_volume"] == 1_000_000)
        & (results["min_cum_trade_count"] == 100)
    ].copy()
    focus["trigger_minutes"] = pd.to_timedelta(focus["trigger_time"]).dt.total_seconds() / 60
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.arange(240, 571, 15)
    ax.hist(focus["trigger_minutes"], bins=bins, color="#4c78a8", alpha=0.85)
    ax.set_title("Trigger time distribution: +5%, $1M, 100 trades")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Triggered stock-days")
    ticks = np.arange(240, 571, 60)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t // 60):02d}:00" for t in ticks])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_report(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    full_summary = pd.read_csv(ROOT / "full_premarket_dynamic_triggers/full_premarket_dynamic_trigger_summary.csv")
    period_summary = pd.read_csv(ROOT / "full_premarket_dynamic_triggers/full_premarket_dynamic_trigger_period_summary.csv")
    full_results = pd.read_csv(ROOT / "full_premarket_dynamic_triggers/full_premarket_dynamic_trigger_results.csv")
    staged = load_csv(ROOT / "staged_entry_analysis/staged_entry_summary.csv")
    high_coverage = load_csv(ROOT / "high_coverage_entry_analysis/high_coverage_entry_summary.csv")

    full_summary["median_trigger_time"] = full_summary["median_trigger_time"].astype(str)
    focus = full_summary[
        (full_summary["trigger_floor"] == 0.05)
        & full_summary["min_cum_trade_count"].isin([50, 100, 200])
    ].copy()
    focus = focus.sort_values(["min_cum_dollar_volume", "min_cum_trade_count"])

    chart_rule = focus[focus["min_cum_trade_count"] == 100].copy()
    labels = [money(v) for v in chart_rule["min_cum_dollar_volume"]]
    save_bar_chart(
        charts_dir / "dynamic_trigger_mfe_mae_by_volume.png",
        labels,
        {
            "Median MFE": chart_rule["median_mfe_to_day_high"].tolist(),
            "Median MAE": chart_rule["median_mae_to_day_low"].tolist(),
        },
        "Dynamic trigger risk/reward by cumulative dollar volume",
        "Return (%)",
    )
    save_bar_chart(
        charts_dir / "dynamic_trigger_hit_rates_by_volume.png",
        labels,
        {
            "Hit +3%": chart_rule["p_mfe_3"].tolist(),
            "Hit +5%": chart_rule["p_mfe_5"].tolist(),
            "Drawdown -5%": chart_rule["p_mae_5"].tolist(),
        },
        "Dynamic trigger hit rates by cumulative dollar volume",
        "Probability (%)",
    )
    save_scatter(charts_dir / "dynamic_trigger_mfe_mae_scatter.png", full_summary)
    save_trigger_time_hist(charts_dir / "dynamic_trigger_time_distribution.png", full_results)

    if not period_summary.empty:
        period_focus = period_summary[
            (period_summary["trigger_floor"] == 0.05)
            & (period_summary["min_cum_dollar_volume"] == 1_000_000)
            & (period_summary["min_cum_trade_count"] == 100)
        ].copy()
        save_bar_chart(
            charts_dir / "train_validation_dynamic_trigger.png",
            period_focus["period"].tolist(),
            {
                "Median MFE": period_focus["median_mfe_to_day_high"].tolist(),
                "Median MAE": period_focus["median_mae_to_day_low"].tolist(),
                "Hit +5%": period_focus["p_mfe_5"].tolist(),
            },
            "Train vs validation: +5%, $1M, 100 trades",
            "Value (%)",
        )

    if not staged.empty:
        staged_focus = staged[
            (staged["period"] == "all_2015_2024")
            & staged["rule"].isin(["starter_weighted_4", "starter_equal_4", "starter_wide_4"])
        ].copy()
        save_bar_chart(
            charts_dir / "staged_entry_comparison.png",
            staged_focus["rule"].tolist(),
            {
                "Median MFE to 09:30": staged_focus["median_mfe_0930"].tolist(),
                "Median MAE to 09:30": staged_focus["median_mae_0930"].tolist(),
                "Capital return to 09:30": staged_focus["trimmed_capital_return_0930"].tolist(),
            },
            "Earlier staged-entry test to 09:30",
            "Return (%)",
        )

    recommended = full_summary[
        (full_summary["trigger_floor"] == 0.05)
        & (full_summary["min_cum_dollar_volume"] == 1_000_000)
        & (full_summary["min_cum_trade_count"] == 100)
    ].iloc[0]
    liquid = full_summary[
        (full_summary["trigger_floor"] == 0.05)
        & (full_summary["min_cum_dollar_volume"] == 5_000_000)
        & (full_summary["min_cum_trade_count"] == 100)
    ].iloc[0]

    sections = [
        "# 盘前动量策略研究总结",
        "",
        "本文档汇总当前项目里已经完成的筛选、盘前信号、动态触发和入场研究。核心目标不是证明固定持有收益，而是判断：T 日大涨股在下一交易日盘前出现动量后，是否仍有足够的当天上冲空间，以及下一步策略开发应该从哪里开始。",
        "",
        "## 最终候选方向",
        "",
        "- 股票池：T 日涨幅前 50。",
        "- 市值：300M-5B。300M-1B 的空间更大，但样本太少；300M-5B 更适合先开发通用策略，再按市值控制仓位。",
        "- 动态信号：04:00-09:30 任意普通股成交触发，不使用固定窗口。",
        "- 基准触发：相对 T 日收盘价上涨至少 5%，累计盘前成交额至少 1M，累计成交笔数至少 100。",
        "- 交易假设：触发后不直接追高满仓，而是进入观察状态，等待回踩、承接或重新突破后分批入场。",
        "",
        "## 基准动态触发结果",
        "",
        markdown_table(
            pd.DataFrame([recommended, liquid]),
            [
                "trigger_floor",
                "min_cum_dollar_volume",
                "min_cum_trade_count",
                "candidate_count",
                "trigger_count",
                "coverage",
                "median_trigger_time",
                "median_trigger_return",
                "median_mfe_to_day_high",
                "p_mfe_3",
                "p_mfe_5",
                "p_mfe_10",
                "median_mae_to_day_low",
                "p_mae_5",
                "trimmed_to_close",
            ],
        ),
        "",
        "解读：`>=5%, >=1M, >=100 trades` 覆盖率足够高，触发后到当天最高点的中位 MFE 约 4%，达到 +5% 的概率约 45%。但中位 MAE 接近 -8%，说明信号只证明“今天有波动空间”，不证明可以直接买入。",
        "",
        "![Dynamic trigger MFE/MAE](charts/dynamic_trigger_mfe_mae_by_volume.png)",
        "",
        "![Dynamic trigger hit rates](charts/dynamic_trigger_hit_rates_by_volume.png)",
        "",
        "![Dynamic trigger scatter](charts/dynamic_trigger_mfe_mae_scatter.png)",
        "",
        "![Trigger time distribution](charts/dynamic_trigger_time_distribution.png)",
        "",
        "## 训练期与验证期",
        "",
        markdown_table(
            period_summary[
                (period_summary["trigger_floor"] == 0.05)
                & (period_summary["min_cum_dollar_volume"] == 1_000_000)
                & (period_summary["min_cum_trade_count"] == 100)
            ],
            [
                "period",
                "trigger_count",
                "median_trigger_time",
                "median_trigger_return",
                "median_mfe_to_day_high",
                "p_mfe_3",
                "p_mfe_5",
                "p_mfe_10",
                "median_mae_to_day_low",
                "p_mae_5",
                "trimmed_to_close",
            ],
        ),
        "",
        "验证期 2022-2024 的 MFE 明显低于训练期，但仍保留约 3% 的中位冲高空间，+5% 命中率约 41%。这说明策略不能依赖旧市场环境里的极端波动，必须靠入场价格和止损结构改善盈亏比。",
        "",
        "![Train validation](charts/train_validation_dynamic_trigger.png)",
        "",
        "## 为什么之前 09:30 版本看起来很差",
        "",
        "之前的 staged-entry 测试只看到 09:30，等于只看开盘瞬间。完整盘前动态触发后到当天最高点的 MFE 显著更大，说明很多上冲发生在 09:30 之后。策略方向应从“盘前触发后开盘前必须冲高”改为“盘前确认能量后，捕捉当天盘中冲高”。",
        "",
    ]

    if not staged.empty:
        staged_table = staged[
            (staged["period"] == "all_2015_2024")
            & staged["rule"].isin(["starter_weighted_4", "starter_equal_4", "starter_wide_4"])
        ].copy()
        sections.extend(
            [
                markdown_table(
                    staged_table,
                    [
                        "rule",
                        "entry_count",
                        "avg_invested_fraction",
                        "full_position_rate",
                        "trimmed_capital_return_0930",
                        "median_mfe_0930",
                        "median_mae_0930",
                    ],
                ),
                "",
                "![Staged entry comparison](charts/staged_entry_comparison.png)",
                "",
            ]
        )

    if not high_coverage.empty:
        high_focus = high_coverage[
            (high_coverage["period"] == "all_2015_2024")
            & high_coverage["method"].isin(["fixed_080100", "fixed_090000", "pb2_fallback_090000"])
        ].copy()
        sections.extend(
            [
                "## 入场研究的保留结论",
                "",
                markdown_table(
                    high_focus,
                    [
                        "method",
                        "entry_count",
                        "median_mfe_0930",
                        "median_mae_0930",
                        "p_hit_up_3pct",
                        "p_hit_up_5pct",
                        "trimmed_return_0930",
                    ],
                ),
                "",
                "固定时间直接买入不适合做最终方案。早买有更大 MFE，但 MAE 也更深；晚买回撤较小但空间变窄。因此下一步应开发触发后的动态入场：回踩买、重新突破买、分批买，再配合移动止损。",
                "",
            ]
        )

    sections.extend(
        [
            "## 下一步策略开发",
            "",
            "1. 使用完整逐笔盘前触发结果作为信号层：`>=5%, >=1M, >=100 trades`。",
            "2. 触发后测试入场层：首次回踩 1%-3%、VWAP 附近承接、重新突破触发价、分批加仓。",
            "3. 测试退出层：+3%/+5% 分批止盈，-2%/-3% 初始止损，盈利后保本，最高价回撤 1%-2% 移动止损。",
            "4. 分市值控制仓位：300M-1B 波动空间更大但仓位更小；1B-5B 可以承担稍大仓位但预期空间更低。",
            "5. 所有最终策略必须用逐笔路径判断止盈止损谁先发生，不能只用当天 high/low。",
            "",
            "## 文件来源",
            "",
            "- 完整逐笔动态触发：`screen_results/.../full_premarket_dynamic_triggers/`。",
            "- 分批入场早期测试：`screen_results/.../staged_entry_analysis/`。",
            "- 高覆盖入场测试：`screen_results/.../high_coverage_entry_analysis/`。",
            "- 本报告图表由 `tester/generate_research_summary_report.py` 生成。",
        ]
    )

    (output_dir / "premarket_strategy_research_summary_zh.md").write_text(
        "\n".join(sections),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="docs/reports/premarket_strategy_summary",
    )
    args = parser.parse_args()
    build_report(Path(args.output_dir))
    print(f"Saved research summary report to: {args.output_dir}")


if __name__ == "__main__":
    main()
