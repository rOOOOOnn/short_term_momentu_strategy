from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = [
    "pre_0700_0800",
    "pre_0800_0900",
    "pre_0900_0915",
    "pre_0915_0930",
    "open_0930_0935",
]

RETURN_BANDS = [
    ("up_2_5", 0.02, 0.05),
    ("up_5_10", 0.05, 0.10),
    ("up_2_10", 0.02, 0.10),
    ("up_10_20", 0.10, 0.20),
]

DOLLAR_VOLUME_THRESHOLDS = [250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
TRADE_COUNT_THRESHOLDS = [50, 100, 200]
MARKET_CAP_RANGES = [
    ("300M-1B", 300_000_000, 1_000_000_000),
    ("1B-5B", 1_000_000_000, 5_000_000_000),
    ("300M-5B", 300_000_000, 5_000_000_000),
]


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:+.2f}%"


def trimmed_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return clean.clip(clean.quantile(0.05), clean.quantile(0.95)).mean()


def load_input(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "next_date"])
    df = df[df["market_cap"].between(300_000_000, 5_000_000_000, inclusive="both")].copy()
    return df


def summarize_signal_space(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for window in WINDOWS:
        close_col = f"{window}_close"
        ret_col = f"{window}_return_from_t_close"
        dv_col = f"{window}_dollar_volume"
        trades_col = f"{window}_trade_count"
        to_high_col = f"{window}_to_next_high_return"
        to_low_col = f"{window}_to_next_low_return"
        to_close_col = f"{window}_to_next_close_return"
        to_open_col = f"{window}_to_next_open_return"
        high_from_t_close_col = f"{window}_high_from_t_close"

        required = [
            close_col,
            ret_col,
            dv_col,
            trades_col,
            to_high_col,
            to_low_col,
            to_close_col,
            to_open_col,
        ]
        if not all(col in df.columns for col in required):
            continue

        base = df[df[close_col].gt(0) & df[ret_col].notna()].copy()
        if base.empty:
            continue

        for cap_name, cap_low, cap_high in MARKET_CAP_RANGES:
            cap_mask = base["market_cap"].between(cap_low, cap_high, inclusive="both")
            for band_name, low, high in RETURN_BANDS:
                ret_mask = base[ret_col].between(low, high, inclusive="left")
                for min_dv in DOLLAR_VOLUME_THRESHOLDS:
                    dv_mask = base[dv_col].ge(min_dv)
                    for min_trades in TRADE_COUNT_THRESHOLDS:
                        x = base[cap_mask & ret_mask & dv_mask & base[trades_col].ge(min_trades)]
                        if len(x) < 100:
                            continue

                        if high_from_t_close_col in x.columns:
                            peak_signal = x[high_from_t_close_col]
                        else:
                            peak_signal = x[ret_col]

                        rows.append(
                            {
                                "window": window,
                                "market_cap_range": cap_name,
                                "signal_band": band_name,
                                "min_dollar_volume": min_dv,
                                "min_trade_count": min_trades,
                                "n": len(x),
                                "avg_signal_return": x[ret_col].mean(),
                                "median_signal_return": x[ret_col].median(),
                                "avg_peak_signal_return": peak_signal.mean(),
                                "avg_mfe_to_day_high": x[to_high_col].mean(),
                                "median_mfe_to_day_high": x[to_high_col].median(),
                                "p_mfe_3": x[to_high_col].ge(0.03).mean(),
                                "p_mfe_5": x[to_high_col].ge(0.05).mean(),
                                "p_mfe_10": x[to_high_col].ge(0.10).mean(),
                                "avg_mae_to_day_low": x[to_low_col].mean(),
                                "median_mae_to_day_low": x[to_low_col].median(),
                                "p_mae_2": x[to_low_col].le(-0.02).mean(),
                                "p_mae_3": x[to_low_col].le(-0.03).mean(),
                                "p_mae_5": x[to_low_col].le(-0.05).mean(),
                                "trimmed_to_close": trimmed_mean(x[to_close_col]),
                                "median_to_close": x[to_close_col].median(),
                                "win_rate_to_close": x[to_close_col].gt(0).mean(),
                                "avg_to_open": x[to_open_col].mean(),
                                "mfe_mae_ratio": (
                                    x[to_high_col].mean() / abs(x[to_low_col].mean())
                                    if x[to_low_col].mean() < 0
                                    else np.nan
                                ),
                                "median_edge": x[to_high_col].median() + x[to_low_col].median(),
                            }
                        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["rank_score"] = (
        summary["median_mfe_to_day_high"]
        + 0.5 * summary["p_mfe_5"]
        - 0.5 * summary["p_mae_5"]
        + summary["median_edge"]
    )
    return summary.sort_values(
        ["rank_score", "median_mfe_to_day_high", "n"],
        ascending=[False, False, False],
    )


def add_period_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["period"] = np.where(out["next_date"].dt.year <= 2021, "train_2015_2021", "validation_2022_2024")
    return out


def summarize_by_period(df: pd.DataFrame, best_rules: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    df = add_period_column(df)

    for _, rule in best_rules.iterrows():
        window = rule["window"]
        ret_col = f"{window}_return_from_t_close"
        dv_col = f"{window}_dollar_volume"
        trades_col = f"{window}_trade_count"
        to_high_col = f"{window}_to_next_high_return"
        to_low_col = f"{window}_to_next_low_return"
        to_close_col = f"{window}_to_next_close_return"

        low, high = next(
            (lo, hi) for name, lo, hi in RETURN_BANDS if name == rule["signal_band"]
        )
        cap_low, cap_high = next(
            (lo, hi) for name, lo, hi in MARKET_CAP_RANGES if name == rule["market_cap_range"]
        )

        mask = (
            df["market_cap"].between(cap_low, cap_high, inclusive="both")
            & df[ret_col].between(low, high, inclusive="left")
            & df[dv_col].ge(rule["min_dollar_volume"])
            & df[trades_col].ge(rule["min_trade_count"])
        )
        selected = df[mask].copy()

        for period, group in selected.groupby("period", observed=True):
            rows.append(
                {
                    "rule_id": rule["rule_id"],
                    "period": period,
                    "n": len(group),
                    "median_mfe_to_day_high": group[to_high_col].median(),
                    "p_mfe_3": group[to_high_col].ge(0.03).mean(),
                    "p_mfe_5": group[to_high_col].ge(0.05).mean(),
                    "median_mae_to_day_low": group[to_low_col].median(),
                    "p_mae_3": group[to_low_col].le(-0.03).mean(),
                    "p_mae_5": group[to_low_col].le(-0.05).mean(),
                    "trimmed_to_close": trimmed_mean(group[to_close_col]),
                    "median_to_close": group[to_close_col].median(),
                    "win_rate_to_close": group[to_close_col].gt(0).mean(),
                }
            )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = df[columns].copy()
    if limit is not None:
        view = view.head(limit)
    numeric_pct_cols = [
        col
        for col in view.columns
        if col.startswith(("avg_", "median_", "p_", "trimmed_", "win_rate", "mfe_mae", "rank"))
    ]
    for col in numeric_pct_cols:
        if col in view.columns and pd.api.types.is_numeric_dtype(view[col]):
            if col in {"n", "rule_id", "mfe_mae_ratio", "rank_score"}:
                continue
            view[col] = view[col].map(pct)
    for col in ["min_dollar_volume"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: f"{int(x):,}")
    view = view.fillna("")
    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, summary: pd.DataFrame, period_summary: pd.DataFrame) -> None:
    top = summary.head(20).copy()
    lines = [
        "# 当天最高点空间信号分析",
        "",
        "这份分析不再把 09:30 当作终点，而是用当天最高价衡量入场后的最大上冲空间。",
        "入场价采用每个信号窗口的窗口收盘价，因此它仍然是信号研究，不是最终逐笔成交回测。",
        "",
        "## 口径",
        "",
        "- 股票池：T 日涨幅前 50 的结果表中，市值 300M-5B 及其子区间。",
        "- 信号窗口：07:00-08:00、08:00-09:00、09:00-09:15、09:15-09:30、09:30-09:35。",
        "- 信号强度：窗口收盘价相对 T 日收盘价上涨 2%-5%、5%-10%、2%-10%、10%-20%。",
        "- 流动性：窗口成交额和成交笔数分层扫描。",
        "- MFE：窗口收盘价到当天最高价。",
        "- MAE：窗口收盘价到当天最低价。",
        "",
        "## 排名前 20 的信号组合",
        "",
        markdown_table(
            top,
            [
                "window",
                "market_cap_range",
                "signal_band",
                "min_dollar_volume",
                "min_trade_count",
                "n",
                "median_mfe_to_day_high",
                "p_mfe_3",
                "p_mfe_5",
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
        lines.append("没有足够样本生成训练期/验证期拆分。")
    else:
        merged = period_summary.merge(
            summary[
                [
                    "rule_id",
                    "window",
                    "market_cap_range",
                    "signal_band",
                    "min_dollar_volume",
                    "min_trade_count",
                ]
            ],
            on="rule_id",
            how="left",
        )
        lines.append(
            markdown_table(
                merged.sort_values(["rule_id", "period"]),
                [
                    "rule_id",
                    "period",
                    "window",
                    "market_cap_range",
                    "signal_band",
                    "min_dollar_volume",
                    "min_trade_count",
                    "n",
                    "median_mfe_to_day_high",
                    "p_mfe_3",
                    "p_mfe_5",
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
            "## 结论用法",
            "",
            "这一步只能判断“有没有上涨空间”，不能证明能实际赚钱。",
            "如果某个组合到当天最高点的 MFE 很高，但 MAE 也很深，下一步必须用逐笔数据测试止盈和止损谁先触发。",
        ]
    )

    (output_dir / "daily_high_signal_space_report_zh.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    output = Path(output_dir) if output_dir else input_path.parent / "daily_high_signal_space"
    output.mkdir(parents=True, exist_ok=True)

    df = load_input(input_path)
    summary = summarize_signal_space(df)
    if summary.empty:
        raise RuntimeError("No signal combinations met the minimum sample size.")

    summary = summary.reset_index(drop=True)
    summary.insert(0, "rule_id", np.arange(1, len(summary) + 1))
    period_summary = summarize_by_period(df, summary.head(20))

    summary.to_csv(output / "daily_high_signal_space_summary.csv", index=False)
    period_summary.to_csv(output / "daily_high_signal_space_period_summary.csv", index=False)
    write_report(output, summary, period_summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output = run(args.input, args.output_dir)
    print(f"Saved daily high signal analysis to: {output}")


if __name__ == "__main__":
    main()
