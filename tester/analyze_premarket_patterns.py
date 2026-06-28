# analyze_premarket_patterns.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOWS = [
    "pre_0400_0700",
    "pre_0700_0800",
    "pre_0800_0900",
    "pre_0900_0915",
    "pre_0915_0930",
]

FOCUS_WINDOWS = [
    "pre_0400_0700",
    "pre_0700_0800",
    "pre_0800_0900",
    "pre_0900_0915",
    "pre_0915_0930",
]

SEGMENT_COLUMNS = [
    "market_cap_bucket",
    "dollar_volume_bucket",
    "return_bucket",
    "relative_volume_bucket",
    "exchange",
]

KEY_PATTERNS = [
    "any positive premarket",
    "0%-5%, dv>=1M",
    "0%-5%, dv>=5M",
    "0%-5%, dv>=1M, trades>=100",
    "0%-5%, dv>=1M, low pullback",
    "0%-5%, dv>=1M, near high and momentum",
    "2%-10%, dv>=1M, near high and momentum",
]

RETURN_BINS = [-np.inf, -0.02, 0, 0.02, 0.05, 0.10, np.inf]
RETURN_LABELS = ["<-2%", "-2%-0%", "0%-2%", "2%-5%", "5%-10%", ">10%"]

DOLLAR_VOLUME_BINS = [0, 50_000, 250_000, 1_000_000, 5_000_000, 20_000_000, np.inf]
DOLLAR_VOLUME_LABELS = ["<50K", "50K-250K", "250K-1M", "1M-5M", "5M-20M", ">20M"]


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value * 100:.2f}%"


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional deps."""

    if df.empty:
        return "_No rows._"

    formatted = df.copy()
    for col in formatted.columns:
        if "return" in col or "rate" in col or "pullback" in col:
            formatted[col] = formatted[col].map(pct)
        elif "dollar_volume" in col:
            formatted[col] = formatted[col].map(money)
        elif col == "count":
            formatted[col] = formatted[col].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        else:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))

    headers = list(formatted.columns)
    rows = formatted.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where((b.notna()) & (b != 0), a / b, np.nan)


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "next_date"])
    df["year"] = df["date"].dt.year
    return df


def add_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for window in WINDOWS:
        open_col = f"{window}_open"
        high_col = f"{window}_high"
        low_col = f"{window}_low"
        close_col = f"{window}_close"
        dv_col = f"{window}_dollar_volume"
        ret_from_close_col = f"{window}_return_from_t_close"

        required = [open_col, high_col, low_col, close_col]
        if not all(col in df.columns for col in required):
            continue

        price_range = df[high_col] - df[low_col]
        df[f"{window}_close_location"] = safe_divide(df[close_col] - df[low_col], price_range)
        df[f"{window}_pullback_from_high"] = df[close_col] / df[high_col] - 1
        df[f"{window}_bounce_from_low"] = df[close_col] / df[low_col] - 1

        if ret_from_close_col in df.columns:
            df[f"{window}_return_bucket"] = pd.cut(
                df[ret_from_close_col],
                bins=RETURN_BINS,
                labels=RETURN_LABELS,
                right=False,
            )

        if dv_col in df.columns:
            df[f"{window}_dollar_volume_bucket"] = pd.cut(
                df[dv_col],
                bins=DOLLAR_VOLUME_BINS,
                labels=DOLLAR_VOLUME_LABELS,
                right=False,
            )

    return df


def summarize_subset(df: pd.DataFrame, window: str, label: str, mask: pd.Series) -> dict[str, object]:
    x = df[mask].copy()

    to_open = f"{window}_to_next_open_return"
    to_close = f"{window}_to_next_close_return"
    to_high = f"{window}_to_next_high_return"
    to_low = f"{window}_to_next_low_return"
    ret_from_close = f"{window}_return_from_t_close"
    window_ret = f"{window}_return"
    dv = f"{window}_dollar_volume"
    pullback = f"{window}_pullback_from_high"
    close_location = f"{window}_close_location"

    cols = [
        col
        for col in [
            to_open,
            to_close,
            to_high,
            to_low,
            ret_from_close,
            window_ret,
            dv,
            pullback,
            close_location,
        ]
        if col in x.columns
    ]
    x = x.dropna(subset=[to_open, to_close]) if to_open in x.columns and to_close in x.columns else x

    row = {
        "window": window,
        "pattern": label,
        "count": len(x),
    }

    if not len(x):
        return row

    row.update(
        {
            "avg_premarket_return_from_t_close": x[ret_from_close].mean() if ret_from_close in cols else np.nan,
            "median_premarket_return_from_t_close": x[ret_from_close].median() if ret_from_close in cols else np.nan,
            "avg_window_return": x[window_ret].mean() if window_ret in cols else np.nan,
            "avg_dollar_volume": x[dv].mean() if dv in cols else np.nan,
            "median_dollar_volume": x[dv].median() if dv in cols else np.nan,
            "avg_close_location": x[close_location].mean() if close_location in cols else np.nan,
            "avg_pullback_from_high": x[pullback].mean() if pullback in cols else np.nan,
            "avg_to_open_return": x[to_open].mean() if to_open in cols else np.nan,
            "median_to_open_return": x[to_open].median() if to_open in cols else np.nan,
            "win_rate_to_open": (x[to_open] > 0).mean() if to_open in cols else np.nan,
            "avg_to_close_return": x[to_close].mean() if to_close in cols else np.nan,
            "median_to_close_return": x[to_close].median() if to_close in cols else np.nan,
            "win_rate_to_close": (x[to_close] > 0).mean() if to_close in cols else np.nan,
            "avg_to_high_return": x[to_high].mean() if to_high in cols else np.nan,
            "median_to_high_return": x[to_high].median() if to_high in cols else np.nan,
            "avg_to_low_return": x[to_low].mean() if to_low in cols else np.nan,
            "median_to_low_return": x[to_low].median() if to_low in cols else np.nan,
            "high_low_spread": x[to_high].mean() - abs(x[to_low].mean()) if to_high in cols and to_low in cols else np.nan,
        }
    )

    return row


def build_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for window in FOCUS_WINDOWS:
        ret = f"{window}_return_from_t_close"
        window_ret = f"{window}_return"
        dv = f"{window}_dollar_volume"
        trades = f"{window}_trade_count"
        close_location = f"{window}_close_location"
        pullback = f"{window}_pullback_from_high"

        if not all(col in df.columns for col in [ret, window_ret, dv, trades, close_location, pullback]):
            continue

        base = df[ret].notna() & df[dv].notna()

        patterns = {
            "any positive premarket": base & (df[ret] > 0),
            "0%-2% from T close, dv>=250K": base & df[ret].between(0, 0.02, inclusive="left") & (df[dv] >= 250_000),
            "2%-5% from T close, dv>=250K": base & df[ret].between(0.02, 0.05, inclusive="left") & (df[dv] >= 250_000),
            "5%-10% from T close, dv>=250K": base & df[ret].between(0.05, 0.10, inclusive="left") & (df[dv] >= 250_000),
            ">10% from T close, dv>=250K": base & (df[ret] >= 0.10) & (df[dv] >= 250_000),
            "0%-5%, dv>=1M": base & df[ret].between(0, 0.05, inclusive="left") & (df[dv] >= 1_000_000),
            "0%-5%, dv>=5M": base & df[ret].between(0, 0.05, inclusive="left") & (df[dv] >= 5_000_000),
            "0%-5%, dv>=1M, closes near high": base
            & df[ret].between(0, 0.05, inclusive="left")
            & (df[dv] >= 1_000_000)
            & (df[close_location] >= 0.80),
            "0%-5%, dv>=1M, positive window momentum": base
            & df[ret].between(0, 0.05, inclusive="left")
            & (df[dv] >= 1_000_000)
            & (df[window_ret] > 0),
            "0%-5%, dv>=1M, near high and momentum": base
            & df[ret].between(0, 0.05, inclusive="left")
            & (df[dv] >= 1_000_000)
            & (df[close_location] >= 0.80)
            & (df[window_ret] > 0),
            "2%-10%, dv>=1M, near high and momentum": base
            & df[ret].between(0.02, 0.10, inclusive="left")
            & (df[dv] >= 1_000_000)
            & (df[close_location] >= 0.80)
            & (df[window_ret] > 0),
            "0%-5%, dv>=1M, low pullback": base
            & df[ret].between(0, 0.05, inclusive="left")
            & (df[dv] >= 1_000_000)
            & (df[pullback] >= -0.01),
            "0%-5%, dv>=1M, trades>=100": base
            & df[ret].between(0, 0.05, inclusive="left")
            & (df[dv] >= 1_000_000)
            & (df[trades] >= 100),
        }

        for label, mask in patterns.items():
            rows.append(summarize_subset(df, window, label, mask))

    summary = pd.DataFrame(rows)
    return summary.sort_values(["window", "avg_to_open_return"], ascending=[True, False])


def build_segment_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Run key pattern summaries inside market-cap/liquidity/momentum segments."""

    rows = []

    for segment_col in SEGMENT_COLUMNS:
        if segment_col not in df.columns:
            continue

        for segment_value, segment_df in df.groupby(segment_col, observed=True):
            if pd.isna(segment_value) or len(segment_df) < 200:
                continue

            pattern_summary = build_pattern_summary(segment_df)
            if pattern_summary.empty:
                continue

            pattern_summary = pattern_summary[pattern_summary["pattern"].isin(KEY_PATTERNS)].copy()
            if pattern_summary.empty:
                continue

            pattern_summary.insert(0, "segment_value", str(segment_value))
            pattern_summary.insert(0, "segment", segment_col)
            rows.append(pattern_summary)

    if not rows:
        return pd.DataFrame()

    summary = pd.concat(rows, ignore_index=True)
    return summary.sort_values(["avg_to_open_return", "count"], ascending=[False, False])


def build_two_way_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize a focused setup by market cap and T-day dollar-volume buckets."""

    rows = []
    pairs = [
        ("market_cap_bucket", "dollar_volume_bucket"),
        ("market_cap_bucket", "return_bucket"),
        ("dollar_volume_bucket", "return_bucket"),
        ("market_cap_bucket", "relative_volume_bucket"),
    ]

    for window in FOCUS_WINDOWS:
        ret = f"{window}_return_from_t_close"
        dv = f"{window}_dollar_volume"
        trades = f"{window}_trade_count"
        to_open = f"{window}_to_next_open_return"
        to_close = f"{window}_to_next_close_return"
        to_high = f"{window}_to_next_high_return"
        to_low = f"{window}_to_next_low_return"

        required = [ret, dv, trades, to_open, to_close, to_high, to_low]
        if not all(col in df.columns for col in required):
            continue

        setup = df[
            df[ret].between(0, 0.05, inclusive="left")
            & (df[dv] >= 1_000_000)
            & (df[trades] >= 100)
        ].copy()

        for col_a, col_b in pairs:
            if col_a not in setup.columns or col_b not in setup.columns:
                continue

            grouped = (
                setup.groupby([col_a, col_b], observed=True)
                .agg(
                    count=(to_open, "count"),
                    avg_to_open_return=(to_open, "mean"),
                    median_to_open_return=(to_open, "median"),
                    win_rate_to_open=(to_open, lambda x: (x > 0).mean()),
                    avg_to_close_return=(to_close, "mean"),
                    median_to_close_return=(to_close, "median"),
                    win_rate_to_close=(to_close, lambda x: (x > 0).mean()),
                    avg_to_high_return=(to_high, "mean"),
                    avg_to_low_return=(to_low, "mean"),
                )
                .reset_index()
            )
            grouped = grouped[grouped["count"] >= 100]
            if grouped.empty:
                continue

            grouped.insert(0, "setup", "0%-5%, premarket dv>=1M, trades>=100")
            grouped.insert(0, "window", window)
            grouped.insert(1, "segment_pair", f"{col_a}+{col_b}")
            rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).sort_values(
        ["avg_to_open_return", "count"], ascending=[False, False]
    )


def build_heatmap_data(df: pd.DataFrame, window: str, outcome: str) -> pd.DataFrame:
    ret_bucket = f"{window}_return_bucket"
    dv_bucket = f"{window}_dollar_volume_bucket"
    outcome_col = f"{window}_{outcome}"

    if not all(col in df.columns for col in [ret_bucket, dv_bucket, outcome_col]):
        return pd.DataFrame()

    table = (
        df.groupby([ret_bucket, dv_bucket], observed=True)[outcome_col]
        .mean()
        .unstack(dv_bucket)
        .reindex(index=RETURN_LABELS, columns=DOLLAR_VOLUME_LABELS)
    )
    return table


def save_heatmap(table: pd.DataFrame, title: str, path: Path) -> None:
    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    values = table.astype(float).values
    vmax = np.nanpercentile(np.abs(values), 95)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.05
    im = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title(title)
    ax.set_xlabel("Window dollar volume")
    ax.set_ylabel("Window return from T close")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value * 100:.1f}%", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Average return")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_pattern_bar(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return

    plot_df = summary[summary["count"] >= 200].copy()
    plot_df = plot_df.sort_values("avg_to_open_return", ascending=False).head(18)
    labels = plot_df["window"] + "\n" + plot_df["pattern"]

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(plot_df))
    ax.barh(y - 0.18, plot_df["avg_to_open_return"] * 100, height=0.35, label="to open")
    ax.barh(y + 0.18, plot_df["avg_to_close_return"] * 100, height=0.35, label="to close")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Average return after window close (%)")
    ax.set_title("Best pattern averages, minimum 200 observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_yearly_chart(df: pd.DataFrame, path: Path) -> None:
    candidates = []
    for window in FOCUS_WINDOWS:
        ret = f"{window}_return_from_t_close"
        dv = f"{window}_dollar_volume"
        outcome = f"{window}_to_next_open_return"

        if not all(col in df.columns for col in [ret, dv, outcome]):
            continue

        mask = df[ret].between(0, 0.05, inclusive="left") & (df[dv] >= 1_000_000)
        temp = df[mask].groupby("year")[outcome].mean().reset_index()
        temp["window"] = window
        candidates.append(temp)

    if not candidates:
        return

    plot_df = pd.concat(candidates, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for window, group in plot_df.groupby("window"):
        ax.plot(group["year"], group[f"{window}_to_next_open_return"] * 100, marker="o", label=window)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Yearly average return to regular open, 0%-5% premarket and dv>=1M")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average return to open (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_segment_bar(segment_summary: pd.DataFrame, path: Path) -> None:
    if segment_summary.empty:
        return

    plot_df = segment_summary[segment_summary["count"] >= 300].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("avg_to_open_return", ascending=False).head(18)
    labels = (
        plot_df["segment"]
        + "="
        + plot_df["segment_value"]
        + "\n"
        + plot_df["window"]
        + " | "
        + plot_df["pattern"]
    )

    fig, ax = plt.subplots(figsize=(13, 8))
    y = np.arange(len(plot_df))
    ax.barh(y - 0.18, plot_df["avg_to_open_return"] * 100, height=0.35, label="to open")
    ax.barh(y + 0.18, plot_df["avg_to_close_return"] * 100, height=0.35, label="to close")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Average return after window close (%)")
    ax.set_title("Best segmented pattern averages, minimum 300 observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_segment_heatmap(
    two_way_summary: pd.DataFrame,
    window: str,
    segment_pair: str,
    path: Path,
) -> None:
    if two_way_summary.empty:
        return

    data = two_way_summary[
        (two_way_summary["window"] == window)
        & (two_way_summary["segment_pair"] == segment_pair)
    ].copy()
    if data.empty:
        return

    col_a, col_b = segment_pair.split("+", 1)
    if col_a not in data.columns or col_b not in data.columns:
        return

    table = data.pivot(index=col_a, columns=col_b, values="avg_to_open_return")
    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    values = table.astype(float).values
    vmax = np.nanpercentile(np.abs(values), 95)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.01
    im = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title(f"{window}: avg to-open return by {segment_pair}")
    ax.set_xlabel(col_b)
    ax.set_ylabel(col_a)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value * 100:.2f}%", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Average to-open return")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    enriched_path: Path,
    summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    two_way_summary: pd.DataFrame,
    total_rows: int,
) -> None:
    report_path = output_dir / "premarket_pattern_research_report.md"

    focus = summary[summary["count"] >= 200].copy()
    best_open = focus.sort_values("avg_to_open_return", ascending=False).head(10)
    worst_close = focus.sort_values("avg_to_close_return", ascending=True).head(10)
    best_segments = segment_summary[segment_summary["count"] >= 300].head(12) if not segment_summary.empty else pd.DataFrame()
    best_two_way = two_way_summary.head(12) if not two_way_summary.empty else pd.DataFrame()

    lines = [
        "# Premarket Pattern Research Report",
        "",
        f"Input file: `{enriched_path}`",
        f"Rows analyzed: `{total_rows:,}`",
        "",
        "This report is an event-study layer, not a simulated trading backtest. Returns are measured from the end of each premarket window to regular open, close, high, and low.",
        "",
        "## Main Takeaways",
        "",
        "- The simple rule `T-day top gainers + premarket up` does not show a clean positive edge after the observation window.",
        "- Stronger premarket return mostly confirms that the move already happened from the prior close; after-window returns to open/close are often flat or negative.",
        "- Bigger premarket moves have larger upside excursion to the day's high, but also materially larger downside excursion to the day's low.",
        "- Any viable version would likely need execution logic around fast profit-taking/risk control, not just filtering and holding to open or close.",
        "",
        "## Charts",
        "",
        "![Pattern bar chart](charts/pattern_bar.png)",
        "",
        "![Yearly chart](charts/yearly_to_open.png)",
        "",
        "![04:00-07:00 heatmap to open](charts/heatmap_pre_0400_0700_to_next_open.png)",
        "",
        "![07:00-08:00 heatmap to open](charts/heatmap_pre_0700_0800_to_next_open.png)",
        "",
        "![09:00-09:15 heatmap to open](charts/heatmap_pre_0900_0915_to_next_open.png)",
        "",
        "![09:15-09:30 heatmap to open](charts/heatmap_pre_0915_0930_to_next_open.png)",
        "",
        "![Segmented pattern bar chart](charts/segment_pattern_bar.png)",
        "",
        "![Segment heatmap](charts/segment_heatmap_pre_0915_0930_market_cap_dollar_volume.png)",
        "",
        "## Best Average Returns To Open",
        "",
        markdown_table(
            best_open[
            [
                "window",
                "pattern",
                "count",
                "avg_to_open_return",
                "median_to_open_return",
                "win_rate_to_open",
                "avg_to_close_return",
                "win_rate_to_close",
                "avg_to_high_return",
                "avg_to_low_return",
            ]
            ]
        ),
        "",
        "## Best Segmented Patterns",
        "",
        markdown_table(
            best_segments[
                [
                    "segment",
                    "segment_value",
                    "window",
                    "pattern",
                    "count",
                    "avg_to_open_return",
                    "median_to_open_return",
                    "win_rate_to_open",
                    "avg_to_close_return",
                    "win_rate_to_close",
                    "avg_to_high_return",
                    "avg_to_low_return",
                ]
            ]
            if not best_segments.empty
            else best_segments
        ),
        "",
        "## Best Two-Way Segments",
        "",
        markdown_table(
            best_two_way[
                [
                    col
                    for col in [
                        "window",
                        "segment_pair",
                        "setup",
                        "market_cap_bucket",
                        "dollar_volume_bucket",
                        "return_bucket",
                        "relative_volume_bucket",
                        "count",
                        "avg_to_open_return",
                        "median_to_open_return",
                        "win_rate_to_open",
                        "avg_to_close_return",
                        "win_rate_to_close",
                        "avg_to_high_return",
                        "avg_to_low_return",
                    ]
                    if col in best_two_way.columns
                ]
            ]
            if not best_two_way.empty
            else best_two_way
        ),
        "",
        "## Worst Average Returns To Close",
        "",
        markdown_table(
            worst_close[
            [
                "window",
                "pattern",
                "count",
                "avg_to_open_return",
                "win_rate_to_open",
                "avg_to_close_return",
                "median_to_close_return",
                "win_rate_to_close",
                "avg_to_high_return",
                "avg_to_low_return",
            ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "The most important reading is not whether `avg_next_open_return` is positive. That value uses the prior day's close as the base and therefore includes the premarket move itself. The relevant practical columns are `avg_to_open_return`, `avg_to_close_return`, `avg_to_high_return`, and `avg_to_low_return`, which start from the window close price.",
        "",
        "Across the tested shapes, moderate premarket continuation with meaningful dollar volume usually has weak or negative average continuation to regular open and worse continuation to close. Large premarket moves increase possible upside to the intraday high, but the average drawdown to the intraday low is often comparable or worse.",
        "",
        "## Output Files",
        "",
        "- `pattern_summary.csv`: metrics for predefined shape rules.",
        "- `charts/`: PNG charts used in this report.",
        "- `premarket_pattern_research_report.md`: this report.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent / "pattern_analysis"
    output_dir = Path(output_dir)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    df = add_shape_features(df)

    summary = build_pattern_summary(df)
    summary_path = output_dir / "pattern_summary.csv"
    summary.to_csv(summary_path, index=False)

    segment_summary = build_segment_pattern_summary(df)
    segment_summary_path = output_dir / "segment_pattern_summary.csv"
    segment_summary.to_csv(segment_summary_path, index=False)

    two_way_summary = build_two_way_segment_summary(df)
    two_way_summary_path = output_dir / "two_way_segment_summary.csv"
    two_way_summary.to_csv(two_way_summary_path, index=False)

    save_pattern_bar(summary, charts_dir / "pattern_bar.png")
    save_segment_bar(segment_summary, charts_dir / "segment_pattern_bar.png")
    save_yearly_chart(df, charts_dir / "yearly_to_open.png")

    for window in FOCUS_WINDOWS:
        for outcome, label in [
            ("to_next_open_return", "to open"),
            ("to_next_close_return", "to close"),
        ]:
            table = build_heatmap_data(df, window, outcome)
            save_heatmap(
                table,
                f"{window}: average return {label}",
                charts_dir / f"heatmap_{window}_{outcome.replace('_return', '')}.png",
            )

    save_segment_heatmap(
        two_way_summary,
        "pre_0915_0930",
        "market_cap_bucket+dollar_volume_bucket",
        charts_dir / "segment_heatmap_pre_0915_0930_market_cap_dollar_volume.png",
    )
    save_segment_heatmap(
        two_way_summary,
        "pre_0900_0915",
        "market_cap_bucket+dollar_volume_bucket",
        charts_dir / "segment_heatmap_pre_0900_0915_market_cap_dollar_volume.png",
    )

    write_report(output_dir, input_path, summary, segment_summary, two_way_summary, len(df))

    print(f"Saved pattern summary to: {summary_path}")
    print(f"Saved segment pattern summary to: {segment_summary_path}")
    print(f"Saved two-way segment summary to: {two_way_summary_path}")
    print(f"Saved report to: {output_dir / 'premarket_pattern_research_report.md'}")
    print(f"Saved charts to: {charts_dir}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to premarket_enriched_results.csv.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_analysis(args.input, args.output_dir)


if __name__ == "__main__":
    main()
