from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.wrds.preprocess import connect_wrds
from data.wrds.preprocess_premarket import normalize_taq_symbol, sql_string_list


TRIGGER_FLOORS = [0.05, 0.07, 0.10]
MIN_DOLLAR_VOLUMES = [500_000, 1_000_000, 2_000_000, 5_000_000]
MIN_TRADE_COUNTS = [50, 100, 200]
PREMARKET_WINDOWS = [
    "pre_0400_0700",
    "pre_0700_0800",
    "pre_0800_0900",
    "pre_0900_0915",
    "pre_0915_0930",
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


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = df[columns].copy()
    if limit is not None:
        view = view.head(limit)

    pct_cols = [
        "trigger_floor",
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
        "win_rate_to_close",
        "median_edge",
    ]
    for col in pct_cols:
        if col in view.columns:
            view[col] = view[col].map(pct)
    for col in ["min_cum_dollar_volume"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: f"{int(float(x)):,}")
    view = view.fillna("")

    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def load_candidates(path: str | Path) -> pd.DataFrame:
    base_usecols = [
        "date",
        "next_date",
        "year",
        "permno",
        "symbol",
        "market_cap",
        "close",
        "return_1d",
        "dollar_volume",
        "next_high",
        "next_low",
        "next_close",
    ]
    header = pd.read_csv(path, nrows=0).columns
    optional_usecols = []
    for window in PREMARKET_WINDOWS:
        for suffix in ["high_from_t_close", "dollar_volume", "trade_count"]:
            col = f"{window}_{suffix}"
            if col in header:
                optional_usecols.append(col)

    df = pd.read_csv(path, usecols=base_usecols + optional_usecols, parse_dates=["date", "next_date"])
    df = df[
        df["next_date"].notna()
        & df["market_cap"].between(300_000_000, 5_000_000_000, inclusive="both")
        & df["close"].gt(0)
        & df["next_high"].gt(0)
        & df["next_low"].gt(0)
    ].copy()
    df["taq_symbol"] = df["symbol"].map(normalize_taq_symbol)
    df = df[df["taq_symbol"].notna()].copy()

    high_cols = [f"{window}_high_from_t_close" for window in PREMARKET_WINDOWS if f"{window}_high_from_t_close" in df.columns]
    dv_cols = [f"{window}_dollar_volume" for window in PREMARKET_WINDOWS if f"{window}_dollar_volume" in df.columns]
    trade_cols = [f"{window}_trade_count" for window in PREMARKET_WINDOWS if f"{window}_trade_count" in df.columns]

    if high_cols and dv_cols and trade_cols:
        possible_high = df[high_cols].max(axis=1)
        possible_dv = df[dv_cols].fillna(0).sum(axis=1)
        possible_trades = df[trade_cols].fillna(0).sum(axis=1)
        df = df[
            possible_high.ge(min(TRIGGER_FLOORS))
            & possible_dv.ge(min(MIN_DOLLAR_VOLUMES))
            & possible_trades.ge(min(MIN_TRADE_COUNTS))
        ].copy()

    return df


def query_taq_trades_for_date(db, trade_date: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
    year = trade_date.year
    ymd = trade_date.strftime("%Y%m%d")
    schema = f"taqm_{year}"
    table = f"ctm_{ymd}"
    symbol_sql = sql_string_list(symbols)

    query = f"""
        SELECT
            date,
            time_m,
            sym_root,
            sym_suffix,
            size,
            price,
            tr_corr,
            tr_seqnum
        FROM {schema}.{table}
        WHERE time_m >= '04:00:00'
            AND time_m < '09:30:00'
            AND sym_root IN ({symbol_sql})
            AND COALESCE(TRIM(sym_suffix), '') = ''
            AND price > 0
            AND size > 0
            AND (tr_corr IS NULL OR tr_corr = '00')
    """

    return db.raw_sql(query, date_cols=["date"])


def prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    trades = trades.copy()
    trades["time_m"] = pd.to_timedelta(trades["time_m"].astype(str))
    for col in ["price", "size", "tr_seqnum"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")
    trades = trades.dropna(subset=["time_m", "sym_root", "price", "size"])
    trades["dollar_volume"] = trades["price"] * trades["size"]
    return trades.sort_values(["sym_root", "time_m", "tr_seqnum"], na_position="last")


def load_or_query_trades(
    db,
    cache_dir: Path,
    trade_date: pd.Timestamp,
    symbols: list[str],
    use_cache: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"taq_common_0400_0930_{trade_date.strftime('%Y%m%d')}.csv.gz"
    if use_cache and cache_path.exists():
        return prepare_trades(pd.read_csv(cache_path, parse_dates=["date"]))

    trades = query_taq_trades_for_date(db, trade_date, symbols)
    if not trades.empty:
        trades.to_csv(cache_path, index=False, compression="gzip")
    return prepare_trades(trades)


def first_trigger(
    symbol_trades: pd.DataFrame,
    prev_close: float,
    trigger_floor: float,
    min_cum_dollar_volume: float,
    min_cum_trade_count: int,
) -> pd.Series | None:
    if symbol_trades.empty:
        return None

    x = symbol_trades.copy()
    x["cum_dollar_volume"] = x["dollar_volume"].cumsum()
    x["cum_trade_count"] = np.arange(1, len(x) + 1)
    x["trigger_return"] = x["price"] / prev_close - 1
    eligible = x[
        x["trigger_return"].ge(trigger_floor)
        & x["cum_dollar_volume"].ge(min_cum_dollar_volume)
        & x["cum_trade_count"].ge(min_cum_trade_count)
    ]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def simulate_candidate(
    candidate: pd.Series,
    trades: pd.DataFrame,
    trigger_floor: float,
    min_cum_dollar_volume: float,
    min_cum_trade_count: int,
) -> dict[str, object] | None:
    symbol_trades = trades[trades["sym_root"] == candidate["taq_symbol"]]
    trigger = first_trigger(
        symbol_trades,
        float(candidate["close"]),
        trigger_floor,
        min_cum_dollar_volume,
        min_cum_trade_count,
    )
    if trigger is None:
        return None

    trigger_price = float(trigger["price"])
    mfe = float(candidate["next_high"]) / trigger_price - 1
    mae = float(candidate["next_low"]) / trigger_price - 1
    to_close = float(candidate["next_close"]) / trigger_price - 1

    return {
        "date": candidate["date"],
        "trade_date": candidate["next_date"],
        "symbol": candidate["symbol"],
        "permno": candidate["permno"],
        "market_cap": candidate["market_cap"],
        "return_1d": candidate["return_1d"],
        "t_day_dollar_volume": candidate["dollar_volume"],
        "trigger_floor": trigger_floor,
        "min_cum_dollar_volume": min_cum_dollar_volume,
        "min_cum_trade_count": min_cum_trade_count,
        "trigger_time": trigger["time_m"],
        "trigger_price": trigger_price,
        "trigger_return": trigger_price / float(candidate["close"]) - 1,
        "cum_dollar_volume": trigger["cum_dollar_volume"],
        "cum_trade_count": trigger["cum_trade_count"],
        "mfe_to_day_high": mfe,
        "mae_to_day_low": mae,
        "to_close": to_close,
    }


def summarize(results: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    candidate_count = len(candidates[["next_date", "symbol"]].drop_duplicates())
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(
        ["trigger_floor", "min_cum_dollar_volume", "min_cum_trade_count"],
        observed=True,
    ):
        trigger_floor, min_dv, min_trades = keys
        rows.append(
            {
                "trigger_floor": trigger_floor,
                "min_cum_dollar_volume": min_dv,
                "min_cum_trade_count": min_trades,
                "candidate_count": candidate_count,
                "trigger_count": len(group),
                "coverage": len(group) / candidate_count,
                "median_trigger_time": group["trigger_time"].median(),
                "median_trigger_return": group["trigger_return"].median(),
                "avg_mfe_to_day_high": group["mfe_to_day_high"].mean(),
                "median_mfe_to_day_high": group["mfe_to_day_high"].median(),
                "p_mfe_3": group["mfe_to_day_high"].ge(0.03).mean(),
                "p_mfe_5": group["mfe_to_day_high"].ge(0.05).mean(),
                "p_mfe_10": group["mfe_to_day_high"].ge(0.10).mean(),
                "median_mae_to_day_low": group["mae_to_day_low"].median(),
                "p_mae_3": group["mae_to_day_low"].le(-0.03).mean(),
                "p_mae_5": group["mae_to_day_low"].le(-0.05).mean(),
                "trimmed_to_close": trimmed_mean(group["to_close"]),
                "median_to_close": group["to_close"].median(),
                "win_rate_to_close": group["to_close"].gt(0).mean(),
                "median_edge": group["mfe_to_day_high"].median()
                + group["mae_to_day_low"].median(),
            }
        )

    summary = pd.DataFrame(rows)
    summary["rank_score"] = (
        summary["median_mfe_to_day_high"]
        + 0.35 * summary["p_mfe_5"]
        - 0.35 * summary["p_mae_5"]
        + summary["median_edge"]
        + 0.10 * summary["coverage"]
    )
    return summary.sort_values(
        ["rank_score", "trigger_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_periods(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    x = results.copy()
    x["period"] = np.where(x["trade_date"].dt.year <= 2021, "train_2015_2021", "validation_2022_2024")
    rows: list[dict[str, object]] = []
    for keys, group in x.groupby(
        ["trigger_floor", "min_cum_dollar_volume", "min_cum_trade_count", "period"],
        observed=True,
    ):
        trigger_floor, min_dv, min_trades, period = keys
        if len(group) < 20:
            continue
        rows.append(
            {
                "trigger_floor": trigger_floor,
                "min_cum_dollar_volume": min_dv,
                "min_cum_trade_count": min_trades,
                "period": period,
                "trigger_count": len(group),
                "median_trigger_time": group["trigger_time"].median(),
                "median_trigger_return": group["trigger_return"].median(),
                "median_mfe_to_day_high": group["mfe_to_day_high"].median(),
                "p_mfe_3": group["mfe_to_day_high"].ge(0.03).mean(),
                "p_mfe_5": group["mfe_to_day_high"].ge(0.05).mean(),
                "p_mfe_10": group["mfe_to_day_high"].ge(0.10).mean(),
                "median_mae_to_day_low": group["mae_to_day_low"].median(),
                "p_mae_3": group["mae_to_day_low"].le(-0.03).mean(),
                "p_mae_5": group["mae_to_day_low"].le(-0.05).mean(),
                "trimmed_to_close": trimmed_mean(group["to_close"]),
                "median_to_close": group["to_close"].median(),
                "win_rate_to_close": group["to_close"].gt(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, summary: pd.DataFrame, period_summary: pd.DataFrame) -> None:
    lines = [
        "# 完整盘前逐笔动态触发分析",
        "",
        "这份分析使用 04:00-09:30 的逐笔普通股成交，不再用固定窗口近似。",
        "只要任意一笔成交后满足累计涨幅、累计成交额和累计成交笔数阈值，就记录第一次触发。",
        "",
        "## 口径",
        "",
        "- 股票池：市值 300M-5B，T 日涨幅前 50。",
        "- 触发时间：04:00-09:30 任意普通股成交。",
        "- 触发涨幅：相对 T 日收盘价。",
        "- MFE：触发价到当天最高价。",
        "- MAE：触发价到当天最低价。",
        "- 这仍然是信号空间研究，不是最终真实交易回测；下一步要加入触发后的入场、止盈、止损顺序。",
        "",
        "## 规则汇总",
        "",
        markdown_table(
            summary,
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
        lines.append("样本不足，未生成拆分。")
    else:
        lines.append(
            markdown_table(
                period_summary.sort_values(
                    ["trigger_floor", "min_cum_dollar_volume", "min_cum_trade_count", "period"]
                ),
                [
                    "trigger_floor",
                    "min_cum_dollar_volume",
                    "min_cum_trade_count",
                    "period",
                    "trigger_count",
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
                limit=80,
            )
        )

    (output_dir / "full_premarket_dynamic_trigger_report_zh.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(
    candidates_path: str | Path,
    output_dir: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_dates: int | None = None,
    use_cache: bool = True,
    progress_every: int = 50,
) -> Path:
    candidates_path = Path(candidates_path)
    output = Path(output_dir) if output_dir else candidates_path.parent / "full_premarket_dynamic_triggers"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "raw_trade_cache"

    candidates = load_candidates(candidates_path)
    if start_date:
        candidates = candidates[candidates["next_date"] >= pd.Timestamp(start_date)]
    if end_date:
        candidates = candidates[candidates["next_date"] <= pd.Timestamp(end_date)]
    if candidates.empty:
        raise RuntimeError("No candidates found.")

    groups = list(candidates.groupby(candidates["next_date"].dt.date, sort=True))
    if max_dates is not None:
        groups = groups[:max_dates]

    db = connect_wrds(interactive=False)
    rows: list[dict[str, object]] = []

    try:
        for i, (trade_date, daily_candidates) in enumerate(groups, start=1):
            trade_date_ts = pd.Timestamp(trade_date)
            symbols = sorted(daily_candidates["taq_symbol"].dropna().unique())
            if not symbols:
                continue

            should_print = i == 1 or i == len(groups) or i % progress_every == 0
            if should_print:
                print(
                    f"[{i}/{len(groups)}] {trade_date_ts.date()} symbols={len(symbols)}",
                    flush=True,
                )
            try:
                trades = load_or_query_trades(db, cache_dir, trade_date_ts, symbols, use_cache)
            except Exception as exc:
                print(f"[{i}/{len(groups)}] query error, reconnecting: {exc}", flush=True)
                try:
                    db.close()
                except Exception:
                    pass
                try:
                    db = connect_wrds(interactive=False)
                except Exception as reconnect_exc:
                    print(
                        f"[{i}/{len(groups)}] skipped reconnect failure: {reconnect_exc}",
                        flush=True,
                    )
                    continue
                try:
                    trades = load_or_query_trades(db, cache_dir, trade_date_ts, symbols, use_cache)
                except Exception as retry_exc:
                    print(
                        f"[{i}/{len(groups)}] skipped after retry: {retry_exc}",
                        flush=True,
                    )
                    continue

            day_rows = []
            for _, candidate in daily_candidates.iterrows():
                for trigger_floor in TRIGGER_FLOORS:
                    for min_dv in MIN_DOLLAR_VOLUMES:
                        for min_trades in MIN_TRADE_COUNTS:
                            row = simulate_candidate(
                                candidate,
                                trades,
                                trigger_floor,
                                min_dv,
                                min_trades,
                            )
                            if row is not None:
                                day_rows.append(row)
            rows.extend(day_rows)
            if should_print:
                print(
                    f"[{i}/{len(groups)}] candidates={len(daily_candidates)} "
                    f"trades={len(trades)} triggers={len(day_rows)}",
                    flush=True,
                )
    finally:
        try:
            db.close()
        except Exception:
            pass

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No triggers found.")

    summary = summarize(results, candidates)
    period_summary = summarize_periods(results)

    results.to_csv(output / "full_premarket_dynamic_trigger_results.csv", index=False)
    summary.to_csv(output / "full_premarket_dynamic_trigger_summary.csv", index=False)
    period_summary.to_csv(output / "full_premarket_dynamic_trigger_period_summary.csv", index=False)
    write_report(output, summary, period_summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    output = run(
        args.input,
        args.output_dir,
        args.start_date,
        args.end_date,
        args.max_dates,
        use_cache=not args.no_cache,
        progress_every=args.progress_every,
    )
    print(f"Saved full premarket dynamic trigger analysis to: {output}")


if __name__ == "__main__":
    main()
