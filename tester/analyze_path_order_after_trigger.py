from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("screen_results/20150102_20241231_gainers_return_1d_top50/premarket")
DEFAULT_TRIGGER_DIR = ROOT / "full_premarket_dynamic_triggers"
DEFAULT_OUTPUT_DIR = ROOT / "path_order_after_trigger"

TAKE_PROFITS = [0.03, 0.05]
STOP_LOSSES = [-0.02, -0.03, -0.05]


def parse_time_delta(value: object) -> pd.Timedelta:
    text = str(value)
    if text.startswith("0 days "):
        text = text.replace("0 days ", "", 1)
    return pd.to_timedelta(text)


def prepare_trades(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    usecols = [col for col in ["date", "time_m", "sym_root", "price", "tr_seqnum"] if col in header]
    trades = pd.read_csv(path, usecols=usecols, parse_dates=["date"])
    if trades.empty:
        return trades
    trades = trades.copy()
    trades["time_m"] = pd.to_timedelta(trades["time_m"].astype(str))
    for col in ["price", "size", "tr_seqnum"]:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")
    trades = trades.dropna(subset=["sym_root", "time_m", "price"])
    sort_cols = ["sym_root", "time_m"]
    if "tr_seqnum" in trades.columns:
        sort_cols.append("tr_seqnum")
    return trades.sort_values(sort_cols).reset_index(drop=True)


def first_hit(path: pd.DataFrame, entry_price: float, threshold: float, direction: str) -> tuple[pd.Timedelta | None, float | None]:
    returns = path["price"] / entry_price - 1
    if direction == "up":
        hit = path[returns.ge(threshold)]
    else:
        hit = path[returns.le(threshold)]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return row["time_m"], float(row["price"]) / entry_price - 1


def simulate_event(event: pd.Series, trades_by_symbol: dict[str, pd.DataFrame]) -> dict[str, object] | None:
    symbol_trades = trades_by_symbol.get(str(event["symbol"]))
    if symbol_trades.empty:
        return None

    trigger_time = parse_time_delta(event["trigger_time"])
    after_trigger = symbol_trades[symbol_trades["time_m"].gt(trigger_time)].copy()
    if after_trigger.empty:
        return None

    entry = after_trigger.iloc[0]
    entry_time = entry["time_m"]
    entry_price = float(entry["price"])
    path = after_trigger[after_trigger["time_m"].ge(entry_time)].copy()
    if path.empty or entry_price <= 0:
        return None

    returns = path["price"] / entry_price - 1
    max_idx = returns.idxmax()
    min_idx = returns.idxmin()
    mfe = float(returns.loc[max_idx])
    mae = float(returns.loc[min_idx])
    mfe_time = path.loc[max_idx, "time_m"]
    mae_time = path.loc[min_idx, "time_m"]

    row: dict[str, object] = {
        "date": event["date"],
        "trade_date": event["trade_date"],
        "symbol": event["symbol"],
        "market_cap": event["market_cap"],
        "return_1d": event["return_1d"],
        "trigger_time": trigger_time,
        "trigger_price": event["trigger_price"],
        "entry_time": entry_time,
        "entry_price": entry_price,
        "entry_slippage_from_trigger": entry_price / float(event["trigger_price"]) - 1,
        "path_trade_count": len(path),
        "last_time": path["time_m"].iloc[-1],
        "last_return": float(path["price"].iloc[-1]) / entry_price - 1,
        "mfe_to_0930": mfe,
        "mfe_time": mfe_time,
        "mae_to_0930": mae,
        "mae_time": mae_time,
        "mfe_before_mae": mfe_time <= mae_time,
        "mae_before_mfe": mae_time < mfe_time,
    }

    for tp in TAKE_PROFITS:
        tp_time, tp_ret = first_hit(path, entry_price, tp, "up")
        row[f"hit_tp_{int(tp * 100)}"] = tp_time is not None
        row[f"tp_{int(tp * 100)}_time"] = tp_time
        row[f"tp_{int(tp * 100)}_return"] = tp_ret
        for sl in STOP_LOSSES:
            sl_abs = int(abs(sl) * 100)
            sl_time, sl_ret = first_hit(path, entry_price, sl, "down")
            row[f"hit_sl_{sl_abs}"] = sl_time is not None
            row[f"sl_{sl_abs}_time"] = sl_time
            row[f"sl_{sl_abs}_return"] = sl_ret
            prefix = f"tp{int(tp * 100)}_vs_sl{sl_abs}"
            if tp_time is None and sl_time is None:
                outcome = "neither"
            elif tp_time is not None and (sl_time is None or tp_time <= sl_time):
                outcome = "tp_first"
            else:
                outcome = "sl_first"
            row[f"{prefix}_outcome"] = outcome

    return row


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = {
        "all": events,
        "train_2015_2021": events[pd.to_datetime(events["trade_date"]).dt.year <= 2021],
        "validation_2022_2024": events[pd.to_datetime(events["trade_date"]).dt.year >= 2022],
    }
    for period, group in groups.items():
        if group.empty:
            continue
        base = {
            "period": period,
            "event_count": len(group),
            "median_entry_time": group["entry_time"].median(),
            "median_entry_slippage_from_trigger": group["entry_slippage_from_trigger"].median(),
            "median_mfe_to_0930": group["mfe_to_0930"].median(),
            "median_mae_to_0930": group["mae_to_0930"].median(),
            "mfe_before_mae_rate": group["mfe_before_mae"].mean(),
            "mae_before_mfe_rate": group["mae_before_mfe"].mean(),
            "p_mfe_3_to_0930": group["mfe_to_0930"].ge(0.03).mean(),
            "p_mfe_5_to_0930": group["mfe_to_0930"].ge(0.05).mean(),
            "p_mae_2_to_0930": group["mae_to_0930"].le(-0.02).mean(),
            "p_mae_3_to_0930": group["mae_to_0930"].le(-0.03).mean(),
            "p_mae_5_to_0930": group["mae_to_0930"].le(-0.05).mean(),
            "median_last_return_to_0930": group["last_return"].median(),
        }
        for tp in TAKE_PROFITS:
            for sl in STOP_LOSSES:
                sl_abs = int(abs(sl) * 100)
                col = f"tp{int(tp * 100)}_vs_sl{sl_abs}_outcome"
                counts = group[col].value_counts(normalize=True)
                base[f"p_tp{int(tp * 100)}_before_sl{sl_abs}"] = counts.get("tp_first", 0.0)
                base[f"p_sl{sl_abs}_before_tp{int(tp * 100)}"] = counts.get("sl_first", 0.0)
                base[f"p_neither_tp{int(tp * 100)}_sl{sl_abs}"] = counts.get("neither", 0.0)
        rows.append(base)
    return pd.DataFrame(rows)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:+.2%}" if value < 0 else f"{value:.2%}"


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    report = [
        "# 触发后路径顺序测试",
        "",
        "口径：T 日涨幅前 50，市值 300M-5B，盘前 04:00-09:30 动态触发，基准条件为相对 T 日收盘价上涨 >=5%，累计盘前成交额 >=1M，累计成交笔数 >=100。",
        "",
        "入场假设：触发后下一笔普通股成交买入。路径扫描范围：本地逐笔缓存可覆盖的触发后到 09:30 之前。注意，这不是全天路径测试，09:30 之后的最高/最低先后顺序需要额外拉取全天 TAQ 才能严谨判断。",
        "",
        "## 核心结果",
        "",
    ]
    view = summary.copy()
    for col in view.columns:
        if col.startswith(("median_", "p_", "mfe_", "mae_")) or col.endswith("_rate"):
            if col not in {"median_entry_time"}:
                view[col] = view[col].map(format_pct)
    view["median_entry_time"] = view["median_entry_time"].astype(str).str.replace("0 days ", "", regex=False).str.slice(0, 8)
    headers = list(view.columns)
    report.append("| " + " | ".join(headers) + " |")
    report.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in view.iterrows():
        report.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    report.extend(
        [
            "",
            "## 解读",
            "",
            "- `p_tp*_before_sl*` 是真实路径顺序指标：从入场后逐笔扫描，先到止盈还是先到止损。",
            "- `mfe_before_mae_rate` 表示盘前剩余路径里，最高价先于最低价出现的比例。",
            "- 如果 `p_sl2_before_tp3` 明显高于 `p_tp3_before_sl2`，说明用 -2% 止损、+3% 第一目标时，很多交易会先被止损，后面的 MFE 不能算作可交易收益。",
        ]
    )
    (output_dir / "path_order_after_trigger_report_zh.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether post-trigger upside or downside happens first.")
    parser.add_argument("--trigger-dir", type=Path, default=DEFAULT_TRIGGER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trigger-floor", type=float, default=0.05)
    parser.add_argument("--min-dollar-volume", type=float, default=1_000_000)
    parser.add_argument("--min-trade-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.trigger_dir / "full_premarket_dynamic_trigger_results.csv"
    triggers = pd.read_csv(results_path, parse_dates=["date", "trade_date"])
    triggers = triggers[
        triggers["trigger_floor"].eq(args.trigger_floor)
        & triggers["min_cum_dollar_volume"].eq(args.min_dollar_volume)
        & triggers["min_cum_trade_count"].eq(args.min_trade_count)
    ].copy()

    cache_dir = args.trigger_dir / "raw_trade_cache"
    events: list[dict[str, object]] = []
    grouped = list(triggers.groupby("trade_date"))
    for i, (trade_date, day_events) in enumerate(grouped, start=1):
        if i == 1 or i % 250 == 0:
            print(f"processing {i}/{len(grouped)} {pd.Timestamp(trade_date).date()}", flush=True)
        cache_path = cache_dir / f"taq_common_0400_0930_{pd.Timestamp(trade_date).strftime('%Y%m%d')}.csv.gz"
        if not cache_path.exists():
            continue
        trades = prepare_trades(cache_path)
        trades_by_symbol = {symbol: group.reset_index(drop=True) for symbol, group in trades.groupby("sym_root", sort=False)}
        for _, event in day_events.iterrows():
            simulated = simulate_event(event, trades_by_symbol)
            if simulated is not None:
                events.append(simulated)

    event_df = pd.DataFrame(events)
    event_df.to_csv(args.output_dir / "path_order_after_trigger_events.csv", index=False)
    summary = summarize(event_df)
    summary.to_csv(args.output_dir / "path_order_after_trigger_summary.csv", index=False)
    write_report(summary, args.output_dir)
    print(f"events={len(event_df)}")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
