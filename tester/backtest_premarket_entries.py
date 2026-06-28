# backtest_premarket_entries.py

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.wrds.preprocess import connect_wrds
from data.wrds.preprocess_premarket import normalize_taq_symbol, sql_string_list


@dataclass(frozen=True)
class EntryRule:
    monitor_start: str
    monitor_end: str
    trigger_return: float
    min_cum_dollar_volume: float
    min_cum_trades: int
    take_profit: float
    stop_loss: float
    exit_time: str

    @property
    def name(self) -> str:
        return (
            f"start{self.monitor_start.replace(':', '')}"
            f"_trig{int(self.trigger_return * 1000):03d}"
            f"_dv{int(self.min_cum_dollar_volume / 1000)}k"
            f"_tr{self.min_cum_trades}"
            f"_tp{int(self.take_profit * 1000):03d}"
            f"_sl{int(self.stop_loss * 1000):03d}"
            f"_exit{self.exit_time.replace(':', '')}"
        )


def load_candidates(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "next_date"])
    df["taq_symbol"] = df["symbol"].map(normalize_taq_symbol)

    df = df[df["next_date"].notna()]
    df = df[df["taq_symbol"].notna()]
    df = df[df["market_cap_bucket"].isin(["300M-1B", "1B-5B"])]

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
            size,
            price,
            tr_corr,
            tr_seqnum
        FROM {schema}.{table}
        WHERE time_m >= '04:00:00'
            AND time_m < '10:00:00'
            AND sym_root IN ({symbol_sql})
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
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["size"] = pd.to_numeric(trades["size"], errors="coerce")
    trades["tr_seqnum"] = pd.to_numeric(trades["tr_seqnum"], errors="coerce")
    trades = trades.dropna(subset=["time_m", "sym_root", "price", "size"])
    trades["dollar_volume"] = trades["price"] * trades["size"]
    trades = trades.sort_values(["sym_root", "time_m", "tr_seqnum"], na_position="last")

    return trades


def load_or_query_daily_trades(
    db,
    cache_dir: Path,
    trade_date: pd.Timestamp,
    symbols: list[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"taq_trades_{trade_date.strftime('%Y%m%d')}.csv.gz"

    if use_cache and cache_path.exists():
        trades = pd.read_csv(cache_path, parse_dates=["date"])
        return prepare_trades(trades)

    trades = query_taq_trades_for_date(db, trade_date, symbols)
    if not trades.empty:
        trades.to_csv(cache_path, index=False, compression="gzip")

    return prepare_trades(trades)


def build_default_rules() -> list[EntryRule]:
    rules = []

    monitor_starts = ["04:00:00", "07:00:00", "08:00:00", "09:00:00", "09:15:00"]
    trigger_returns = [0.02, 0.03, 0.05]
    min_dollar_volumes = [250_000, 500_000, 1_000_000]
    min_trades_values = [25, 50, 100]
    take_profits = [0.03, 0.05, 0.08]
    stop_losses = [0.015, 0.02, 0.03]
    exit_times = ["09:35:00", "10:00:00"]

    for monitor_start in monitor_starts:
        for trigger_return in trigger_returns:
            for min_dollar_volume in min_dollar_volumes:
                for min_trades in min_trades_values:
                    for take_profit in take_profits:
                        for stop_loss in stop_losses:
                            for exit_time in exit_times:
                                rules.append(
                                    EntryRule(
                                        monitor_start=monitor_start,
                                        monitor_end="09:30:00",
                                        trigger_return=trigger_return,
                                        min_cum_dollar_volume=min_dollar_volume,
                                        min_cum_trades=min_trades,
                                        take_profit=take_profit,
                                        stop_loss=stop_loss,
                                        exit_time=exit_time,
                                    )
                                )

    return rules


def build_focused_rules() -> list[EntryRule]:
    rules = []

    for monitor_start in ["04:00:00", "07:00:00", "08:00:00", "09:00:00", "09:15:00"]:
        for trigger_return in [0.02, 0.03]:
            for min_dollar_volume in [250_000, 500_000, 1_000_000]:
                for exit_time in ["09:35:00", "10:00:00"]:
                    rules.append(
                        EntryRule(
                            monitor_start=monitor_start,
                            monitor_end="09:30:00",
                            trigger_return=trigger_return,
                            min_cum_dollar_volume=min_dollar_volume,
                            min_cum_trades=1,
                            take_profit=0.05,
                            stop_loss=0.02,
                            exit_time=exit_time,
                        )
                    )

    return rules


def find_entry(trades: pd.DataFrame, prev_close: float, rule: EntryRule) -> tuple[int | None, float | None, pd.Timedelta | None]:
    start = pd.to_timedelta(rule.monitor_start)
    end = pd.to_timedelta(rule.monitor_end)
    window = trades[(trades["time_m"] >= start) & (trades["time_m"] < end)].copy()

    if window.empty or not np.isfinite(prev_close) or prev_close <= 0:
        return None, None, None

    window["cum_dollar_volume"] = window["dollar_volume"].cumsum()
    window["cum_trades"] = np.arange(1, len(window) + 1)

    trigger_price = prev_close * (1 + rule.trigger_return)
    eligible = window[
        (window["price"] >= trigger_price)
        & (window["cum_dollar_volume"] >= rule.min_cum_dollar_volume)
        & (window["cum_trades"] >= rule.min_cum_trades)
    ]

    if eligible.empty:
        return None, None, None

    trigger_idx = eligible.index[0]
    next_trades = trades[trades.index > trigger_idx]
    if next_trades.empty:
        return None, None, None

    entry_idx = next_trades.index[0]
    entry_price = float(next_trades.iloc[0]["price"])
    entry_time = next_trades.iloc[0]["time_m"]

    return entry_idx, entry_price, entry_time


def simulate_exit(
    trades: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    rule: EntryRule,
    slippage_bps: float,
) -> dict[str, object] | None:
    entry_fill = entry_price * (1 + slippage_bps / 10_000)
    take_profit_price = entry_fill * (1 + rule.take_profit)
    stop_loss_price = entry_fill * (1 - rule.stop_loss)
    exit_end = pd.to_timedelta(rule.exit_time)

    after_entry = trades[(trades.index > entry_idx) & (trades["time_m"] <= exit_end)]
    if after_entry.empty:
        return None

    for _, trade in after_entry.iterrows():
        price = float(trade["price"])
        if price >= take_profit_price:
            exit_fill = take_profit_price * (1 - slippage_bps / 10_000)
            return {
                "exit_reason": "take_profit",
                "exit_time": trade["time_m"],
                "exit_price": exit_fill,
                "return": exit_fill / entry_fill - 1,
            }

        if price <= stop_loss_price:
            exit_fill = stop_loss_price * (1 - slippage_bps / 10_000)
            return {
                "exit_reason": "stop_loss",
                "exit_time": trade["time_m"],
                "exit_price": exit_fill,
                "return": exit_fill / entry_fill - 1,
            }

    last_trade = after_entry.iloc[-1]
    exit_fill = float(last_trade["price"]) * (1 - slippage_bps / 10_000)
    return {
        "exit_reason": "timeout",
        "exit_time": last_trade["time_m"],
        "exit_price": exit_fill,
        "return": exit_fill / entry_fill - 1,
    }


def simulate_candidate_rules(
    candidate: pd.Series,
    trades: pd.DataFrame,
    rules: list[EntryRule],
    slippage_bps: float,
) -> list[dict[str, object]]:
    symbol_trades = trades[trades["sym_root"] == candidate["taq_symbol"]].copy()
    if symbol_trades.empty:
        return []

    rows = []
    prev_close = float(candidate["close"])

    for rule in rules:
        entry_idx, entry_price, entry_time = find_entry(symbol_trades, prev_close, rule)
        if entry_idx is None or entry_price is None or entry_time is None:
            continue

        exit_result = simulate_exit(symbol_trades, entry_idx, entry_price, rule, slippage_bps)
        if exit_result is None:
            continue

        rows.append(
            {
                "date": candidate["date"],
                "trade_date": candidate["next_date"],
                "symbol": candidate["symbol"],
                "taq_symbol": candidate["taq_symbol"],
                "permno": candidate["permno"],
                "market_cap_bucket": candidate["market_cap_bucket"],
                "dollar_volume_bucket": candidate["dollar_volume_bucket"],
                "return_bucket": candidate["return_bucket"],
                "relative_volume_bucket": candidate["relative_volume_bucket"],
                "close": candidate["close"],
                "return_1d": candidate["return_1d"],
                "market_cap": candidate["market_cap"],
                "dollar_volume": candidate["dollar_volume"],
                "rule": rule.name,
                "monitor_start": rule.monitor_start,
                "trigger_return": rule.trigger_return,
                "min_cum_dollar_volume": rule.min_cum_dollar_volume,
                "min_cum_trades": rule.min_cum_trades,
                "take_profit": rule.take_profit,
                "stop_loss": rule.stop_loss,
                "exit_time_limit": rule.exit_time,
                "entry_time": entry_time,
                "entry_price": entry_price,
                **exit_result,
            }
        )

    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    summary = (
        results.groupby(
            [
                "monitor_start",
                "trigger_return",
                "min_cum_dollar_volume",
                "min_cum_trades",
                "take_profit",
                "stop_loss",
                "exit_time_limit",
            ],
            observed=True,
        )
        .agg(
            trade_count=("return", "count"),
            avg_return=("return", "mean"),
            median_return=("return", "median"),
            win_rate=("return", lambda x: (x > 0).mean()),
            gross_profit=("return", lambda x: x[x > 0].sum()),
            gross_loss=("return", lambda x: -x[x < 0].sum()),
            take_profit_rate=("exit_reason", lambda x: (x == "take_profit").mean()),
            stop_loss_rate=("exit_reason", lambda x: (x == "stop_loss").mean()),
            timeout_rate=("exit_reason", lambda x: (x == "timeout").mean()),
        )
        .reset_index()
    )
    summary["profit_factor"] = summary["gross_profit"] / summary["gross_loss"].replace(0, np.nan)
    summary = summary.sort_values(["avg_return", "profit_factor", "trade_count"], ascending=[False, False, False])

    return summary


def run_backtest(
    candidates_path: str | Path,
    output_dir: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_dates: int | None = None,
    max_symbols_per_date: int | None = None,
    slippage_bps: float = 10.0,
    use_cache: bool = True,
    focused_rules: bool = False,
) -> pd.DataFrame:
    candidates_path = Path(candidates_path)
    candidates = load_candidates(candidates_path)

    if candidates.empty:
        raise ValueError("No 300M-5B candidates found.")

    if start_date is not None:
        candidates = candidates[candidates["next_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        candidates = candidates[candidates["next_date"] <= pd.Timestamp(end_date)]
    if candidates.empty:
        raise ValueError("No candidates found in the requested date range.")

    output_dir = Path(output_dir) if output_dir else candidates_path.parent / "entry_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "raw_trade_cache"

    rules = build_focused_rules() if focused_rules else build_default_rules()
    grouped = list(candidates.groupby(candidates["next_date"].dt.date, sort=True))
    if max_dates is not None:
        grouped = grouped[:max_dates]

    db = connect_wrds()
    rows = []

    try:
        for i, (trade_date, daily_candidates) in enumerate(grouped, start=1):
            trade_date_ts = pd.Timestamp(trade_date)
            if max_symbols_per_date is not None:
                daily_candidates = daily_candidates.head(max_symbols_per_date)

            symbols = sorted(daily_candidates["taq_symbol"].dropna().unique().tolist())
            if not symbols:
                continue

            try:
                print(
                    f"[{i}/{len(grouped)}] querying {trade_date_ts.date()} "
                    f"symbols={len(symbols):,}",
                    flush=True,
                )
                trades = load_or_query_daily_trades(
                    db=db,
                    cache_dir=cache_dir,
                    trade_date=trade_date_ts,
                    symbols=symbols,
                    use_cache=use_cache,
                )
            except Exception as exc:
                print(f"[{i}/{len(grouped)}] skipped {trade_date_ts.date()} due to query error: {exc}")
                continue

            day_rows = []
            for _, candidate in daily_candidates.iterrows():
                day_rows.extend(simulate_candidate_rules(candidate, trades, rules, slippage_bps))

            rows.extend(day_rows)
            print(
                f"[{i}/{len(grouped)}] {trade_date_ts.date()} "
                f"candidates={len(daily_candidates):,} trades={len(trades):,} fills={len(day_rows):,}",
                flush=True,
            )
    finally:
        db.close()

    results = pd.DataFrame(rows)
    results_path = output_dir / "entry_backtest_trades.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved trade-level results to: {results_path}")

    summary = summarize_results(results)
    summary_path = output_dir / "entry_backtest_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True, help="Path to all_screen_results.csv.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None, help="Filter by T+1 trade date.")
    parser.add_argument("--end-date", type=str, default=None, help="Filter by T+1 trade date.")
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--max-symbols-per-date", type=int, default=None)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--focused-rules",
        action="store_true",
        help="Only run +2/+3 triggers, 250K/500K/1M volume, TP 5%%, SL 2%%, exit 09:35/10:00.",
    )
    args = parser.parse_args()

    run_backtest(
        candidates_path=args.candidates,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_dates=args.max_dates,
        max_symbols_per_date=args.max_symbols_per_date,
        slippage_bps=args.slippage_bps,
        use_cache=not args.no_cache,
        focused_rules=args.focused_rules,
    )


if __name__ == "__main__":
    main()
