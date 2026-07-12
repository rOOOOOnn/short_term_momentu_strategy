from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DEFAULT_TRIGGER_DIR, LAB_ROOT, ensure_dir, load_baseline_triggers, prepare_trades

from data.wrds.preprocess import connect_wrds
from data.wrds.preprocess_premarket import sql_string_list


DEFAULT_CACHE_DIR = LAB_ROOT / "data" / "taq_0400_1100_cache"


def query_taq_trades(
    db,
    trade_date: pd.Timestamp,
    symbols: list[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    schema = f"taqm_{trade_date.year}"
    table = f"ctm_{trade_date.strftime('%Y%m%d')}"
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
        WHERE time_m >= '{start_time}'
            AND time_m < '{end_time}'
            AND sym_root IN ({sql_string_list(symbols)})
            AND COALESCE(TRIM(sym_suffix), '') = ''
            AND price > 0
            AND size > 0
            AND (tr_corr IS NULL OR tr_corr = '00')
    """
    return db.raw_sql(query, date_cols=["date"])


def cache_path(cache_dir: Path, trade_date: pd.Timestamp) -> Path:
    return cache_dir / f"taq_common_0400_1100_{trade_date.strftime('%Y%m%d')}.csv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache TAQ common-share trades from 04:00 to 11:00.")
    parser.add_argument("--trigger-dir", type=Path, default=DEFAULT_TRIGGER_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Allow WRDS to prompt for credentials.")
    parser.add_argument("--start-time", default="04:00:00")
    parser.add_argument("--end-time", default="11:00:00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = ensure_dir(args.cache_dir)
    triggers = load_baseline_triggers(args.trigger_dir)
    if args.start_date:
        triggers = triggers[triggers["trade_date"].ge(pd.Timestamp(args.start_date))]
    if args.end_date:
        triggers = triggers[triggers["trade_date"].le(pd.Timestamp(args.end_date))]

    grouped = list(triggers.groupby("trade_date", sort=True))
    if args.max_dates:
        grouped = grouped[: args.max_dates]

    db = connect_wrds(interactive=args.interactive)
    try:
        for index, (trade_date, day_events) in enumerate(grouped, start=1):
            trade_date = pd.Timestamp(trade_date)
            path = cache_path(cache_dir, trade_date)
            if path.exists() and not args.refresh:
                if index == 1 or index % 50 == 0 or index == len(grouped):
                    print(f"[{index}/{len(grouped)}] cached {trade_date.date()} {path.name}", flush=True)
                continue

            symbols = sorted(day_events["symbol"].astype(str).str.upper().unique().tolist())
            trades = None
            for attempt in range(1, 4):
                try:
                    raw = query_taq_trades(db, trade_date, symbols, args.start_time, args.end_time)
                    trades = prepare_trades(raw)
                    break
                except Exception as exc:
                    print(
                        f"[{index}/{len(grouped)}] {trade_date.date()} attempt {attempt}/3 failed: {exc}",
                        flush=True,
                    )
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = connect_wrds(interactive=args.interactive)
            if trades is None:
                print(f"[{index}/{len(grouped)}] skip {trade_date.date()} after 3 attempts", flush=True)
                continue

            trades.to_csv(path, index=False, compression="gzip")
            print(
                f"[{index}/{len(grouped)}] saved {trade_date.date()} symbols={len(symbols)} "
                f"trades={len(trades):,}",
                flush=True,
            )
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
