from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.wrds.preprocess import connect_wrds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate WRDS TAQ access and profile one symbol-day without downloading a full table."
    )
    parser.add_argument("--date", required=True, help="Trading date in YYYY-MM-DD format.")
    parser.add_argument("--symbol", required=True, help="TAQ root symbol, for example PDYN.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trade_date = datetime.strptime(args.date, "%Y-%m-%d")
    symbol = args.symbol.strip().upper()
    schema = f"taqm_{trade_date.year}"
    table = f"ctm_{trade_date.strftime('%Y%m%d')}"

    db = connect_wrds(interactive=False)
    try:
        current_user = db.raw_sql("SELECT current_user AS username")
        print("CURRENT USER")
        print(current_user.to_string(index=False))

        columns = db.raw_sql(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %(schema)s
              AND table_name = %(table)s
            ORDER BY ordinal_position
            """,
            params={"schema": schema, "table": table},
        )
        if columns.empty:
            raise RuntimeError(f"Table {schema}.{table} is not visible to the current account.")

        print(f"\nTABLE {schema}.{table}")
        print(columns.to_string(index=False))

        profile_sql = f"""
            SELECT
                COUNT(*) AS symbol_rows,
                COUNT(*) FILTER (
                    WHERE COALESCE(TRIM(sym_suffix), '') = ''
                      AND price > 0
                      AND size > 0
                      AND (tr_corr IS NULL OR tr_corr = '00')
                ) AS usable_rows,
                COUNT(*) FILTER (WHERE time_m < '04:00:00') AS before_0400_rows,
                COUNT(*) FILTER (
                    WHERE time_m >= '04:00:00' AND time_m < '09:30:00'
                ) AS premarket_rows,
                COUNT(*) FILTER (
                    WHERE time_m >= '09:30:00' AND time_m < '16:00:00'
                ) AS regular_rows,
                COUNT(*) FILTER (WHERE time_m >= '16:00:00') AS after_1600_rows,
                COUNT(*) FILTER (WHERE price IS NULL OR price <= 0) AS invalid_price_rows,
                COUNT(*) FILTER (WHERE size IS NULL OR size <= 0) AS invalid_size_rows,
                COUNT(*) FILTER (
                    WHERE tr_corr IS NOT NULL AND tr_corr <> '00'
                ) AS corrected_trade_rows,
                MIN(time_m) AS min_time,
                MAX(time_m) AS max_time,
                MIN(price) FILTER (WHERE price > 0) AS min_positive_price,
                MAX(price) FILTER (WHERE price > 0) AS max_positive_price
            FROM {schema}.{table}
            WHERE sym_root = %(symbol)s
        """
        profile = db.raw_sql(profile_sql, params={"symbol": symbol})
        print(f"\nPROFILE {symbol}")
        print(profile.to_string(index=False))

        sample_sql = f"""
            SELECT date, time_m, sym_root, sym_suffix, size, price, tr_corr, tr_seqnum
            FROM {schema}.{table}
            WHERE sym_root = %(symbol)s
              AND time_m >= '09:29:59'
              AND time_m < '09:30:02'
            ORDER BY time_m, tr_seqnum
            LIMIT 10
        """
        sample = db.raw_sql(sample_sql, params={"symbol": symbol}, date_cols=["date"])
        print("\nOPEN SAMPLE")
        print(sample.to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
