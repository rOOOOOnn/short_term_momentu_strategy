# preprocess.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wrds


def connect_wrds() -> wrds.Connection:
    """
    连接 WRDS。
    第一次运行可能会要求输入 WRDS 用户名和密码。
    """
    return wrds.Connection()


def download_crsp_daily(
    start_date: str,
    end_date: str,
    output_raw_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    从 WRDS CRSP 拉取日线数据。

    默认使用：
    - crsp.dsf: Daily Stock File
    - crsp.dsenames: stock names / share codes / exchange codes

    shrcd:
        10, 11 通常代表普通股 common shares
    exchcd:
        1 NYSE
        2 AMEX
        3 NASDAQ
    """

    db = connect_wrds()

    query = f"""
        SELECT
            a.permno,
            a.date,
            a.prc,
            a.openprc,
            a.askhi,
            a.bidlo,
            a.vol,
            a.ret,
            a.shrout,

            b.ticker,
            b.comnam,
            b.shrcd,
            b.exchcd,
            b.siccd

        FROM crsp.dsf AS a

        LEFT JOIN crsp.dsenames AS b
            ON a.permno = b.permno
            AND a.date >= b.namedt
            AND a.date <= b.nameendt

        WHERE a.date >= '{start_date}'
            AND a.date <= '{end_date}'
            AND b.shrcd IN (10, 11)
            AND b.exchcd IN (1, 2, 3)
    """

    df = db.raw_sql(query, date_cols=["date"])

    if output_raw_path is not None:
        output_raw_path = Path(output_raw_path)
        output_raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_raw_path, index=False)

    db.close()
    return df


def clean_crsp_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 CRSP 日线数据，并统一字段名。
    """

    df = df.copy()

    # 统一列名。
    df = df.rename(
        columns={
            "permno": "permno",
            "date": "date",
            "ticker": "symbol",
            "comnam": "company_name",
            "prc": "close",
            "openprc": "open",
            "askhi": "high",
            "bidlo": "low",
            "vol": "volume",
            "ret": "crsp_return",
            "shrout": "shares_outstanding",
            "shrcd": "share_code",
            "exchcd": "exchange_code",
            "siccd": "sic_code",
        }
    )

    df["date"] = pd.to_datetime(df["date"])

    # CRSP prc 可能为负，取绝对值。
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["shares_outstanding"] = pd.to_numeric(df["shares_outstanding"], errors="coerce")

    # ret 里面可能有字符型缺失，例如 C/B 之类，统一转 numeric。
    df["crsp_return"] = pd.to_numeric(df["crsp_return"], errors="coerce")

    # 基础清洗。
    df = df.dropna(subset=["date", "permno", "close", "volume"])
    df = df[df["close"] > 0]
    df = df[df["volume"] > 0]

    # 如果 high/low/open 缺失，先保留 close 和 volume，但后续 CLV 会是 NaN。
    df = df.sort_values(["permno", "date"])

    # 市值：CRSP shrout 一般是千股。
    df["market_cap"] = df["close"] * df["shares_outstanding"] * 1000

    # 成交额。
    df["dollar_volume"] = df["close"] * df["volume"]

    # 交易所名称。
    exchange_map = {
        1: "NYSE",
        2: "AMEX",
        3: "NASDAQ",
    }
    df["exchange"] = df["exchange_code"].map(exchange_map)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 T 日已知的特征。
    """

    df = df.copy()
    df = df.sort_values(["permno", "date"])

    group = df.groupby("permno", group_keys=False)

    # 前一日收盘。
    df["prev_close"] = group["close"].shift(1)

    # 自己计算日收益，避免 crsp_return 类型/缺失问题。
    df["return_1d"] = df["close"] / df["prev_close"] - 1

    # 20 日平均成交量和成交额。
    df["avg_volume_20d"] = group["volume"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )

    df["avg_dollar_volume_20d"] = group["dollar_volume"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )

    # 相对成交量。
    df["relative_volume_20d"] = df["volume"] / df["avg_volume_20d"]

    # CLV: close location value，表示收盘价在当天 high-low 区间的位置。
    price_range = df["high"] - df["low"]

    df["clv"] = np.where(
        price_range > 0,
        (df["close"] - df["low"]) / price_range,
        np.nan,
    )

    # 上影线比例。
    df["upper_shadow_ratio"] = np.where(
        price_range > 0,
        (df["high"] - df["close"]) / price_range,
        np.nan,
    )

    # 下影线比例。
    df["lower_shadow_ratio"] = np.where(
        price_range > 0,
        (df["close"] - df["low"]) / price_range,
        np.nan,
    )

    return df


def add_next_day_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 T+1 的结果变量。
    注意：这些变量只能用于评价，不能用于筛选。
    """

    df = df.copy()
    df = df.sort_values(["permno", "date"])

    group = df.groupby("permno", group_keys=False)

    df["next_open"] = group["open"].shift(-1)
    df["next_high"] = group["high"].shift(-1)
    df["next_low"] = group["low"].shift(-1)
    df["next_close"] = group["close"].shift(-1)

    df["next_open_return"] = df["next_open"] / df["close"] - 1
    df["next_high_return"] = df["next_high"] / df["close"] - 1
    df["next_low_return"] = df["next_low"] / df["close"] - 1
    df["next_close_return"] = df["next_close"] / df["close"] - 1

    df["next_intraday_return"] = df["next_close"] / df["next_open"] - 1

    return df


def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    最终清理。
    """

    df = df.copy()

    # 去除明显异常收益。
    # 第一阶段先保守一点，避免脏数据污染统计。
    df = df[df["return_1d"].between(-0.95, 5.0, inclusive="both") | df["return_1d"].isna()]

    # 关键特征缺失的行可以保留，但筛选时会再过滤。
    df = df.sort_values(["date", "permno"])

    return df


def preprocess_wrds_crsp(
    start_date: str,
    end_date: str,
    output_path: str | Path,
    raw_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    完整预处理流程。
    """

    print(f"Downloading CRSP daily data from WRDS: {start_date} to {end_date}")

    df = download_crsp_daily(
        start_date=start_date,
        end_date=end_date,
        output_raw_path=raw_output_path,
    )

    print(f"Raw rows: {len(df):,}")

    df = clean_crsp_daily(df)
    df = add_features(df)
    df = add_next_day_returns(df)
    df = final_clean(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    print(f"Processed rows: {len(df):,}")
    print(f"Saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=str, required=True, help="Start date, e.g. 2020-01-01")
    parser.add_argument("--end", type=str, required=True, help="End date, e.g. 2025-12-31")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/crsp_daily_features.parquet",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="Optional path to save raw WRDS data.",
    )

    args = parser.parse_args()

    preprocess_wrds_crsp(
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        raw_output_path=args.raw_output,
    )


if __name__ == "__main__":
    main()
