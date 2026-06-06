# filters.py

from __future__ import annotations

import pandas as pd


def base_universe_filter(
    df: pd.DataFrame,
    min_price: float = 5.0,
    min_dollar_volume: float = 20_000_000,
    require_20d_history: bool = True,
) -> pd.DataFrame:
    """
    基础可交易股票池过滤。

    这不是策略条件，只是排除：
    - 太便宜的股票
    - 成交额太低的股票
    - 没有足够历史数据的股票
    """

    df = df.copy()

    df = df[df["close"] >= min_price]
    df = df[df["dollar_volume"] >= min_dollar_volume]

    if require_20d_history:
        df = df[df["avg_volume_20d"].notna()]
        df = df[df["relative_volume_20d"].notna()]
        df = df[df["avg_dollar_volume_20d"].notna()]

    df = df[df["return_1d"].notna()]

    return df


def yesterday_gainer_filter(
    df: pd.DataFrame,
    min_return: float = 0.05,
    max_return: float = 5.0,
) -> pd.DataFrame:
    """
    筛选昨日上涨股票。
    """

    df = df.copy()

    return df[
        (df["return_1d"] >= min_return)
        & (df["return_1d"] <= max_return)
    ]


def market_cap_filter(
    df: pd.DataFrame,
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
) -> pd.DataFrame:
    """
    市值筛选。
    """

    df = df.copy()

    if "market_cap" not in df.columns:
        raise ValueError("DataFrame does not contain market_cap column.")

    df = df[df["market_cap"].notna()]

    if min_market_cap is not None:
        df = df[df["market_cap"] >= min_market_cap]

    if max_market_cap is not None:
        df = df[df["market_cap"] < max_market_cap]

    return df


def dollar_volume_filter(
    df: pd.DataFrame,
    min_dollar_volume: float = 50_000_000,
    max_dollar_volume: float | None = None,
) -> pd.DataFrame:
    """
    成交额筛选。
    """

    df = df.copy()

    df = df[df["dollar_volume"] >= min_dollar_volume]

    if max_dollar_volume is not None:
        df = df[df["dollar_volume"] < max_dollar_volume]

    return df


def relative_volume_filter(
    df: pd.DataFrame,
    min_relative_volume: float = 2.0,
    max_relative_volume: float | None = None,
) -> pd.DataFrame:
    """
    相对成交量筛选。
    """

    df = df.copy()

    df = df[df["relative_volume_20d"].notna()]
    df = df[df["relative_volume_20d"] >= min_relative_volume]

    if max_relative_volume is not None:
        df = df[df["relative_volume_20d"] < max_relative_volume]

    return df


def clv_filter(
    df: pd.DataFrame,
    min_clv: float = 0.7,
) -> pd.DataFrame:
    """
    收盘位置筛选。
    CLV 越高，说明越接近当天高位收盘。
    """

    df = df.copy()

    df = df[df["clv"].notna()]
    df = df[df["clv"] >= min_clv]

    return df


def exchange_filter(
    df: pd.DataFrame,
    exchanges: list[str] | None = None,
) -> pd.DataFrame:
    """
    交易所筛选。
    例如 ["NYSE", "NASDAQ"]。
    """

    df = df.copy()

    if exchanges is None:
        return df

    return df[df["exchange"].isin(exchanges)]


def sic_filter(
    df: pd.DataFrame,
    min_sic: int | None = None,
    max_sic: int | None = None,
) -> pd.DataFrame:
    """
    粗略按 SIC code 筛选行业。
    第一版先用 SIC，后面可以再接 Compustat / NAICS / GICS。
    """

    df = df.copy()

    if "sic_code" not in df.columns:
        return df

    df["sic_code"] = pd.to_numeric(df["sic_code"], errors="coerce")
    df = df[df["sic_code"].notna()]

    if min_sic is not None:
        df = df[df["sic_code"] >= min_sic]

    if max_sic is not None:
        df = df[df["sic_code"] <= max_sic]

    return df


def add_market_cap_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加市值分组。
    """

    df = df.copy()

    bins = [
        0,
        300_000_000,
        1_000_000_000,
        5_000_000_000,
        20_000_000_000,
        float("inf"),
    ]

    labels = [
        "<300M",
        "300M-1B",
        "1B-5B",
        "5B-20B",
        ">20B",
    ]

    df["market_cap_bucket"] = pd.cut(
        df["market_cap"],
        bins=bins,
        labels=labels,
        right=False,
    )

    return df


def add_dollar_volume_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加成交额分组。
    """

    df = df.copy()

    bins = [
        0,
        20_000_000,
        50_000_000,
        100_000_000,
        300_000_000,
        1_000_000_000,
        float("inf"),
    ]

    labels = [
        "<20M",
        "20M-50M",
        "50M-100M",
        "100M-300M",
        "300M-1B",
        ">1B",
    ]

    df["dollar_volume_bucket"] = pd.cut(
        df["dollar_volume"],
        bins=bins,
        labels=labels,
        right=False,
    )

    return df


def add_return_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加昨日涨幅分组。
    """

    df = df.copy()

    bins = [
        -float("inf"),
        0.05,
        0.10,
        0.20,
        0.40,
        0.80,
        float("inf"),
    ]

    labels = [
        "<5%",
        "5%-10%",
        "10%-20%",
        "20%-40%",
        "40%-80%",
        ">80%",
    ]

    df["return_bucket"] = pd.cut(
        df["return_1d"],
        bins=bins,
        labels=labels,
        right=False,
    )

    return df


def add_relative_volume_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加相对成交量分组。
    """

    df = df.copy()

    bins = [
        0,
        1,
        2,
        3,
        5,
        10,
        float("inf"),
    ]

    labels = [
        "<1x",
        "1x-2x",
        "2x-3x",
        "3x-5x",
        "5x-10x",
        ">10x",
    ]

    df["relative_volume_bucket"] = pd.cut(
        df["relative_volume_20d"],
        bins=bins,
        labels=labels,
        right=False,
    )

    return df


def add_all_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加所有分组标签。
    """

    df = df.copy()

    df = add_market_cap_bucket(df)
    df = add_dollar_volume_bucket(df)
    df = add_return_bucket(df)
    df = add_relative_volume_bucket(df)

    return df


def apply_default_strategy_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    第一版策略筛选条件。

    这不是最终策略，只是 baseline。
    """

    df = base_universe_filter(
        df,
        min_price=5.0,
        min_dollar_volume=20_000_000,
        require_20d_history=True,
    )

    df = yesterday_gainer_filter(
        df,
        min_return=0.10,
        max_return=0.40,
    )

    df = market_cap_filter(
        df,
        min_market_cap=1_000_000_000,
        max_market_cap=None,
    )

    df = dollar_volume_filter(
        df,
        min_dollar_volume=100_000_000,
    )

    df = relative_volume_filter(
        df,
        min_relative_volume=3.0,
        max_relative_volume=None,
    )

    df = clv_filter(
        df,
        min_clv=0.70,
    )

    return df
