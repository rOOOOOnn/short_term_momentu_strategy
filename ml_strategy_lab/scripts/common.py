from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = PROJECT_ROOT / "ml_strategy_lab"
DEFAULT_PREMARKET_ROOT = (
    PROJECT_ROOT
    / "screen_results"
    / "20150102_20241231_gainers_return_1d_top50"
    / "premarket"
)
DEFAULT_TRIGGER_DIR = DEFAULT_PREMARKET_ROOT / "full_premarket_dynamic_triggers"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_time(value: str) -> pd.Timedelta:
    return pd.to_timedelta(value)


def parse_trigger_time(value: object) -> pd.Timedelta:
    text = str(value)
    if text.startswith("0 days "):
        text = text.replace("0 days ", "", 1)
    return pd.to_timedelta(text)


def load_baseline_triggers(
    trigger_dir: str | Path = DEFAULT_TRIGGER_DIR,
    trigger_floor: float = 0.05,
    min_dollar_volume: float = 1_000_000,
    min_trade_count: int = 100,
) -> pd.DataFrame:
    path = Path(trigger_dir) / "full_premarket_dynamic_trigger_results.csv"
    df = pd.read_csv(path, parse_dates=["date", "trade_date"])
    selected = df[
        df["trigger_floor"].eq(trigger_floor)
        & df["min_cum_dollar_volume"].eq(min_dollar_volume)
        & df["min_cum_trade_count"].eq(min_trade_count)
    ].copy()
    selected["event_id"] = (
        selected["trade_date"].dt.strftime("%Y%m%d")
        + "_"
        + selected["symbol"].astype(str)
        + "_"
        + selected["permno"].astype(str)
    )
    return selected.sort_values(["trade_date", "symbol", "permno"]).reset_index(drop=True)


def prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["time_m"] = pd.to_timedelta(df["time_m"].astype(str))
    for column in ["price", "size", "tr_seqnum"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    required = ["time_m", "sym_root", "price"]
    if "size" in df.columns:
        required.append("size")
    df = df.dropna(subset=required)
    if "size" in df.columns:
        df["dollar_volume"] = df["price"] * df["size"]
    sort_cols = ["sym_root", "time_m"]
    if "tr_seqnum" in df.columns:
        sort_cols.append("tr_seqnum")
    return df.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def read_cached_trades(path: str | Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    usecols = [
        col
        for col in ["date", "time_m", "sym_root", "sym_suffix", "size", "price", "tr_corr", "tr_seqnum"]
        if col in header
    ]
    return prepare_trades(pd.read_csv(path, usecols=usecols, parse_dates=["date"]))

