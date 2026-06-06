"""Default parameters for the screening workflow."""

DEFAULT_INPUT_PATH = "data/processed/crsp_daily_features.parquet"
DEFAULT_OUTPUT_DIR = "screen_results"
DEFAULT_TOP_N = 20
DEFAULT_MODE = "gainers"

BASE_UNIVERSE = {
    "min_price": 5.0,
    "min_dollar_volume": 20_000_000,
    "require_20d_history": True,
}

GAINERS_FILTER = {
    "min_return": 0.05,
    "max_return": 5.0,
}
