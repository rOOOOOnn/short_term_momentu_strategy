from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from common import LAB_ROOT, ensure_dir, load_json


DEFAULT_DATASET = LAB_ROOT / "data" / "model_dataset.csv"
DEFAULT_CONFIG = LAB_ROOT / "configs" / "lightgbm_baseline.json"
DEFAULT_REPORT = LAB_ROOT / "reports" / "lightgbm_baseline_report.md"
DEFAULT_MODEL_DIR = LAB_ROOT / "models"


LEAKAGE_COLUMNS = {
    "capital_return",
    "position_return",
    "max_runup",
    "max_drawdown",
    "hit_stop",
    "hit_take_profit",
    "no_entry",
    "trade_entered",
    "large_loss",
    "filled_entry_count",
    "first_entry_time",
    "last_entry_time",
    "avg_entry_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "invested_fraction",
    "validation_period",
}

ID_COLUMNS = {
    "event_id",
    "date",
    "trade_date",
    "symbol",
    "permno",
    "strategy_id",
    "entry_type",
    "weight_scheme",
    "exit_name",
    "exit_type",
    "trigger_time",
    # This flag is deterministic from strategy_id. SKIP is a decision fallback,
    # not a strategy whose zero outcome should be learned by the return/risk models.
    "is_skip_strategy",
}


def load_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise SystemExit(
            "lightgbm is not installed in the current Python environment. "
            "Install it with: python -m pip install lightgbm"
        ) from exc
    return LGBMClassifier, LGBMRegressor


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = LEAKAGE_COLUMNS | ID_COLUMNS
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return cols


def summarize_returns(name: str, selected: pd.DataFrame) -> dict[str, object]:
    returns = selected["capital_return"].fillna(0.0)
    entered_mask = selected["trade_entered"].astype(bool) if "trade_entered" in selected else selected["strategy_id"].ne("SKIP")
    entered_returns = returns[entered_mask]
    losses = returns[returns < 0]
    wins = returns[returns > 0]
    cumulative = returns.cumsum()
    running_high = cumulative.cummax()
    drawdown = cumulative - running_high
    return {
        "name": name,
        "events": selected["event_id"].nunique(),
        "entered": int(entered_mask.sum()),
        "avg_event_return": returns.mean(),
        "avg_entered_return": entered_returns.mean() if not entered_returns.empty else 0.0,
        "median_return": returns.median(),
        "return_sum_proxy": returns.sum(),
        "event_win_rate": returns.gt(0).mean(),
        "entered_win_rate": entered_returns.gt(0).mean() if not entered_returns.empty else 0.0,
        "avg_win": wins.mean() if not wins.empty else 0.0,
        "avg_loss": losses.mean() if not losses.empty else 0.0,
        "profit_factor": wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.nan,
        "p_loss_2pct": returns.le(-0.02).mean(),
        "p_gain_3pct": returns.ge(0.03).mean(),
        "max_drawdown_proxy": drawdown.min() if not drawdown.empty else 0.0,
    }


def choose_by_model(df: pd.DataFrame, min_return: float, max_risk: float) -> pd.DataFrame:
    chosen = []
    for _, group in df.groupby("event_id", sort=False):
        eligible = group[
            group["strategy_id"].ne("SKIP")
            & group["pred_return"].ge(min_return)
            & group["pred_large_loss_prob"].le(max_risk)
        ].copy()
        if eligible.empty:
            skip = group[group["strategy_id"].eq("SKIP")]
            chosen.append(skip.iloc[0] if not skip.empty else group.iloc[group["pred_return"].argmax()])
            continue
        chosen.append(eligible.sort_values(["pred_return", "pred_large_loss_prob"], ascending=[False, True]).iloc[0])
    return pd.DataFrame(chosen)


def choose_baseline(df: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby("event_id", sort=False):
        selected = group[group["strategy_id"].eq(strategy_id)]
        if selected.empty:
            selected = group[group["strategy_id"].eq("SKIP")]
        rows.append(selected.iloc[0])
    return pd.DataFrame(rows)


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:+.2%}" if value < 0 else f"{value:.2%}"


def markdown_table(df: pd.DataFrame) -> str:
    view = df.copy()
    pct_cols = [
        "avg_event_return",
        "avg_entered_return",
        "median_return",
        "return_sum_proxy",
        "event_win_rate",
        "entered_win_rate",
        "avg_win",
        "avg_loss",
        "profit_factor",
        "p_loss_2pct",
        "p_gain_3pct",
        "max_drawdown_proxy",
    ]
    for col in pct_cols:
        if col in view.columns and col != "profit_factor":
            view[col] = view[col].map(pct)
        elif col in view.columns:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def annual_summary(selected: pd.DataFrame) -> pd.DataFrame:
    view = selected.copy()
    view["year"] = pd.to_datetime(view["trade_date"]).dt.year
    return (
        view.groupby("year", as_index=False)
        .agg(
            events=("event_id", "nunique"),
            entered=("trade_entered", "sum"),
            avg_event_return=("capital_return", "mean"),
            return_sum_proxy=("capital_return", "sum"),
        )
    )


def bootstrap_mean_interval(returns: pd.Series, seed: int, repetitions: int = 10_000) -> tuple[float, float]:
    values = returns.fillna(0.0).to_numpy(dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    # Chunking avoids allocating a repetitions-by-events matrix all at once.
    for start in range(0, repetitions, 500):
        stop = min(start + 500, repetitions)
        draws = rng.choice(values, size=(stop - start, len(values)), replace=True)
        means[start:stop] = draws.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM return/risk strategy selector.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    LGBMClassifier, LGBMRegressor = load_lightgbm()
    args = parse_args()
    config = load_json(args.config)
    df = pd.read_csv(args.dataset, parse_dates=["date", "trade_date"])
    df = df.replace([np.inf, -np.inf], np.nan)
    feature_cols = numeric_feature_columns(df)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    train_all = df[df["year"].le(int(config["train_end_year"]))].copy()
    validation = df[df["year"].ge(int(config["validation_start_year"]))].copy()
    train = train_all[train_all["strategy_id"].ne("SKIP")].copy()
    validation_trades = validation[validation["strategy_id"].ne("SKIP")].copy()
    X_train = train[feature_cols]
    y_return = train[config["return_target"]]
    y_risk = train[config["risk_target"]]

    return_model = LGBMRegressor(random_state=int(config["random_state"]), **config["return_model_params"])
    risk_model = LGBMClassifier(random_state=int(config["random_state"]), **config["risk_model_params"])
    return_model.fit(X_train, y_return)
    risk_model.fit(X_train, y_risk)

    validation = validation.copy()
    validation["pred_return"] = np.nan
    validation["pred_large_loss_prob"] = np.nan
    validation.loc[validation_trades.index, "pred_return"] = return_model.predict(validation_trades[feature_cols])
    validation.loc[validation_trades.index, "pred_large_loss_prob"] = risk_model.predict_proba(validation_trades[feature_cols])[:, 1]
    validation_trades = validation.loc[validation_trades.index]

    return_mae = mean_absolute_error(validation_trades[config["return_target"]], validation_trades["pred_return"])
    zero_return_mae = mean_absolute_error(
        validation_trades[config["return_target"]],
        np.zeros(len(validation_trades)),
    )
    risk_auc = (
        roc_auc_score(validation_trades[config["risk_target"]], validation_trades["pred_large_loss_prob"])
        if validation_trades[config["risk_target"]].nunique() > 1
        else np.nan
    )

    selected = choose_by_model(
        validation,
        float(config["min_predicted_return"]),
        float(config["max_predicted_large_loss_probability"]),
    )

    summaries = [summarize_returns("LightGBM selector", selected)]
    for strategy_id in sorted(validation["strategy_id"].unique()):
        if strategy_id == "SKIP" or strategy_id.startswith("PB235"):
            summaries.append(summarize_returns(strategy_id, choose_baseline(validation, strategy_id)))
    summary_df = pd.DataFrame(summaries)
    annual_df = annual_summary(selected)
    ci_low, ci_high = bootstrap_mean_interval(
        selected["capital_return"], int(config["random_state"])
    )

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "return_importance": return_model.feature_importances_,
            "risk_importance": risk_model.feature_importances_,
        }
    ).sort_values(["return_importance", "risk_importance"], ascending=False)

    ensure_dir(args.model_dir)
    return_model.booster_.save_model(str(args.model_dir / "lightgbm_return.txt"))
    risk_model.booster_.save_model(str(args.model_dir / "lightgbm_risk.txt"))

    ensure_dir(Path(args.report).parent)
    report_title = config.get("report_title", "LightGBM Strategy Selector Baseline")
    report = [
        f"# {report_title}",
        "",
        "## Setup",
        "",
        f"- Train: years <= {config['train_end_year']}.",
        f"- Validation: years >= {config['validation_start_year']}.",
        f"- Rows: {len(df):,}; events: {df['event_id'].nunique():,}; strategies: {df['strategy_id'].nunique():,} (8 tradable plus SKIP).",
        f"- Feature count: {len(feature_cols)}.",
        "- Models are trained and evaluated only on tradable strategy rows. SKIP is used only when no strategy clears both model thresholds.",
        f"- Return MAE on tradable validation rows: {return_mae:.6f}; zero-prediction benchmark: {zero_return_mae:.6f}.",
        f"- Risk AUC on tradable validation rows: {risk_auc:.4f}." if not pd.isna(risk_auc) else "- Risk AUC unavailable.",
        "",
        "## Validation Strategy Selection",
        "",
        markdown_table(summary_df),
        "",
        "The return-sum and drawdown columns are non-compounded diagnostics over event returns; they are not portfolio returns. Events with SKIP or no fill contribute zero to event-level statistics.",
        "",
        f"Bootstrap 95% interval for the selector's mean event return: {pct(ci_low)} to {pct(ci_high)} (10,000 event-level resamples, seed {config['random_state']}).",
        "",
        "## Selector Results by Validation Year",
        "",
        markdown_table(annual_df),
        "",
        "## Formula and Methodology Review",
        "",
        "- Entry weights, average entry price, fixed-exit capital return, stop/take-profit thresholds, timestamps, no-entry handling, row uniqueness, feature joins, and declared leakage exclusions are checked by `tester/validate_intraday_results.py`.",
        "- The simulator applies configured adverse entry and exit slippage, but treats every qualifying print as fully fillable. It does not model quote depth, latency, commissions, halts, or market impact.",
        "- Trade condition, exchange, and reporting-facility fields are retained in the cache but are not yet used to screen prints. Results therefore require execution-quality sensitivity tests before any live use.",
        "- Partial take-profit rows cannot be reconstructed from the result table alone because the partial fill time and price are not persisted. The aggregate P&L logic was code-reviewed, but this subset lacks an independent output-only replay check.",
        "- Weak or negative validation performance is a strategy result, not a formula failure; no profitability claim should be inferred from this report.",
        "",
        "## Top Feature Importances",
        "",
        markdown_table(importances.head(30)),
    ]
    Path(args.report).write_text("\n".join(report), encoding="utf-8")
    selected.to_csv(Path(args.report).with_name("lightgbm_selected_validation_trades.csv"), index=False)
    print(Path(args.report).resolve())


if __name__ == "__main__":
    main()
