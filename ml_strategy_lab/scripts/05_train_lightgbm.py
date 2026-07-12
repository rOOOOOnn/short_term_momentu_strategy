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
    losses = returns[returns < 0]
    wins = returns[returns > 0]
    cumulative = returns.cumsum()
    running_high = cumulative.cummax()
    drawdown = cumulative - running_high
    return {
        "name": name,
        "events": selected["event_id"].nunique(),
        "entered": int(selected["trade_entered"].sum()) if "trade_entered" in selected else int((selected["strategy_id"] != "SKIP").sum()),
        "avg_return": returns.mean(),
        "median_return": returns.median(),
        "total_equal_weight_return": returns.sum(),
        "win_rate": returns.gt(0).mean(),
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
            group["pred_return"].ge(min_return)
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
        "avg_return",
        "median_return",
        "total_equal_weight_return",
        "win_rate",
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

    train = df[df["year"].le(int(config["train_end_year"]))].copy()
    validation = df[df["year"].ge(int(config["validation_start_year"]))].copy()
    X_train = train[feature_cols]
    y_return = train[config["return_target"]]
    y_risk = train[config["risk_target"]]

    return_model = LGBMRegressor(random_state=int(config["random_state"]), **config["return_model_params"])
    risk_model = LGBMClassifier(random_state=int(config["random_state"]), **config["risk_model_params"])
    return_model.fit(X_train, y_return)
    risk_model.fit(X_train, y_risk)

    validation = validation.copy()
    validation["pred_return"] = return_model.predict(validation[feature_cols])
    validation["pred_large_loss_prob"] = risk_model.predict_proba(validation[feature_cols])[:, 1]

    return_mae = mean_absolute_error(validation[config["return_target"]], validation["pred_return"])
    risk_auc = (
        roc_auc_score(validation[config["risk_target"]], validation["pred_large_loss_prob"])
        if validation[config["risk_target"]].nunique() > 1
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
    report = [
        "# LightGBM Strategy Selector Baseline",
        "",
        "## Setup",
        "",
        f"- Train: years <= {config['train_end_year']}.",
        f"- Validation: years >= {config['validation_start_year']}.",
        f"- Rows: {len(df):,}; events: {df['event_id'].nunique():,}; strategies: {df['strategy_id'].nunique():,}.",
        f"- Feature count: {len(feature_cols)}.",
        f"- Return MAE on validation rows: {return_mae:.6f}.",
        f"- Risk AUC on validation rows: {risk_auc:.4f}" if not pd.isna(risk_auc) else "- Risk AUC unavailable.",
        "",
        "## Validation Strategy Selection",
        "",
        markdown_table(summary_df),
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
