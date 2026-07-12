# ML Strategy Lab

This folder contains the first machine-learning strategy selection experiment for the premarket momentum project.

## Research Scope

- Signal window: `04:00-09:30`.
- Path window: `04:00-09:30`.
- Entry window: `trigger_time-09:30`.
- Forced exit: latest trade at or before `09:30`.
- Decision model: one decision at `trigger_time`; all model features must be known at or before `trigger_time`.
- Data source: the existing local `04:00-09:30` TAQ cache. The current workflow does not require WRDS access.

## First Strategy Grid

The first grid intentionally stays small:

- `SKIP`: no trade.
- Entry: staged pullback buys at `PB2/PB3/PB5`.
- Weights:
  - `equal`: `33% / 33% / 34%`.
  - `back_weighted`: `20% / 30% / 50%`.
- Exits:
  - `fixed_sl2_tp3`: stop `-2%`, take profit `+3%`.
  - `fixed_sl3_tp5`: stop `-3%`, take profit `+5%`.
  - `half_tp3_trail15_sl3`: stop `-3%`, sell half at `+3%`, trail rest by `1.5%`.
  - `half_tp5_trail2_sl5`: stop `-5%`, sell half at `+5%`, trail rest by `2%`.

All stops and targets are measured against current weighted average cost.

## Workflow

Run from the repository root.

```powershell
python ml_strategy_lab\scripts\02_build_strategy_grid.py
python ml_strategy_lab\scripts\03_build_event_features.py
python ml_strategy_lab\scripts\04_build_model_dataset.py
python ml_strategy_lab\scripts\05_train_lightgbm.py
```

Use `--max-dates` on the first two scripts for a quick smoke test. The optional
`01_cache_taq_0400_1100.py` script is retained for future use if WRDS access is
reactivated, but it is not part of this experiment.

## Outputs

- `data/strategy_grid_results.csv`: realized return for every event and strategy template.
- `data/event_features.csv`: as-of features computed only from data available at trigger time.
- `data/model_dataset.csv`: event features joined with strategy parameters and labels.
- `reports/lightgbm_baseline_report.md`: validation metrics and baseline comparison.

## Current Baseline Result

The full local run contains 2,291 events and 20,619 event-strategy rows.
The models train on 2015-2021 and validate on 2022-2024.

- Risk model validation AUC: `0.8310`.
- Return model validation MAE: `0.018615`.
- The selector entered 311 of 812 validation events.
- Selector return sum: `+26.31%`; average per validation event: `+0.032%`.
- Entered-trade average return: `+0.085%`.
- Validation-year return sums: 2022 `-35.19%`, 2023 `+31.05%`, 2024 `+30.45%`.
- Cumulative-sum drawdown proxy: `-53.72%`.
- Bootstrap 95% interval for average event return: approximately `-0.095%` to `+0.157%`.

This is a research baseline, not a deployable strategy. The risk classifier has
useful separation, but the return edge is small, unstable by year, and not
statistically distinguishable from zero in the validation period.
