# Short-Term Momentum Strategy Research

English | [中文](README.md)

An empirical research project on U.S. premarket momentum. It asks a focused question: **after a stock ranks among the top 50 gainers on day T and confirms momentum before the next market open, is there still meaningful upside left?**

The repository covers the full research workflow: universe construction from CRSP daily data, TAQ trade processing, event-driven signal analysis, entry/exit simulation, time-based validation, and a leakage-aware LightGBM strategy-selection experiment.

> This repository is for research and backtesting only. It is not investment advice and does not present a production-ready trading strategy.

## Research Snapshot

The latest event study uses ordinary-share TAQ trades from `04:00` to `09:30` rather than fixed-window approximations.

**Candidate universe**

- Top 50 day-T stocks ranked by one-day return
- Market capitalization between `$300M` and `$5B`

**Premarket trigger**

- Price at least `5%` above the day-T close
- At least `$1M` in cumulative premarket dollar volume
- At least `100` cumulative trades

**2015-2024 sample results**

| Metric | Result |
| --- | ---: |
| Candidate events | 3,402 |
| Triggered events | 2,291 |
| Trigger coverage | 67.3% |
| Median trigger time | 07:56 |
| Median maximum favorable excursion to the daily high | +4.03% |
| Probability of reaching +5% | 45.13% |
| Median maximum adverse excursion to the daily low | -7.92% |

![Train and validation trigger results](docs/reports/premarket_strategy_summary/charts/train_validation_dynamic_trigger.png)

The signal identifies stocks that often retain intraday upside, but the adverse excursion is substantial. The evidence therefore does **not** support blindly buying at the trigger; execution, position sizing, and exit design remain central research problems.

## Machine-Learning Experiment

The [`ml_strategy_lab`](ml_strategy_lab/) evaluates whether a model can choose among staged pullback entries and several risk-management templates using only information available at trigger time.

- One decision is made at the trigger timestamp; all features are computed as-of that timestamp.
- Models train on `2015-2021` and validate on `2022-2024`.
- The full run contains 2,291 events and 20,619 event-strategy observations.
- Validation risk-model AUC: `0.8310`.
- Validation return-model MAE: `0.018615`.
- The selector entered 311 of 812 validation events.

The result is intentionally reported with its weaknesses: validation-year returns are unstable, the cumulative-sum drawdown proxy is `-53.72%`, and the bootstrap 95% interval for average event return crosses zero. This is a useful research baseline, not evidence of a deployable edge.

See [`ml_strategy_lab/README.md`](ml_strategy_lab/README.md) for the experiment design, strategy grid, and complete baseline results.

## Research Workflow

```text
CRSP daily data
    -> rank day-T gainers and build the candidate universe
TAQ premarket trades (04:00-09:30)
    -> replay cumulative price, dollar-volume, and trade-count conditions
Triggered events
    -> measure favorable/adverse paths and simulate entry/exit templates
As-of event features + realized strategy outcomes
    -> train on 2015-2021 and validate on 2022-2024
```

## Repository Structure

```text
data/wrds/
  preprocess.py                         # CRSP download and daily features
  preprocess_premarket.py               # TAQ premarket aggregation helpers

tester/
  run_screen.py                         # Day-T top-gainer screening
  analyze_full_premarket_dynamic_triggers.py
                                         # Trade-level 04:00-09:30 trigger replay
  analyze_high_coverage_entries.py       # Fixed and pullback entry analysis
  analyze_staged_entries.py              # Staged-entry analysis
  generate_research_summary_report.py    # Final report and chart generation

ml_strategy_lab/
  configs/                               # Strategy grid and model settings
  scripts/                               # Dataset construction and LightGBM pipeline

docs/reports/premarket_strategy_summary/
                                         # Research report, PDF, and charts
```

Large raw datasets, TAQ caches, credentials, and generated local results are intentionally excluded from version control.

## Reproducing the Analysis

The project uses Python with pandas, NumPy, Matplotlib, scikit-learn, LightGBM, and WRDS. Run commands from the repository root.

Generate the final research summary and charts:

```powershell
python tester\generate_research_summary_report.py
```

Run the trade-level dynamic-trigger analysis:

```powershell
python tester\analyze_full_premarket_dynamic_triggers.py `
  --input screen_results\20150102_20241231_gainers_return_1d_top50\premarket\premarket_enriched_results.csv `
  --progress-every 100
```

Run the machine-learning pipeline:

```powershell
python ml_strategy_lab\scripts\02_build_strategy_grid.py
python ml_strategy_lab\scripts\03_build_event_features.py
python ml_strategy_lab\scripts\04_build_model_dataset.py
python ml_strategy_lab\scripts\05_train_lightgbm.py
```

The ML workflow can use the existing local `04:00-09:30` TAQ cache. Use `--max-dates` on the first two pipeline scripts for a quick smoke test.

## Data Access

CRSP and TAQ data are accessed through WRDS and cannot be redistributed in this repository. Local credentials may be supplied through an ignored `.env` file:

```text
WRDS_USERNAME=...
WRDS_PASSWORD=...
```

## Known Limitations and Next Steps

- Add transaction costs, slippage, liquidity constraints, and realistic fill assumptions.
- Replace the drawdown proxy with a capital-aware portfolio backtest.
- Expand walk-forward and regime-based validation.
- Test pullback, re-breakout, VWAP-area, and staged entries after the trigger.
- Develop explicit stop-loss, partial-profit, trailing-stop, and market-cap-based sizing rules.

The detailed Chinese research report is available in [Markdown](docs/reports/premarket_strategy_summary/premarket_strategy_research_summary_zh.md) and [PDF](docs/reports/premarket_strategy_summary/premarket_strategy_research_summary_zh.pdf).
