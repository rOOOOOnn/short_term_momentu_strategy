# Intraday Strategy Selection Report

## Executive Summary

The resumed WRDS TAQ pipeline completed successfully for all 1,036 required
trading dates. It produced 2,291 events and 20,619 event-strategy observations
covering eight tradable strategy templates plus a no-trade fallback.

The implementation and key accounting formulas passed the independent checks
described below. However, the out-of-sample result is negative. The LightGBM
selector's mean return per validation event is **-0.16%**, with a bootstrap 95%
interval of **-0.29% to -0.02%**. The current strategy should therefore be
treated as a valid negative research result, not as a deployable trading edge.

## Data and Evaluation Design

- Signal information: trades known at or before the event trigger.
- Intraday path: `04:00-16:00` US Eastern time.
- Staged entries: 2%, 3%, and 5% pullbacks from the trigger price.
- Entry deadline: `11:00`; forced exit: `16:00`.
- Execution penalty: 25 bps on entry and 25 bps on exit.
- Training period: 2015-2021.
- Validation period: 2022-2024.
- Validation events: 812.
- Models are trained and evaluated on tradable strategies only. `SKIP` is a
  decision fallback and is not treated as a zero-return training observation.

## Model Quality

| Metric | Model | Reference | Interpretation |
| --- | ---: | ---: | --- |
| Return MAE | 0.028397 | 0.029506 for an always-zero prediction | Small improvement; return prediction remains noisy |
| Large-loss AUC | 0.7220 | 0.5000 random ranking | Useful separation, but insufficient to create positive returns |

The selector chose `SKIP` for 520 of 812 events. Of the 292 non-SKIP choices,
277 actually filled and 15 had no qualifying entry fill.

## Out-of-Sample Strategy Results

| Strategy | Events | Entered | Mean event return | Mean entered return | Event win rate | Entered win rate | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM selector | 812 | 277 | -0.16% | -0.46% | 19.21% | 56.32% | 0.71 |
| Back-weighted, fixed SL2/TP3 | 812 | 756 | -0.66% | -0.71% | 38.55% | 41.40% | 0.45 |
| Back-weighted, fixed SL3/TP5 | 812 | 756 | -1.00% | -1.08% | 40.52% | 43.52% | 0.55 |
| Back-weighted, half TP3/trail 1.5/SL3 | 812 | 756 | -0.72% | -0.77% | 58.74% | 63.10% | 0.53 |
| Back-weighted, half TP5/trail 2/SL5 | 812 | 756 | -0.98% | -1.05% | 52.09% | 55.95% | 0.59 |
| Equal-weighted, fixed SL2/TP3 | 812 | 756 | -0.87% | -0.93% | 34.73% | 37.30% | 0.42 |
| Equal-weighted, fixed SL3/TP5 | 812 | 756 | -1.25% | -1.35% | 35.96% | 38.62% | 0.49 |
| Equal-weighted, half TP3/trail 1.5/SL3 | 812 | 756 | -0.92% | -0.99% | 53.45% | 57.41% | 0.47 |
| Equal-weighted, half TP5/trail 2/SL5 | 812 | 756 | -1.27% | -1.36% | 47.29% | 50.79% | 0.53 |

The entered-trade win rate looks positive, but the average loss is materially
larger than the average win. For the selector, the average winning event is
2.01%, while the average losing event is -3.63%. That asymmetry explains why a
56.32% entered-trade win rate still loses money.

### Selector Results by Year

| Validation year | Events | Entered | Mean event return | Non-compounded return sum |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 191 | 61 | -0.29% | -55.95% |
| 2023 | 222 | 84 | -0.04% | -8.81% |
| 2024 | 399 | 132 | -0.15% | -61.65% |

The return sum and cumulative-sum drawdown are diagnostics over event returns,
not compounded portfolio returns. Overlapping positions and capital constraints
are not represented by those figures.

## Formula and Integrity Checks

All of the following checks completed with zero failures:

| Check | Coverage |
| --- | ---: |
| Cache date coverage | 1,036 dates |
| Grid row count and event-strategy uniqueness | 20,619 rows |
| Nine strategies per event | 2,291 events |
| Entry and exit time bounds | 16,472 entered strategy rows |
| Invested-fraction formula | 20,619 rows |
| Weighted average entry-price formula | 16,472 entered strategy rows |
| Fixed-exit capital-return formula | 8,236 rows |
| Stop-loss trigger threshold | 7,860 rows |
| Take-profit trigger threshold | 3,409 rows |
| No-entry zero-exposure state | 1,856 rows |
| Model dataset uniqueness and feature joins | 20,619 rows |
| Declared target/leakage exclusion | 84 model features |
| Partial-exit accounting hand calculation | 1 deterministic unit case |

The original model pipeline included `SKIP` rows as if they were normal
strategies with known zero returns. This inflated the apparent model metrics
and allowed the fallback to participate in model ranking. The issue was fixed:
models now use only tradable rows, and `SKIP` is applied only after no tradable
strategy meets both prediction thresholds.

## Limitations and Next Research Steps

1. Every qualifying trade print is treated as fully fillable. Quote depth,
   latency, commissions, halts, and market impact are not modeled.
2. Exchange, sale-condition, and reporting-facility fields are retained in the
   cache but are not yet used to reject potentially ineligible prints.
3. A single print can satisfy multiple staged pullback levels, which assumes
   enough liquidity to fill all triggered orders.
4. The simulator applies adverse slippage to entry and exit, but this remains a
   simplified execution model rather than an order-book simulation.
5. For 4,622 historical partial-take-profit rows, the first partial exit's time
   and price were not persisted. The partial-exit accounting function passes a
   deterministic hand-calculated test, but those historical rows cannot be
   replayed independently from the output table alone.
6. The next useful experiment is not simply a larger model. It is an
   execution-quality sensitivity study: filter trade conditions, constrain
   fills by reported size, persist every fill leg, add commissions, and test
   whether the negative conclusion changes.

## Reproducibility

The local generated artifacts are intentionally excluded from version control.
Run the pipeline and its independent validator from the repository root:

```powershell
conda run -n wrdsenv python ml_strategy_lab\scripts\06_run_intraday_pipeline.py
conda run -n wrdsenv python tester\validate_intraday_results.py
```

The generated detailed report is written to
`ml_strategy_lab/reports/intraday/lightgbm_intraday_report.md`, and the machine-
readable audit is written to
`ml_strategy_lab/reports/intraday/pipeline_validation.json`.
