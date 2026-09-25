# short_term_momentu_strategy

[English](README_EN.md) | 中文

短线动量策略研究项目。当前重点是研究 **T 日涨幅前 50 的股票，在下一交易日盘前出现动量信号后，是否还有当天上冲空间**，并为后续策略开发确定可行方向。

> 这是研究和回测代码，不是投资建议。

## 当前结论

最新完整逐笔盘前研究使用 04:00-09:30 的普通股 TAQ 成交数据，不再使用固定窗口近似。

当前建议的信号层：

```text
股票池：T 日涨幅前 50
市值：300M-5B
盘前时间：04:00-09:30
触发条件：
  相对 T 日收盘价上涨 >= 5%
  累计盘前成交额 >= 1M
  累计成交笔数 >= 100
```

该条件在 2015-2024 样本中：

```text
候选数：3402
触发数：2291
覆盖率：67.3%
中位触发时间：07:56
中位 MFE 到当天最高：+4.03%
达到 +5% 概率：45.13%
中位 MAE 到当天最低：-7.92%
```

关键解释：

- 信号证明这类股票当天通常仍有上冲空间。
- 但 MAE 很深，不能触发后直接追高满仓。
- 下一步策略开发应研究触发后的入场和退出：回踩买、重新突破买、分批买、移动止损、分批止盈。

完整报告和图表：

[docs/reports/premarket_strategy_summary/premarket_strategy_research_summary_zh.md](docs/reports/premarket_strategy_summary/premarket_strategy_research_summary_zh.md)

## 项目结构

```text
data/wrds/
  preprocess.py                    # CRSP 日线数据下载和特征处理
  preprocess_premarket.py           # TAQ 盘前窗口聚合

tester/
  run_screen.py                     # T 日涨幅榜筛选和年度结果输出
  analyze_premarket_patterns.py     # 盘前窗口形态分析
  analyze_dynamic_premarket_triggers.py
                                    # 窗口级动态触发分析
  analyze_full_premarket_dynamic_triggers.py
                                    # 04:00-09:30 逐笔动态触发分析
  analyze_high_coverage_entries.py  # 高覆盖固定/回踩入场分析
  analyze_staged_entries.py         # 分批入场分析
  generate_research_summary_report.py
                                    # 生成最终研究报告和图表

docs/reports/
  premarket_strategy_summary/       # 可提交到 GitHub 的总结报告和图表
```

本地大结果、逐笔缓存和原始数据默认不提交：

```text
screen_results/
data/raw/
data/processed/
.env
```

## 常用命令

生成最终总结报告：

```powershell
python tester\generate_research_summary_report.py
```

运行完整逐笔动态触发分析：

```powershell
python tester\analyze_full_premarket_dynamic_triggers.py `
  --input screen_results\20150102_20241231_gainers_return_1d_top50\premarket\premarket_enriched_results.csv `
  --progress-every 100
```

运行窗口级动态触发分析：

```powershell
python tester\analyze_dynamic_premarket_triggers.py `
  --input screen_results\20150102_20241231_gainers_return_1d_top50\premarket\premarket_enriched_results.csv
```

## WRDS 登录

本地 `.env` 支持：

```text
WRDS_USERNAME=...
WRDS_PASSWORD=...
```

`.env` 已被 `.gitignore` 排除，不应提交到仓库。

## 下一步

1. 基于完整逐笔触发信号开发入场层：
   - 触发后回踩 1%-3% 买入。
   - 重新突破触发价买入。
   - VWAP/成交密集区附近承接买入。
   - 分批入场降低追高风险。
2. 开发退出层：
   - +3%/+5% 分批止盈。
   - -2%/-3% 初始止损。
   - 盈利后保本。
   - 最高价回撤 1%-2% 移动止损。
3. 按市值分层控制仓位：
   - 300M-1B：波动空间大，但仓位更小。
   - 1B-5B：容量更好，但预期空间更低。
