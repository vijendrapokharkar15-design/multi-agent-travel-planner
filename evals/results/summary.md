# Evaluation results

Run on 2026-10-06 00:38 with model `gpt-5.6-luna`, 8 cases x 3 repeats per system. Both systems use the same model, the same mock data and the same deterministic scorer.

## Overall

| system | success | runs_with_hard_violations | hard_violations_per_run | format_issues_per_run | said_infeasible | within_budget | avg_activities | avg_revisions | avg_llm_calls | median_latency_s | avg_cost_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 54% | 46% | 0.50 | 0.00 | 12% | 88% | 5.4 | 0.00 | 1.0 | 16.7 | 0.0026 |
| multi_agent | 100% | 0% | 0.00 | 0.00 | 0% | 88% | 6.0 | 1.17 | 7.2 | 32.9 | 0.0048 |

## Successful runs per case

| case | baseline | multi_agent |
|---|---|---|
| 01_standard_lisbon | 2/3 | 3/3 |
| 02_tight_barcelona | 0/3 | 3/3 |
| 03_impossible_budget | 3/3 | 3/3 |
| 04_same_day | 3/3 | 3/3 |
| 05_budget_style | 2/3 | 3/3 |
| 06_monday_closures | 1/3 | 3/3 |
| 07_group_of_five | 1/3 | 3/3 |
| 08_free_text | 1/3 | 3/3 |
