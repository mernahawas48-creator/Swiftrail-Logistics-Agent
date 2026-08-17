# Live Planning Cost / Quality Benchmark

- Model: `mistral-small-latest`
- LLM calls: real Mistral API calls
- Token counts: provider-reported usage metadata
- Tool evidence: fixed Swiftrail seed-data-shaped snapshots
- Evaluation format: shared plain ACTION-line contract
- Standard input price: $0.15/1M tokens
- Standard output price: $0.6/1M tokens

## Aggregate comparison

| Method | Success | Avg LLM calls | Avg input tokens | Avg output tokens | Avg total tokens | Avg latency | Estimated cost/run | Avg tool calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decomposition-first | 2/2 | 14 | 9630.5 | 2729.5 | 12360 | 23344.177 ms | $0.00308227 | 5 |
| Dynamic decomposition | 1/2 | 5.5 | 3097 | 470.5 | 3567.5 | 5954.995 ms | $0.00074685 | 4.5 |
| Plan-and-Solve | 2/2 | 1 | 1051.5 | 166 | 1217.5 | 2321.562 ms | $0.00025732 | 0 |
| Tree of Thoughts | 1/2 | 9 | 8415 | 1017.5 | 9432.5 | 11503.719 ms | $0.00187275 | 0 |
| LATS ungrounded | 2/2 | 2 | 1939 | 83.5 | 2022.5 | 1649.617 ms | $0.00034095 | 0 |
| LATS grounded | 2/2 | 2 | 1936 | 77.5 | 2013.5 | 1684.898 ms | $0.00033690 | 0 |
| Self-Refine | 2/2 | 2 | 2079 | 134.5 | 2213.5 | 1978.337 ms | $0.00039255 | 0 |
| Reflexion | 2/2 | 1 | 1050.5 | 48 | 1098.5 | 922.317 ms | $0.00018638 | 0 |

## Per-case runs

| Group | Case | Method | Success | LLM calls | Input tokens | Output tokens | Total tokens | Latency | Cost | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| decomposition | stable_minor_hold | Decomposition-first | True | 13 | 8613 | 2717 | 11330 | 22278.399 ms | $0.00292215 |  |
| decomposition | stable_minor_hold | Dynamic decomposition | False | 6 | 3439 | 572 | 4011 | 6851.850 ms | $0.00085905 |  |
| decomposition | severe_hold_sales_rep | Decomposition-first | True | 15 | 10648 | 2742 | 13390 | 24409.954 ms | $0.00324240 |  |
| decomposition | severe_hold_sales_rep | Dynamic decomposition | True | 5 | 2755 | 369 | 3124 | 5058.140 ms | $0.00063465 |  |
| planning | stable_minor_hold | Plan-and-Solve | True | 1 | 1099 | 186 | 1285 | 1633.471 ms | $0.00027645 |  |
| planning | stable_minor_hold | Tree of Thoughts | False | 9 | 8776 | 1009 | 9785 | 11486.715 ms | $0.00192180 |  |
| planning | stable_minor_hold | LATS ungrounded | True | 2 | 2034 | 82 | 2116 | 1682.600 ms | $0.00035430 |  |
| planning | stable_minor_hold | LATS grounded | True | 2 | 2031 | 78 | 2109 | 1847.554 ms | $0.00035145 |  |
| planning | above_authority_rate | Plan-and-Solve | True | 1 | 1004 | 146 | 1150 | 3009.653 ms | $0.00023820 |  |
| planning | above_authority_rate | Tree of Thoughts | True | 9 | 8054 | 1026 | 9080 | 11520.723 ms | $0.00182370 |  |
| planning | above_authority_rate | LATS ungrounded | True | 2 | 1844 | 85 | 1929 | 1616.633 ms | $0.00032760 |  |
| planning | above_authority_rate | LATS grounded | True | 2 | 1841 | 77 | 1918 | 1522.242 ms | $0.00032235 |  |
| self_correction | above_authority_rate | Self-Refine | True | 2 | 1863 | 71 | 1934 | 1616.130 ms | $0.00032205 |  |
| self_correction | above_authority_rate | Reflexion | True | 1 | 1005 | 48 | 1053 | 1194.490 ms | $0.00017955 |  |
| self_correction | severe_hold_sales_rep | Self-Refine | True | 2 | 2295 | 198 | 2493 | 2340.543 ms | $0.00046305 |  |
| self_correction | severe_hold_sales_rep | Reflexion | True | 1 | 1096 | 48 | 1144 | 650.145 ms | $0.00019320 |  |
