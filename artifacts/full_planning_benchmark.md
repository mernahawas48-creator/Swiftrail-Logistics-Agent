# Full Planning Cost / Quality Benchmark

## Aggregate comparison

| Method | Success | Avg. LLM calls | Avg. tokens | Avg. latency | Est. cost/run | Avg. tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Decomposition-first | 2/2 | 15 | 5495 | 14.131 ms | $0.001225 | 5 |
| Dynamic decomposition | 1/2 | 10.5 | 5674 | 45.374 ms | $0.000999 | 3 |
| Plan-and-Solve | 1/2 | 1 | 226.5 | 0.124 ms | $0.000039 | 0 |
| Tree of Thoughts | 2/2 | 9 | 2386 | 1.352 ms | $0.000467 | 0 |
| LATS ungrounded | 0/2 | 2 | 570.5 | 0.360 ms | $0.000120 | 0 |
| LATS grounded | 2/2 | 4 | 1032.5 | 0.660 ms | $0.000200 | 0 |
| Self-Refine | 1/2 | 2 | 533.5 | 1.177 ms | $0.000092 | 0 |
| Reflexion | 2/2 | 3 | 560 | 0.431 ms | $0.000108 | 0 |

## Per-case runs

| Group | Case | Method | Success | LLM calls | Tokens | Latency | Cost | Tool calls |
|---|---|---|---:|---:|---:|---:|---:|---:|
| decomposition | stable_minor_hold | Decomposition-first | True | 15 | 5327 | 7.382 ms | $0.001182 | 5 |
| decomposition | stable_minor_hold | Dynamic decomposition | False | 14 | 7497 | 46.193 ms | $0.001306 | 3 |
| decomposition | severe_hold_sales_rep | Decomposition-first | True | 15 | 5663 | 20.879 ms | $0.001268 | 5 |
| decomposition | severe_hold_sales_rep | Dynamic decomposition | True | 7 | 3851 | 44.555 ms | $0.000691 | 3 |
| planning | stable_minor_hold | Plan-and-Solve | True | 1 | 234 | 0.131 ms | $0.000042 | 0 |
| planning | stable_minor_hold | Tree of Thoughts | True | 9 | 2434 | 1.336 ms | $0.000478 | 0 |
| planning | stable_minor_hold | LATS ungrounded | False | 2 | 582 | 0.377 ms | $0.000123 | 0 |
| planning | stable_minor_hold | LATS grounded | True | 4 | 1053 | 0.667 ms | $0.000205 | 0 |
| planning | above_authority_rate | Plan-and-Solve | False | 1 | 219 | 0.117 ms | $0.000036 | 0 |
| planning | above_authority_rate | Tree of Thoughts | True | 9 | 2338 | 1.367 ms | $0.000456 | 0 |
| planning | above_authority_rate | LATS ungrounded | False | 2 | 559 | 0.344 ms | $0.000117 | 0 |
| planning | above_authority_rate | LATS grounded | True | 4 | 1012 | 0.653 ms | $0.000196 | 0 |
| self_correction | above_authority_rate | Self-Refine | True | 2 | 508 | 1.938 ms | $0.000089 | 0 |
| self_correction | above_authority_rate | Reflexion | True | 3 | 545 | 0.410 ms | $0.000101 | 0 |
| self_correction | severe_hold_sales_rep | Self-Refine | False | 2 | 559 | 0.417 ms | $0.000096 | 0 |
| self_correction | severe_hold_sales_rep | Reflexion | True | 3 | 575 | 0.452 ms | $0.000115 | 0 |

## Required-case checks

- static_favored_case_present: PASS
- dynamic_divergence_case_present: PASS
- linear_case_favors_ps: PASS
- lookahead_case_needs_search: PASS
- ungrounded_lats_misses_failure: PASS
- reflexion_cross_trial_case: PASS

This benchmark executes the repository's real decomposition, PS, ToT, LATS, Self-Refine, and Reflexion loops with deterministic scripted model responses and fixed Swiftrail seed-data-shaped snapshots. Token counts are a stable local proxy; production provider latency/billing can be measured separately without changing the fixed cases.
