# Full Planning Cost / Quality Benchmark

## Aggregate comparison

| Method | Success | Avg. LLM calls | Avg. tokens | Avg. latency | Est. cost/run | Avg. tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Decomposition-first | 2/2 | 15 | 5508 | 3.984 ms | $0.001227 | 5 |
| Dynamic decomposition | 1/2 | 10.5 | 5689 | 5.550 ms | $0.001001 | 3 |
| Plan-and-Solve | 1/2 | 1 | 226.5 | 0.081 ms | $0.000039 | 0 |
| Tree of Thoughts | 2/2 | 9 | 2386 | 0.849 ms | $0.000467 | 0 |
| LATS ungrounded | 0/2 | 2 | 570.5 | 0.231 ms | $0.000120 | 0 |
| LATS grounded | 2/2 | 4 | 1032.5 | 0.387 ms | $0.000200 | 0 |
| Self-Refine | 1/2 | 2 | 548 | 0.358 ms | $0.000095 | 0 |
| Reflexion | 2/2 | 3 | 566.5 | 0.247 ms | $0.000109 | 0 |

## Per-case runs

| Group | Case | Method | Success | LLM calls | Tokens | Latency | Cost | Tool calls |
|---|---|---|---:|---:|---:|---:|---:|---:|
| decomposition | stable_minor_hold | Decomposition-first | True | 15 | 5327 | 4.462 ms | $0.001182 | 5 |
| decomposition | stable_minor_hold | Dynamic decomposition | False | 14 | 7497 | 6.067 ms | $0.001306 | 3 |
| decomposition | severe_hold_sales_rep | Decomposition-first | True | 15 | 5689 | 3.506 ms | $0.001272 | 5 |
| decomposition | severe_hold_sales_rep | Dynamic decomposition | True | 7 | 3881 | 5.032 ms | $0.000696 | 3 |
| planning | stable_minor_hold | Plan-and-Solve | True | 1 | 234 | 0.082 ms | $0.000042 | 0 |
| planning | stable_minor_hold | Tree of Thoughts | True | 9 | 2434 | 0.845 ms | $0.000478 | 0 |
| planning | stable_minor_hold | LATS ungrounded | False | 2 | 582 | 0.246 ms | $0.000123 | 0 |
| planning | stable_minor_hold | LATS grounded | True | 4 | 1053 | 0.402 ms | $0.000205 | 0 |
| planning | above_authority_rate | Plan-and-Solve | False | 1 | 219 | 0.079 ms | $0.000036 | 0 |
| planning | above_authority_rate | Tree of Thoughts | True | 9 | 2338 | 0.853 ms | $0.000456 | 0 |
| planning | above_authority_rate | LATS ungrounded | False | 2 | 559 | 0.217 ms | $0.000117 | 0 |
| planning | above_authority_rate | LATS grounded | True | 4 | 1012 | 0.371 ms | $0.000196 | 0 |
| self_correction | above_authority_rate | Self-Refine | True | 2 | 508 | 0.464 ms | $0.000089 | 0 |
| self_correction | above_authority_rate | Reflexion | True | 3 | 545 | 0.251 ms | $0.000101 | 0 |
| self_correction | severe_hold_sales_rep | Self-Refine | False | 2 | 588 | 0.251 ms | $0.000101 | 0 |
| self_correction | severe_hold_sales_rep | Reflexion | True | 3 | 588 | 0.243 ms | $0.000117 | 0 |

## Required-case checks

- static_favored_case_present: PASS
- dynamic_divergence_case_present: PASS
- linear_case_favors_ps: PASS
- lookahead_case_needs_search: PASS
- ungrounded_lats_misses_failure: PASS
- reflexion_cross_trial_case: PASS

This benchmark executes the repository's real decomposition, PS, ToT, LATS, Self-Refine, and Reflexion loops with deterministic scripted model responses and fixed Swiftrail seed-data-shaped snapshots. Token counts are a stable local proxy; production provider latency/billing can be measured separately without changing the fixed cases.
