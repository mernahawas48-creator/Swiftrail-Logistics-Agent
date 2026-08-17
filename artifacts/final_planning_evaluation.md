# Final Planning Evaluation

Evaluation mode: deterministic offline evaluation without API keys.

## Decomposition-First vs Dynamic

| Case | Preferred | Diverged |
|---|---|---:|
| stable_evidence | decomposition_first | False |
| severe_hold_discovered | dynamic | True |

## PS vs ToT vs LATS Routing

| Case | Expected | Selected | Correct |
|---|---|---|---:|
| linear_evidence_synthesis | plan_and_solve | plan_and_solve | True |
| compare_resolution_alternatives | tree_of_thoughts | tree_of_thoughts | True |
| high_stakes_final_decision | lats | lats | True |

## Grounded vs Ungrounded

| Case | Ungrounded bad plan | Grounded bad plan | Grounded good plan |
|---|---:|---:|---:|
| severe_hold_sales_rep | True | False | True |
| above_authority_rate | True | False | True |

## Self-Refine vs Reflexion

| Case | Method | Success | LLM Calls | Total Tokens | Latency ms |
|---|---|---:|---:|---:|---:|
| severe_hold_sales_rep | Self-Refine (grounded) | True | 2 | 311 | 0.74 |
| severe_hold_sales_rep | Reflexion (grounded) | True | 3 | 318 | 0.33 |
| above_authority_rate | Self-Refine (grounded) | True | 2 | 271 | 0.25 |
| above_authority_rate | Reflexion (grounded) | True | 3 | 294 | 0.26 |

## Final Checks

- decomposition_cases_pass: PASS
- all_router_cases_pass: PASS
- all_grounding_cases_pass: PASS
- all_self_correction_cases_pass: PASS

## Note

Real provider latency, token usage, and billing must come from a live model run. This offline evaluation does not invent production metrics.
