# Person 3 Comparison Table

Fixed cases: `severe_hold_sales_rep` and `above_authority_rate`.

| Method | Success | Avg LLM calls | Avg total tokens | Avg latency | Avg estimated cost |
|---|---:|---:|---:|---:|---:|
| Self-Refine (grounded) | 100% | 2.0 | 297.0 | local harness | $0.000057 |
| Reflexion (grounded) | 100% | 3.0 | 263.0 | local harness | $0.000053 |

The scripted evaluation is reproducible without an API key. Latency and cost are harness/provider-model estimates, not production billing numbers.

## Grounding result

| Case | Ungrounded | Grounded | Grounded evidence |
|---|---|---|---|
| Severe credit hold + sales rep | ACCEPT | REJECT | role lacks authority; finance escalation required |
| 25% rate exception + sales rep | ACCEPT | REJECT | exception exceeds sales-rep approval authority |

## Interpretation

- Self-Refine is the cheaper/faster correction scope when one draft can be repaired in one revision.
- Reflexion is appropriate when the first complete attempt fails and the next attempt needs a verbal lesson from the previous trial.
- Grounding is mandatory for current operational state because an ungrounded critic can accept a structurally plausible but unauthorized action.
