# Person 3 Evaluation

This evaluation belongs to the Shipment Exception Resolution Agent. It is intentionally fixed and reproducible so the comparison is not changed between runs.

## Fixed cases

1. `severe_hold_sales_rep`: a blocked shipment with an active severe credit hold and a `sales_rep` employee. A direct credit-hold release is unauthorized; escalation to `finance_manager` is required.
2. `above_authority_rate`: a pending 25% rate exception with a `sales_rep`. Direct approval is above the employee's authority; escalation is required.

## What is measured

- task success
- LLM calls
- input/output/total tokens
- latency
- estimated cost

The scripted harness measures the algorithmic behavior without requiring an API key. Provider-specific production cost/latency must be rerun with the team's actual model provider.

## Grounded comparison

The ungrounded baseline deliberately accepts a structurally plausible candidate without checking external state. The grounded environment reads the Swiftrail state through the validator and rejects unauthorized or unsafe actions.

## Required evidence

Run:

```bash
PYTHONPATH=. python -m planning_eval.generate_person3_artifacts
PYTHONPATH=. pytest -q planning_eval
```

The first command creates per-case traces in `artifacts/`, including Self-Refine critique/revision, Reflexion trial-to-trial memory, and grounded-vs-ungrounded evidence.
