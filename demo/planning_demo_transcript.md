# Planning Demo Transcript

This transcript is generated from `python -m planning_eval.full_benchmark` using the fixed Swiftrail seed-data-shaped cases. It is deterministic and requires no API key.

## 1. Same request type: decomposition-first vs dynamic

### Stable blocked shipment — decomposition-first

Request: Review blocked shipment 2/customer 2 and resolve financial blockers safely.

Execution batches:

```text
[["t1", "t2", "t3", "t4", "t5"], ["t6"], ["t7"], ["t8"]]
```

Tool sequence:

```text
['get_shipment_status', 'search_customer', 'list_customer_invoices', 'list_customer_credit_holds', 'get_shipment_rate_exception']
```

Reasoning routes observed in the DAG:

```text
['plan_and_solve', 'tree_of_thoughts', 'lats']
```

Grounded result: `True`.

### Stable blocked shipment — dynamic

The dynamic planner stopped after:

```text
['get_shipment_status', 'search_customer', 'list_customer_credit_holds']
```

Its final candidate was rejected by the grounded validator because required evidence was still missing:

```text
['Blocked shipment plan is missing required observations: check_invoices, check_rate_exception.']
```

This is the fixed case that favors decomposition-first: when the required reconnaissance is known and stable, committing the complete DAG avoids premature stopping.

### Severe hold — real divergence

Request: Resolve blocked shipment 3/customer 3 for a sales representative.

Decomposition-first committed this complete read sequence before seeing any result:

```text
['get_shipment_status', 'search_customer', 'list_customer_invoices', 'list_customer_credit_holds', 'get_shipment_rate_exception']
```

Dynamic decomposition instead moved the credit-hold lookup earlier and observed:

```text
['get_shipment_status', 'search_customer', 'list_customer_credit_holds']
```

Forced step(s): `[4]`.

At that point the severe active hold triggers the deterministic safety branch, so the next step becomes finance-manager escalation instead of blindly continuing the remaining up-front sequence. Grounded result: `True`.

## 2. Plan-and-Solve, Tree of Thoughts, and LATS

Case: 25% pending rate exception on shipment 5 for a `sales_rep`.

### Plan-and-Solve

```text
ACTION: approve_rate_exception exception_id=2
```

Grounded success: `False`.

### Tree of Thoughts

The beam search generates competing branches, evaluates them, prunes the unsafe direct-approval path, and returns:

```text
ACTION: check_shipment
ACTION: check_customer
ACTION: check_rate_exception
ACTION: escalate role=finance_manager
```

Grounded success: `True`.

### LATS — ungrounded environment

The randomized environment reports search success on the first branch, but the real validator re-checks the selected output:

```text
ACTION: approve_rate_exception exception_id=2
```

Search reported success: `True`.
Real grounded success: `False`.

This is the required failure that an ungrounded evaluator misses.

### LATS — grounded environment

The real Swiftrail validator rejects the unauthorized direct approval, LATS records a branch reflection, explores the safer branch, and returns:

```text
ACTION: check_shipment
ACTION: check_customer
ACTION: check_rate_exception
ACTION: escalate role=finance_manager
```

Grounded success: `True`.

## 3. Self-Refine vs Reflexion

Case: severe active credit hold for a sales representative.

### Self-Refine

Draft:

```text
ACTION: check_shipment
ACTION: check_customer
ACTION: check_invoices
ACTION: check_credit_hold
ACTION: release_credit_hold hold_id=2
ACTION: release_shipment
```

Critique:

```text
The draft violates or incompletely covers the grounded Swiftrail authority and observation requirements.
```

Single revision:

```text
ACTION: check_customer
ACTION: check_credit_hold
ACTION: escalate role=finance_manager
```

Grounded success after the one allowed revision: `False`.

### Reflexion

Trial 1 success: `False`.

Stored reflection:

```text
I must carry the grounded authority and missing-observation failures into the next full trial before proposing any write.
```

Trial 2 receives that episodic reflection and produces:

```text
ACTION: check_shipment
ACTION: check_customer
ACTION: check_invoices
ACTION: check_credit_hold
ACTION: check_rate_exception
ACTION: escalate role=finance_manager
```

Trial 2 success: `True`.

This is the fixed case where one Self-Refine revision is insufficient but Reflexion succeeds by carrying a grounded lesson across trials.

## 4. Cost / quality summary

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

The benchmark uses the actual repository algorithms with deterministic scripted model responses. Token counts use a fixed local token proxy, latency is measured locally, and cost uses the repository's explicit illustrative accounting rates ($0.15/M input tokens and $0.60/M output tokens). No production-provider billing is claimed.
