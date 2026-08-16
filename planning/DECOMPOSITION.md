# Decomposition & Orchestration

Owner: Person 1 — Task decomposition (both methods), DAG construction,
subtask execution adapter, static-vs-dynamic divergence, trace logging,
planning agent orchestration.

## The real problem

A sales rep or finance manager needs to review a blocked shipment: the
shipment itself, the customer's credit standing, overdue invoices, any
active credit holds, and any pending rate exception, then decide a safe
resolution sequence — without exceeding their own authority. No single MCP
tool call can answer this; it genuinely needs a DAG of lookups followed by
a reasoning step, and a wrong recommendation (e.g. releasing a severe hold
without escalating) has a real cost.

> Review shipment {id} and customer {id}. Identify the financial blockers,
> determine the safest resolution sequence, perform only the actions
> permitted by my authority, and escalate anything that cannot be resolved
> safely.

## Where each concern lives

| Concern | File | Notes |
|---|---|---|
| DAG Construction / Subtask Dependency Mapping / Acyclicity | `planning/swiftrail_subtask.py` | Reuses the toolkit's `Plan` model (cycle check happens at construction — `Plan.validate_dag`). Adds only what the toolkit didn't have: whether a node is a real MCP tool call or a reasoning node. |
| Topological Execution | `planning/algorithms/decomposition.py` → `execute_plan_swiftrail()` | Executes batch-by-batch via `Plan.execution_batches()` (toolkit-provided), `asyncio.gather` within a batch. |
| Decomposition-First | `planning/algorithms/decomposition.py` → `decompose_blocked_shipment()` | Whole DAG generated in one LLM call, grounded to the real tool catalog (not free text). |
| Dynamic / Interleaved Decomposition + Dynamic Replanning | `planning/algorithms/dynamic_decomposition.py` → `dynamic_decompose_blocked_shipment()` | One step at a time, real tool call, then decide the next step from the actual result. Includes a **hard-coded, non-LLM safety override** (`_forced_next_step`) that forces escalation the moment a severe active hold is observed — this is what makes the divergence real and reproducible, not just prompted. |
| Subtask Execution Adapter | `planning/execution_adapter.py` | Routes `tool_call` nodes to the real MCP client (`agent.client.SwiftrailAgent`); routes `reasoning` nodes to `planning_router.solve_subtask` (PS/ToT/LATS). |
| Static vs Dynamic Divergence Handling | `planning/divergence.py` | Structurally compares the two tool-call sequences and returns the exact index + reason they diverge. |
| Decomposition Trace Logging | `planning/trace_logger.py` | Writes to the same `artifacts/` folder the toolkit's `cli.py` already uses — extends the schema, doesn't duplicate the logging system. |
| Planning Agent Orchestration | `planning/orchestrator.py` | `SwiftrailPlanningOrchestrator` — one entry point: `run()` for a single method, `run_both_and_log_divergence()` for the comparison suite. |

Two new read-only MCP tools also support this DAG: `list_customer_credit_holds`
and `get_shipment_rate_exception` (`mcp_server/tools/read_tools.py`,
`mcp_server/schemas.py`) — nothing previously scoped a hold/rate-exception
lookup to a single customer/shipment.

`planning/swiftrail_env.py` is a **narrow placeholder** grounded
`Environment` (checks one unsafe pattern: recommending hold release without
escalation), just enough for orchestration to be runnable end-to-end. It is
not the team's full grounded-environment deliverable — see the
self-correction/grounding concern for that.

## Tests

`planning/test_swiftrail_decomposition.py` — 6 unit tests, no DB/MCP server
needed: cycle rejection at construction, metadata-completeness check,
topological batching order, divergence detection (both the diverges and
no-false-positive cases), and dependency-scoped execution context.

`planning_eval/offline_wiring_check.py` — runs decomposition-first, dynamic
decomposition, and divergence detection end-to-end against a fake LLM and
fake MCP agent (no DB, no API key). Proves the wiring itself is correct
offline; it is not the lab's required real-request comparison suite.

## Test-case prompts for the real planning_eval/ suite

Concrete, real requests to seed the fixed comparison test suite:

1. `shipment_id=500, customer_id=3` — clean case, no holds, both methods agree (baseline / no-divergence case).
2. A shipment whose customer has one **minor** hold only — both methods proceed normally, no escalation.
3. A shipment whose customer has one **severe, active** hold — dynamic should diverge (escalate early); decomposition-first still runs its full read-only reconnaissance before its terminal task notices the same fact late.
4. A shipment with a **pending rate exception above 15%** but no credit hold — tests the `fetch_rate_exception` branch independent of the hold branch.
5. A shipment where the session role is `sales_rep` and the customer has a severe hold — tests that the terminal/escalation output correctly defers to `finance_manager`, not just flags the hold.
6. A shipment/customer pair with **both** a severe hold and an above-authority discount — the "everything blocked" case, good for showing LATS-worthy branching.
7. A nonexistent shipment id — tests that a tool_call failure (`SHIPMENT_NOT_FOUND`) surfaces cleanly through the adapter instead of the DAG silently continuing.
8. A customer with a **released** (not active) hold plus a new active one — tests that `list_customer_credit_holds` filtering (`active_holds`) is actually being read, not just the raw `holds` list.

## Open items

- Wire `SwiftrailPlanningOrchestrator` into a small CLI or an `agent/agent_loop.py` routing branch (kept separate from the memory/RAG path).
- Build the real `planning_eval/` suite: run the 8 cases above through `run_both_and_log_divergence`, collect the `artifacts/` traces, produce the comparison table.
- Real token/latency counts once a live LLM/MCP server is wired (currently `llm_calls` is counted structurally, not measured from an API response).
