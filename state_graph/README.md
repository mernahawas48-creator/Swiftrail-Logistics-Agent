# Swiftrail State Graphs

The final-project runtime extends the existing MCP server, MySQL database,
Mistral generation layer, and RAG collection. It does not create a parallel
backend.

## Shared core

`state_graph/core/` is graph-agnostic. It provides:

- validated graph definitions with branches and cycles;
- a shared serializable state envelope;
- durable MySQL checkpoints and SQLite test checkpoints;
- node-execution receipts for crash recovery and idempotent replay;
- an explicit `HITLNode` and persisted admin decisions;
- failure tickets with `open -> investigating -> resolved` transitions;
- separate external-wait, HITL, and unplanned-failure paths;
- a `GraphService` facade for the agent and platform teams.

Production state is stored in the existing Swiftrail MySQL database. Apply
`db/migrations/001_state_graph_and_delivery_recovery.sql` once to an existing
development database. Unit tests use temporary SQLite files.

## Graph 1 — Delivery Exception Recovery & Customer Rerouting

Graph 1 handles a shipment in `delivery_exception` state. It builds a recovery
plan, retrieves the authorized rerouting policy, creates a persisted recovery
case, and waits for a real customer choice. Rejected options cycle back to
option generation. Risky choices pause at an admin HITL node. MCP/DB/model/RAG
failures create tickets and resume from the failed node after resolution.

### LLM additions

1. **Task decomposition:** `decompose_recovery_plan` calls Mistral and validates
   the returned ordered steps, customer question, and policy query.
2. **RAG:** `retrieve_rerouting_policy` calls the existing Hybrid RAG pipeline
   against `delivery_exception_policy`; the graph refuses unverified evidence.

### Main transitions

```text
load_shipment -> validate_delivery_exception -> decompose_recovery_plan
-> retrieve_rerouting_policy -> create_recovery_case
-> generate_recovery_options -> wait_for_customer
```

```text
customer rejects -> generate_recovery_options -> wait_for_customer
customer chooses safe option -> apply_reroute -> verify -> complete
customer chooses risky option -> wait_for_admin -> apply_admin_decision
admin rejects -> generate_recovery_options
admin approves -> apply_reroute -> verify -> complete
```

## Graph 2 — Rate Exception Approval & Recovery

Graph 2 remains under `state_graph/graph_2/`. Its business nodes are owned by
Person 2. It should be migrated to the shared store and engine instead of
maintaining a graph-specific checkpoint implementation.

## Tests

```powershell
python -m pytest state_graph/tests -q
python -m ruff check state_graph mcp_server
```

The core tests prove transition validation, durable checkpoint reopening,
external wait/resume, HITL persistence, strict ticket lifecycle, failed-node
resume, and execution-receipt replay without repeating a completed node.

## Graph 1 live CLI

After applying the migration, indexing the RAG policy, and starting the MCP
server, use the persistent CLI to start or resume the same run from separate
processes. This makes crash-and-resume observable without the platform UI:

```powershell
python -m state_graph.graph_1.cli start --shipment-id 6 --session-id graph1-demo-001 --employee-id 1 --failure-reason "Customer was unavailable at the delivery destination."
python -m state_graph.graph_1.cli status --run-id <RUN_ID>
python -m state_graph.graph_1.cli customer --run-id <RUN_ID> --choice-json '{"action":"reroute","new_destination":"Giza Warehouse","destination_verified":false,"estimated_cost":700}'
python -m state_graph.graph_1.cli hitl-tasks
python -m state_graph.graph_1.cli resolve-hitl --task-id <TASK_ID> --approve --note "Finance manager approved the verified recovery plan." --admin-employee-id 3
```

Failure tickets can be viewed and moved through their strict lifecycle with
`tickets`, `investigate-ticket`, and `resolve-ticket`. Resolving a ticket
resumes the graph from its persisted failed node.
