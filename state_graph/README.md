# State Graph 2 — Rate Exception Approval & Recovery

## Problem
A shipment can carry a pending rate exception. Discounts at or below the delegated 15% authority can be resolved automatically; an above-authority discount must wait for a finance-manager decision. The run also needs durable recovery when an MCP/RAG step fails.

## Graph
`START -> load_shipment -> load_rate_exception -> retrieve_policy -> classify_authority`

- `auto_approve -> complete -> END`
- `wait_for_admin -> apply_admin_decision -> complete -> END`
- `failure_ticket -> resume_from_checkpoint -> failed node`

## LLM additions
1. **RAG** in `retrieve_policy`: retrieves the existing `rate_exception_policy` corpus.
2. **Constrained ReAct** in `classify_authority`: the decision is restricted to registered MCP tools and the 15% authority boundary.

## Integration
Graph 2 can run in two modes:

- Offline/unit-test mode uses the existing Python MCP handlers directly.
- Live mode (`RateExceptionGraph(live_mcp=True)`) uses the existing `agent/client.py` MCP client against the live Streamable HTTP server. It authenticates the session and calls `get_shipment_status`, `get_shipment_rate_exception`, and `approve_rate_exception` through MCP.

For an above-authority decision collected by the platform, `approve_rate_exception` accepts an optional validated `decision` payload. If omitted, the existing MCP elicitation path remains available.

## Recovery
Every meaningful transition is saved in SQLite. HITL requests and failure tickets are persisted separately. A resolved failure resumes from the exact failed node.
