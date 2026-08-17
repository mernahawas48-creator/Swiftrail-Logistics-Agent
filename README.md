# Swiftrail MCP Agent with Memory, RAG, and Planning

## Problem Statement

Enterprise logistics agents operate across long, multi-step conversations involving customers, shipments, invoices, credit holds, and rate exceptions. As these interactions grow, the agent may lose important context, repeat previous work, retrieve irrelevant information, or rely on outdated facts.

A basic conversation history is not sufficient for this environment. The system must distinguish between temporary working context, important past events, and stable long-term knowledge. It must also prevent unverified, conflicting, or expired information from being stored and reused in future decisions.

Memory and retrieval solve recall and evidence access, but they do not solve a different operational problem: deciding **what to do and in what order** when one request spans several dependent tools, policies, and authority constraints. A blocked shipment can require checking shipment state, customer exposure, overdue invoices, credit-hold severity, rate-exception status, employee role, and policy authority before the agent can decide which action is safe. An early result can invalidate later steps, so a single tool call or one fixed LLM response is not enough.

This project extends the Swiftrail Logistics MCP agent with structured Memory and Retrieval-Augmented Generation, and adds a separate planning layer for complex multi-step requests. The system combines short-term, episodic, and semantic memory, context-management strategies, metadata-aware retrieval, evidence verification, task decomposition, planning search, and grounded validation.

The objective is to build an agent that can preserve relevant context across interactions, retrieve accurate domain knowledge efficiently, and safely plan multi-step operational work without relying on unsupported, stale, or unauthorized assumptions.


## Planning and Decomposition System

### Planning Problem

A real planning problem in Swiftrail is **resolving a shipment that is blocked by multiple financial or authority constraints**. An employee may ask the agent to review the shipment, identify every blocker, determine the safest order for resolving them, perform only actions allowed by the current role, and escalate anything that requires higher authority.

A representative request is:

> Review shipment 500 and customer 3. Identify the financial blockers, determine the safest resolution sequence, perform only the actions permitted by my authority, and escalate anything that cannot be resolved safely.

No single MCP tool can safely resolve this request. The agent may need to inspect the shipment, customer account, overdue invoices, credit-hold state, rate exception, current employee role, and applicable policy before choosing the next action. The result of an early step can change the rest of the plan. For example, discovering a severe credit hold while the current user is a `sales_rep` should redirect the plan toward finance-manager escalation instead of blindly attempting protected downstream actions.

This makes the problem suitable for decomposition and planning because it has:

- **Dependent sub-tasks:** later actions depend on facts discovered earlier.
- **Real branching:** different credit, invoice, role, or exception states lead to different resolution paths.
- **Dynamic replanning:** a failed or unauthorized action can make part of the original plan stale.
- **High cost of a wrong plan:** an unsafe sequence can attempt unauthorized financial writes, waste tool calls, or incorrectly allow a blocked shipment to proceed.
- **Groundable decisions:** candidate plans can be checked against real MCP tool results, database state, and business rules instead of relying only on model self-judgment.

### Planning Approach

The planning extension is kept separate from the existing Memory/RAG path and reuses the same `mcp_server/` and `db/`. Complex requests are decomposed into dependent sub-tasks, then reasoning-heavy sub-tasks are routed to the planning method that best fits their shape:

```text
Complex Swiftrail request
        |
        v
DAG decomposition
        |
        v
Sub-task planning router
   |        |        |
   v        v        v
Plan-and-  Tree of   LATS
Solve      Thoughts  + grounded feedback
   \        |        /
        validated result
```

- **Plan-and-Solve** is used for mostly linear reasoning where one explicit plan is sufficient.
- **Tree of Thoughts** is used when several plausible reasoning paths need lookahead and pruning before committing.
- **LATS** is used for high-stakes branching tasks when an external grounded validator is available.
- **Decomposition-first and dynamic decomposition** are both applied to the same request type so the system can compare a fixed up-front plan with a plan that changes after new observations.
- **Self-Refine and Reflexion** provide local revision and cross-trial correction for outputs that fail their evaluation criteria.

The reference planning toolkit is adapted rather than reimplemented from scratch, while Swiftrail-specific prompts, routing, MCP/database grounding, and evaluation are added around the provided algorithm interfaces.


### Reference Toolkit and Planning Concern Map

The planning implementation is adapted from the required reference toolkit:

```text
https://github.com/AmrSheta22/task_decomposition_and_planning
```

The grader can locate each required planning concern directly:

| Concern | Main implementation |
|---|---|
| DAG construction, acyclicity, topological execution | `planning/models.py`, `planning/algorithms/decomposition.py` |
| Decomposition-first vs. dynamic/interleaved planning | `planning/algorithms/decomposition.py`, `planning/algorithms/dynamic_decomposition.py`, `planning/divergence.py` |
| Plan-and-Solve | `planning/algorithms/plan_and_solve.py` |
| Tree of Thoughts + beam/BFS-style pruning | `planning/algorithms/tree_of_thoughts.py` |
| LATS / MCTS + branch reflection + backpropagation | `planning/algorithms/lats.py` |
| PS / ToT / LATS routing | `planning/planning_router.py` |
| Grounded external feedback | `planning/algorithms/environment.py`, `planning/swiftrail_validator.py` |
| Self-Refine | `planning/algorithms/self_refine.py` |
| Reflexion + capped episodic reflection memory | `planning/algorithms/reflexion.py`, `planning/episodic_buffer.py` |
| Real MCP execution + post-write verification | `planning/execution_adapter.py`, `planning/action_executor.py` |
| Planning-agent orchestration | `planning/orchestrator.py`, `planning/run_swiftrail.py` |
| Fixed planning evaluation and artifacts | `planning_eval/`, `artifacts/` |

### Current Offline Planning Validation

The deterministic planning harness was rerun without API keys. The planning test suite passed **20/20 tests**, including DAG cycle rejection, decomposition divergence, PS/ToT/LATS routing, grounded validation, safe action verification, Self-Refine, and Reflexion memory carry-over.

The current offline evaluation produced these checks:

| Check | Result |
|---|---|
| Decomposition-first stable case | PASS — no divergence |
| Dynamic severe-hold case | PASS — divergence detected |
| Linear reasoning routed to Plan-and-Solve | PASS |
| Branching reasoning routed to Tree of Thoughts | PASS |
| High-stakes grounded reasoning routed to LATS | PASS |
| Ungrounded severe-hold plan accepted by baseline | YES |
| Same severe-hold plan rejected by grounded validator | YES |
| Self-Refine grounded revision | PASS |
| Reflexion cross-trial reflection carry-over | PASS |

For the severe-credit-hold case, the ungrounded baseline accepted the unsafe candidate, while the grounded validator rejected it with score **0.4** because a `sales_rep` cannot release a severe hold, finance-manager escalation was missing, `check_rate_exception` was missing, and the shipment remained blocked by the hold.

Current scripted self-correction measurements:

| Case | Method | Success | LLM calls | Total tokens |
|---|---|---:|---:|---:|
| `severe_hold_sales_rep` | Self-Refine (grounded) | 100% | 2 | 327 |
| `severe_hold_sales_rep` | Reflexion (grounded) | 100% | 3 | 326 |
| `above_authority_rate` | Self-Refine (grounded) | 100% | 2 | 271 |
| `above_authority_rate` | Reflexion (grounded) | 100% | 3 | 294 |

These offline checks validate structure and deterministic behavior. Provider latency, provider token accounting, estimated production cost, and the full cross-method quality comparison must come from the final fixed-suite benchmark rather than being invented from the offline harness.


### Final Planning Cost / Quality Benchmark

The fixed benchmark executes the repository's actual decomposition, Plan-and-Solve, Tree of Thoughts, LATS, Self-Refine, and Reflexion loops against Swiftrail seed-data-shaped requests. It uses deterministic scripted model responses so the comparison is reproducible without an API key. Token counts are a fixed local proxy, latency is measured locally, and estimated cost uses the same explicit local accounting rates already used by the planning evaluator ($0.15/M input tokens and $0.60/M output tokens).

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

The fixed cases satisfy the required method-selection evidence:

- **Decomposition-first** is preferred when the required evidence checklist is stable: it completed the `stable_minor_hold` case, while the dynamic planner stopped before all required evidence was gathered and failed grounded validation.
- **Dynamic decomposition** earns its extra decision overhead on the severe-hold case: after three real-shaped reads, the deterministic severe-hold rule forces finance-manager escalation instead of continuing the original sequence.
- **Plan-and-Solve** is the cheapest correct choice for the linear stable case (1 LLM call).
- **Tree of Thoughts** is selected for lookahead: on the 25% rate-exception case, Plan-and-Solve failed while ToT searched alternatives and returned the safe finance-manager escalation branch.
- **Grounded LATS** succeeded on both planning cases; the randomized ungrounded LATS accepted unsafe branches that the real validator rejected.
- **Self-Refine** remains the cheaper local correction when one revision is enough. On the fixed severe-hold cross-trial case, its single revision still failed, while **Reflexion** succeeded on trial 2 by carrying the grounded failure lesson forward.

Full benchmark artifacts:

```text
artifacts/full_planning_benchmark.json
artifacts/full_planning_benchmark.md
artifacts/full_planning_benchmark_summary.json
```

Run the fixed benchmark with:

```powershell
python -m planning_eval.full_benchmark
```

The complete planning demo transcript generated from the same fixed run is:

```text
demo/planning_demo_transcript.md
```

## Memory System

### Problem

Swiftrail employees (sales reps and finance managers) work the same
customers across many separate sessions. Two things go wrong without a
memory layer:

- **Nothing persists across sessions.** A sales rep who checked on a
  customer yesterday has to re-explain the situation to the agent
  today -- the agent has no way to recall that a credit hold was
  placed, or that a rate exception was already rejected for a specific
  reason.
- **Nothing reconciles conflicting history.** A customer's standing
  changes over time (a hold gets released, then a new one gets placed
  months later). Without an explicit place these facts live and get
  updated, the agent either repeats stale information or has no
  record at all of which version is current.

Both failures are costly in this domain: approving a shipment for a
customer who is actually back on a severe credit hold, or re-approving
a rate exception that was already rejected for cause, are real
operational mistakes, not cosmetic ones.

### Architecture

```
ShortTermBuffer  --overflow-->  PromoteDropRouter --(episodic only)--> EpisodicMemory
Scratchpad (separate, survives pruning)                                     |
                                                              ConsolidationLayer (periodic)
                                                                              |
                                                                      SemanticMemory
```

| Component | Role | Swiftrail example |
|---|---|---|
| `ShortTermBuffer` | Rolling window of recent conversation turns | Last N turns of a triage call |
| `Scratchpad` | Current goal / sub-goal / working state, isolated from the buffer so pruning never destroys it | "Review open rate exceptions before approving shipment 512" survives even after the tool-call chatter that gathers each exception gets pruned |
| `PromoteDropRouter` | Decides forget vs. episodic for each turn evicted from short-term memory, with a logged reason. Never writes to semantic memory. | A credit hold placement is promoted; "good morning" is forgotten |
| `EpisodicMemory` | Durable, queryable-by-customer store of promoted events | "Credit hold placed on customer 12, severe, 90+ days overdue" |
| `ConsolidationLayer` | Separate, periodic pass over episodic memory that derives/updates semantic facts (never triggered inline by the router) | Turns repeated credit-hold episodes into a current `customer_risk_level` fact |
| `SemanticMemory` | Versioned, expiring facts with explicit conflict resolution | See conflict example below |

### A real conflict, resolved

1. Customer 12's credit hold is released → consolidation writes
   `customer_risk_level = good_standing` (version 1, active).
2. Weeks later, a new severe credit hold lands on the same customer
   (90+ days overdue) → the next consolidation pass detects that this
   contradicts the active fact.
3. Resolution: version 1 is marked `superseded` (not deleted) and
   points forward to version 2; version 2 (`high_risk`) becomes
   active, with `conflict_reason = "Superseded version 1
   ('good_standing' -> 'high_risk') based on episode 7."`
4. `fact_history()` still returns both versions, so the full timeline
   of the customer's risk status is auditable, not overwritten.

This is exercised end-to-end in `memory/demo_memory.py` and asserted in
`memory/test_consolidation.py::test_consolidation_resolves_a_real_conflict_across_two_runs`.

Facts also expire on a TTL if nothing reaffirms them (`expire_stale_facts`),
so a semantic fact that stops being reinforced by new episodes ages out
rather than staying authoritative forever.

See `memory/README.md` for exactly where each concern lives in the code.


## MCP Server

The operational layer is implemented with FastMCP and MySQL.

| MCP Concern | Implementation |
|---|---|
| Capability negotiation | The client reads the server's declared capabilities and gates protocol operations on them |
| Notifications | Authentication can change the exposed tool set and triggers `tools/list_changed` |
| Elicitation | Above-authority discounts and severe credit-hold releases require explicit human input |
| Sampling | `run_portfolio_risk_sweep` can request a narrative summary from the connected client model |
| Resources | Credit/discount authority policy is exposed as an MCP resource |
| Prompts | Parameterized rate-exception justification prompt |
| Transport | stdio for local development and Streamable HTTP for remote execution |
| Progress tracking | Portfolio risk sweep reports progress while customers are processed |
| Defensive tools | Strict Pydantic schemas, server-side validation, role/state checks, safe failures, and re-authorization before writes |

Main tools:

- `authenticate`
- `search_customer`
- `get_shipment_status`
- `list_customer_invoices`
- `approve_rate_exception`
- `release_credit_hold`
- `run_portfolio_risk_sweep`
- `list_portfolio_credit_exposure` for authorized finance-manager sessions

The MySQL schema and fixed seed data are under `db/`.

## Context Management

Four context strategies are implemented under `context_eval/strategies/`:

| Strategy | Purpose |
|---|---|
| Sliding Window | Keep the most recent messages |
| Recursive Summarization | Compress older context into a summary |
| Tool Output Masking | Mask older tool outputs while preserving the conversation |
| Zone-Based Pruning | Preserve system/important zones and recent context |

The current live `AgentLoop` uses Sliding Window. The other strategies remain independently testable.

## Agent Integration

`agent/agent_loop.py` routes requests between the selected RAG path, verified memory recall, and the existing operational path. The planning extension is designed as a separate sibling path for complex multi-step requests rather than a replacement for the Memory/RAG agent.

- policy, authority, guideline, and exact section-ID questions -> **Hybrid RAG**
- cross-session recall questions -> **verified episodic/semantic memory**
- shipment, invoice, customer, and credit operations -> **MCP operational path**
- short-term overflow -> **Promote/Drop routing into episodic memory**

The real MCP protocol lifecycle and tool execution remain in `agent/client.py`; the agent loop does not duplicate the server or database.


## RAG System

### Knowledge Corpus and Ingestion

The corpus contains six Swiftrail policy/reference documents covering credit holds, rate exceptions, portfolio risk, invoice collection, employee access, and shipment pricing. These documents contain 22 policy sections used as retrieval units.

```text
Documents
  -> validated loading
  -> section-aware chunking
  -> metadata validation
  -> embeddings
  -> Qdrant indexing
```

| Component | Configuration |
|---|---|
| Chunking | Section-aware, max 1000 characters, 120-character overlap |
| Embeddings | `BAAI/bge-small-en-v1.5` with FastEmbed |
| Vector size | 384 |
| Vector database | Qdrant |
| Similarity | Cosine |
| ANN index | HNSW |
| Metadata filtering | Role, status, document, and section metadata applied during retrieval |

### Retrieval Architectures

**Naive RAG**  
Dense query embedding -> Qdrant retrieval -> grounded generation.

**Hybrid RAG**  
Dense vector retrieval + BM25 lexical retrieval -> Reciprocal Rank Fusion (RRF). Exact identifiers such as `RE-2` are handled explicitly.

**Agentic RAG**  
Plan -> retrieve -> grade evidence -> rewrite/retrieve again when evidence is incomplete -> accumulate evidence -> grounded generation. The controller is capped at two retrieval attempts.

### Self-RAG Verification

Both RAG and memory recall use explicit verification:

1. check retrieved evidence for relevance;
2. generate only from the retrieved evidence;
3. validate citations and factual support;
4. check numeric claims against evidence or scenario values;
5. return a safe abstention when verification fails.

This prevents unsupported or unauthorized information from being returned as a confident answer.

### Retrieval-Level Evaluation

Dense and hybrid retrieval were evaluated on 28 fixed retrieval cases.

| Retrieval Method | Hit@1 | MRR@5 | Access Safety@5 |
|---|---:|---:|---:|
| Dense Retrieval | 92.31% | 94.36% | 100% |
| Hybrid Retrieval | 100% | 100% | 100% |

### End-to-End Architecture Comparison

Naive, Hybrid, and Agentic RAG were evaluated on the same fixed 10 Swiftrail questions using `gemini-3.5-flash-lite`.

| Architecture | Correct / Total | Accuracy | Avg. Input Tokens | Avg. Output Tokens | Avg. Total Tokens | Avg. Latency | Avg. Retrieval Attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive RAG | 7/10 | 70.0% | 285.8 | 41.0 | 326.8 | 0.610s | 1.00 |
| Hybrid RAG | 9/10 | 90.0% | 261.3 | 40.0 | 301.3 | 0.501s | 1.00 |
| Agentic RAG | 9/10 | 90.0% | 640.9 | 41.9 | 682.8 | 0.546s | 1.40 |

The fixed set covers semantic questions, exact policy identifiers, multi-section questions, an authorization-sensitive case, and an unsupported-information case.

### Selected Architecture

**Hybrid RAG is the final retrieval architecture used by the live agent.**

Hybrid and Agentic RAG both reached **90% accuracy**, but Hybrid used fewer tokens, lower average latency, and one retrieval attempt per query. This matches Swiftrail's common mix of semantic policy questions and exact policy identifiers without paying the extra cost of iterative retrieval on every request.

Agentic RAG is retained for complex multi-part cases. In the discount-and-severe-hold case, the first retrieval missed `CH-3`; Agentic RAG detected the missing policy facet, rewrote the query, performed a second retrieval, accumulated `RE-2`, `RE-4`, and `CH-3`, and answered successfully. Hybrid RAG did not solve that case.

Detailed results:

```text
retrieval_eval/results/architecture_comparison.json
retrieval_eval/results/architecture_comparison.md
```

---

## Project Structure

```text
agent/              Agent loop, routing, MCP client, sessions
context_eval/       Context-management strategies and tests
db/                 MySQL schema, seed data, ERD
demo/               Captured MCP and RAG/Self-RAG demo evidence
mcp_server/         FastMCP server, schemas, tools, resources, prompts
memory/             Short-term, episodic, semantic memory and verified recall
planning/           Decomposition, PS/ToT/LATS planning, self-correction, and grounding interfaces
planning_eval/      Fixed planning, grounding, self-correction, and comparison evaluation
rag/                Corpus, ingestion, vector store, RAG architectures, verification
retrieval_eval/     Fixed end-to-end architecture evaluation
```

## Setup

### Install dependencies

```powershell
pip install -r mcp_server\requirements.txt
pip install -r agent\requirements.txt
pip install -r rag\embeddings\requirements.txt
pip install -r rag\vector_store\requirements.txt
pip install pytest
```

### Configure MySQL

Create a local database named `swiftrail_db`, then run:

```text
db/schema.sql
db/seed.sql
```

Copy `mcp_server/.env.example` to `mcp_server/.env` and set the local database credentials.

### Configure Gemini

Create a root `.env` file:

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Do not commit real credentials.

### Start Qdrant and ingest the corpus

```powershell
docker compose -f rag\vector_store\docker-compose.yml up -d
python -m rag.ingestion.pipeline --recreate
```

## Running

MCP server with stdio:

```powershell
python mcp_server\server.py
```

Streamable HTTP:

```powershell
python mcp_server\server.py --http
```

MCP demo:

```powershell
python agent\demo.py --transport stdio
```

Memory demo:

```powershell
python -m memory.demo_memory
```

RAG examples:

```powershell
python -m rag.naive_rag.cli --query "Who can release a severe credit hold?" --role finance_manager
python -m rag.hybrid_rag.cli --query "RE-2" --role finance_manager
python -m rag.agentic_rag.cli --query "An 18 percent discount is requested for a customer with a severe credit hold. Who must approve the discount, who may release the hold, and does discount approval release the hold?" --role finance_manager --top-k 5 --max-attempts 2
```

Architecture evaluation:

```powershell
python -m retrieval_eval.evaluate_architectures
```

Do not edit `retrieval_eval/questions.json` between architecture runs.


Planning Agent — decomposition-first:

```powershell
python -m planning.run_swiftrail --shipment-id 3 --customer-id 3 --employee-id 1
```

Planning Agent — dynamic/interleaved decomposition:

```powershell
python -m planning.run_swiftrail --shipment-id 3 --customer-id 3 --employee-id 1 --method dynamic
```

The planning agent is a separate sibling agent. It authenticates through the existing `agent/client.py`, reuses the same MCP server and MySQL database, and does not replace the Memory/RAG agent path.

Offline planning evaluation:

```powershell
python -m planning_eval.final_evaluation
```

Full planning cost/quality benchmark and generated demo transcript:

```powershell
python -m planning_eval.full_benchmark
```

The generated planning-evaluation artifacts are:

```text
artifacts/final_planning_evaluation.json
artifacts/final_planning_evaluation.md
```

## Tests

```powershell
python -m pytest memory -q
python -m pytest context_eval -q
python -m pytest rag\tests -q
python -m pytest retrieval_eval\test_evaluate_architectures.py -q
python -m pytest agent -q
python -m pytest planning -q
python -m pytest planning_eval -q
```

Integration tests that depend on MySQL or Qdrant require those services to be running.

## Demo Evidence

The captured protocol demo and RAG/Self-RAG evidence are documented in:

```text
demo/demo_transcript.md
```

The complete planning demo transcript is documented in:

```text
demo/planning_demo_transcript.md
```


Planning evidence currently generated by the fixed offline harness is stored under:

```text
artifacts/final_planning_evaluation.json
artifacts/final_planning_evaluation.md
artifacts/severe_hold_sales_rep_grounding.json
artifacts/severe_hold_sales_rep_self_refine.json
artifacts/severe_hold_sales_rep_reflexion.json
artifacts/above_authority_rate_grounding.json
artifacts/above_authority_rate_self_refine.json
artifacts/above_authority_rate_reflexion.json
```

A final planning demo should show the same real request decomposed both ways, the divergence point, one routed sub-task for each of Plan-and-Solve / Tree of Thoughts / LATS, a Self-Refine revision, Reflexion memory carried into the next trial, and the grounded validator rejecting a failure that an ungrounded critique accepts.

This requirement is now captured in `demo/planning_demo_transcript.md`, generated by the fixed full benchmark.

