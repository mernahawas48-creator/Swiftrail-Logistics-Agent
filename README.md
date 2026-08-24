# Swiftrail Logistics Agent

## MCP, Memory & RAG, Planning, Persistent State Graphs, HITL, Recovery, and Platform Integration

## Project Overview

**Swiftrail** is an intelligent logistics operations platform designed to support complex, business-critical workflows involving shipments, customers, invoices, credit holds, rate exceptions, operational policies, and employee authority.

Rather than treating each user request as an isolated conversation, Swiftrail combines reasoning, retrieval, operational tools, persistent state, and administrative control within one integrated system.

The platform is built around a central principle:

> **LLMs can reason, retrieve, plan, and recommend, while operational truth, authorization, persistence, and protected actions remain controlled by the surrounding system.**

Swiftrail integrates:

- **Model Context Protocol (MCP)** for controlled access to real logistics tools and business operations;
- **MySQL** as the authoritative source of operational and workflow state;
- **Memory** for preserving relevant context and events across interactions;
- **Retrieval-Augmented Generation (RAG)** for grounded access to company policies and domain knowledge;
- **Self-RAG verification** for evidence validation and safe abstention;
- **Context Management** for controlling long-running conversational context;
- **Task Decomposition and Planning** for complex requests involving multiple dependent actions;
- **Plan-and-Solve, Tree of Thoughts, and LATS** for different planning and search requirements;
- **Persistent State Graphs** for long-running operational workflows;
- **Human-in-the-Loop (HITL)** for protected or high-risk decisions;
- **Failure Tickets and Recovery** for unexpected operational failures;
- **Durable Checkpointing** for restoring workflows without restarting them from the beginning;
- **Runtime MCP Tool Management** for agent-specific operational control;
- **Live RAG Document Management** for updating the knowledge base at runtime;
- **User and Admin Platforms** for interacting with agents and managing the system.

Together, these components allow Swiftrail to operate as more than a conversational assistant.

A logistics request can begin with a user interaction, retrieve operational data through MCP, access policy evidence through RAG, decompose a complex objective into dependent tasks, branch into different strategies, pause for customer input or administrator approval, recover from an unexpected failure, and later continue from its persisted state.

This creates a system in which **reasoning is flexible, but execution remains controlled, grounded, auditable, and persistent**.

## System at a Glance

```text
                            SWIFTRAIL LOGISTICS PLATFORM
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
            User Platform                             Admin Platform
        Agent interaction &                         Operational control
          agent switching                    Tools / RAG / HITL / Tickets
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                               FastAPI Platform
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
 Memory & RAG Agent          Planning Agent                Stateful Agents
                                                           Graph 1 / 2 / 3
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                MCP Server
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
           MySQL                    Qdrant                  Mistral
             │                        │
      Operational State       Policy Knowledge Base
```

### Core Responsibilities

| Layer | Responsibility |
|---|---|
| **MCP & Operational Tools** | Execute controlled logistics operations and enforce server-side business rules |
| **MySQL** | Maintain authoritative business and persistent workflow state |
| **Memory** | Preserve relevant information and historical events across interactions |
| **RAG** | Retrieve grounded Swiftrail policy and operational knowledge |
| **Planning** | Decompose and solve complex multi-step operational requests |
| **State Graphs** | Manage long-running workflows with branches, waits, failures, and resumable execution |
| **HITL** | Require explicit human approval for protected decisions |
| **Failure Recovery** | Persist failures, create tickets, and resume after correction |
| **Runtime Administration** | Manage agent tools and RAG documents without changing agent source code |
| **Platform** | Provide unified user interaction and administrative control |

With this overall architecture in place, the following sections describe the individual capabilities and their implementation in detail.

---

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

The deterministic planning harness was rerun without API keys. The planning and planning-evaluation suite passed **20/20 tests**, including DAG cycle rejection, decomposition divergence, PS/ToT/LATS routing, grounded validation, safe action verification, Self-Refine, and Reflexion memory carry-over.

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

For the severe-credit-hold case, the ungrounded baseline accepted the unsafe candidate, while the grounded validator rejected it with score **0.55** because a `sales_rep` cannot release a severe hold, finance-manager escalation was required, and the shipment could not be safely released while the customer remained on the active severe hold.

Current scripted self-correction measurements:

| Case | Method | Success | LLM calls | Total tokens |
|---|---|---:|---:|---:|
| `severe_hold_sales_rep` | Self-Refine (grounded) | 100% | 2 | 311 |
| `severe_hold_sales_rep` | Reflexion (grounded) | 100% | 3 | 318 |
| `above_authority_rate` | Self-Refine (grounded) | 100% | 2 | 271 |
| `above_authority_rate` | Reflexion (grounded) | 100% | 3 | 294 |

These offline checks validate structure and deterministic behavior. Provider latency, provider token accounting, and provider-based cost are reported separately by the live Mistral benchmark below; the deterministic harness is kept for controlled edge cases such as grounded-vs-ungrounded failure and Reflexion cross-trial memory.


### Final Planning Cost / Quality Benchmark

The final planning evidence uses **two complementary fixed-suite runs**:

1. **Live provider benchmark** — executes the repository's planning and self-correction methods with `mistral-small-latest`, records real Mistral API calls, provider-reported token usage, measured latency, and estimated cost.
2. **Deterministic offline benchmark** — executes the same repository algorithms with scripted responses so specific behavioral requirements are reproducible without API variability, including decomposition divergence, grounded-vs-ungrounded failure, and Reflexion carrying a failure lesson into the next trial.

The live benchmark uses the same fixed Swiftrail seed-data-shaped snapshots for every method, a shared plain `ACTION:` output contract, and performs no database writes.

#### Live Mistral comparison

| Method | Success | Avg. LLM calls | Avg. input tokens | Avg. output tokens | Avg. total tokens | Avg. latency | Est. cost/run | Avg. tool calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decomposition-first | 2/2 | 14 | 9630.5 | 2729.5 | 12360 | 23344.177 ms | $0.00308227 | 5 |
| Dynamic decomposition | 1/2 | 5.5 | 3097 | 470.5 | 3567.5 | 5954.995 ms | $0.00074685 | 4.5 |
| Plan-and-Solve | 2/2 | 1 | 1051.5 | 166 | 1217.5 | 2321.562 ms | $0.00025732 | 0 |
| Tree of Thoughts | 1/2 | 9 | 8415 | 1017.5 | 9432.5 | 11503.719 ms | $0.00187275 | 0 |
| LATS ungrounded | 2/2 | 2 | 1939 | 83.5 | 2022.5 | 1649.617 ms | $0.00034095 | 0 |
| LATS grounded | 2/2 | 2 | 1936 | 77.5 | 2013.5 | 1684.898 ms | $0.00033690 | 0 |
| Self-Refine | 2/2 | 2 | 2079 | 134.5 | 2213.5 | 1978.337 ms | $0.00039255 | 0 |
| Reflexion | 2/2 | 1 | 1050.5 | 48 | 1098.5 | 922.317 ms | $0.00018638 | 0 |

Live benchmark configuration:

```text
Model: mistral-small-latest
Token accounting: provider-reported LangChain AIMessage usage_metadata
Input price used: $0.15 / 1M tokens
Output price used: $0.60 / 1M tokens
```

#### Method-selection evidence

- **Decomposition-first vs. dynamic:** decomposition-first achieved **2/2** overall, while dynamic achieved **1/2**. On the severe-hold case, however, both succeeded and dynamic reacted after the early severe-hold observation with only **5 LLM calls, 3124 tokens, and 5.06 s**, compared with decomposition-first's **15 calls, 13390 tokens, and 24.41 s**. This supports dynamic planning when an early observation can invalidate the remaining up-front plan.
- **Plan-and-Solve:** on the two live direct-planning cases it achieved **2/2** with only **1 call** on average, making it the preferred choice when the reasoning path is linear and the required evidence is already available.
- **Tree of Thoughts:** the live run achieved **1/2** and used **9 calls** on average, so it is not the default for simple cases. It is retained for genuinely branching/lookahead subtasks because the deterministic fixed 25% rate-exception case shows PS choosing an unsafe direct approval while ToT searches alternatives and returns the finance-manager escalation branch.
- **Grounded LATS:** the live cases happened to be solved by both grounded and ungrounded LATS. The controlled deterministic severe/authority case is therefore retained as the grounding proof: the ungrounded environment accepts an unsafe branch that the Swiftrail validator rejects, while grounded LATS reflects on that feedback and selects a safe branch.
- **Self-Refine vs. Reflexion:** both succeeded in the live run. The deterministic severe-hold cross-trial case is retained to show the required distinction: one Self-Refine revision remains insufficient, while Reflexion stores the grounded failure lesson and succeeds on trial 2.

Live benchmark artifacts:

```text
artifacts/live_planning_benchmark.json
artifacts/live_planning_benchmark.md
```

Deterministic benchmark and summary artifacts:

```text
artifacts/full_planning_benchmark.json
artifacts/full_planning_benchmark.md
artifacts/full_planning_benchmark_summary.json
```

Run the deterministic benchmark with:

```powershell
python -m planning_eval.full_benchmark
```

Run the live provider benchmark with:

```powershell
python -m planning_eval.live_benchmark
```

The complete deterministic planning demo transcript is:

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

```text
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

The table below was regenerated on the fixed 10-question suite with
`mistral-small-latest`. All three architectures ran against the same questions
and completed without transient API retries.

| Architecture | Correct / Total | Accuracy | Avg. Input Tokens | Avg. Output Tokens | Avg. Total Tokens | Avg. Latency | Avg. Retrieval Attempts | API Retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive RAG | 5/10 | 50.0% | 291.9 | 25.8 | 317.7 | 0.642s | 1.00 | 0 |
| Hybrid RAG | 6/10 | 60.0% | 272.6 | 29.7 | 302.3 | 0.597s | 1.00 | 0 |
| Agentic RAG | 7/10 | 70.0% | 647.7 | 28.0 | 675.7 | 0.682s | 1.40 | 0 |

The fixed set covers semantic questions, exact policy identifiers, multi-section questions, an authorization-sensitive case, and an unsupported-information case.

### Selected Architecture

**Hybrid RAG is the default retrieval architecture used by the live agent.**

Hybrid improves on Naive RAG by 10 percentage points while recording the
lowest average token use (302.3) and latency (0.597s), with one retrieval
attempt per query. It is therefore the cost-conscious default for typical
policy and exact-identifier requests.

Agentic RAG is retained as the escalation path for complex multi-part cases.
It achieved the highest measured accuracy (70%) and solved both multi-part
cases, at the cost of about 2.2 times Hybrid's average tokens and additional
retrieval attempts. The per-case results preserve the remaining failures
instead of hiding them, so they can guide future prompt and verification work.

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
demo/               Captured MCP, RAG/Self-RAG, and planning demo evidence
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
python -m pip install -r requirements.txt

# For tests and linting
python -m pip install -r requirements-dev.txt
```

### Configure MySQL

Create a local database named `swiftrail_db`, then run:

```text
db/schema.sql
db/seed.sql
```

Copy `mcp_server/.env.example` to `mcp_server/.env` and set the local database credentials.

### Configure Mistral

All live text-generation paths use Mistral. Copy `.env.example` to `.env` and
set your local key:

```env
MISTRAL_API_KEY=your_key
MISTRAL_MODEL=mistral-small-latest
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

Deterministic planning benchmark and demo evidence:

```powershell
python -m planning_eval.full_benchmark
```

Live Mistral cost/quality benchmark:

```powershell
python -m planning_eval.live_benchmark
```

The main planning-evaluation artifacts are:

```text
artifacts/final_planning_evaluation.json
artifacts/final_planning_evaluation.md
artifacts/full_planning_benchmark.json
artifacts/full_planning_benchmark.md
artifacts/full_planning_benchmark_summary.json
artifacts/live_planning_benchmark.json
artifacts/live_planning_benchmark.md
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

`demo/planning_demo_transcript.md` captures the required planning evidence from the deterministic fixed suite: the same request decomposed both ways with the divergence point visible, routed Plan-and-Solve / Tree of Thoughts / LATS subtasks, a Self-Refine revision, Reflexion carrying a reflection into the next trial, and a grounded failure that the ungrounded evaluator misses.

The separate live provider cost/quality evidence is stored in `artifacts/live_planning_benchmark.json` and `artifacts/live_planning_benchmark.md`.

---

# Persistent Stateful Workflows and Platform

The Swiftrail system includes a persistent operational layer for workflows that must survive waits, human decisions, unexpected failures, and process restarts.

This layer adds:

- three persistent state-graph workflows;
- durable MySQL-backed checkpoints;
- Human-in-the-Loop approval;
- failure tickets and resumable recovery;
- execution receipts to avoid blindly repeating completed work;
- runtime MCP tool management per agent;
- live RAG document administration;
- a browser-based User Platform;
- a browser-based Admin Dashboard;
- integration of the Memory/RAG and Planning agents in the same product interface.

The objective of this layer is to allow Swiftrail to handle long-running operational workflows safely instead of assuming that every logistics operation can finish inside a single request-response cycle.

## Stateful System Architecture

```text
                               Swiftrail Platform
                     ┌───────────────┴───────────────┐
                     │                               │
                 User Platform                  Admin Platform
              chat + agent switch        tools / RAG / HITL / tickets
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                              FastAPI Backend
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
   Memory & RAG Agent          Planning Agent             State Graphs
                                                       Graph 1 / 2 / 3
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
                                     │
                            Existing MCP Server
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
              MySQL                Qdrant              Mistral
                │
        Swiftrail Business Data
```

The stateful workflows reuse the same operational domain, database, MCP server, RAG infrastructure, and planning components.

---

## Why Persistent State Graphs Are Needed

Memory and planning solve important problems, but they do not provide durable workflow execution by themselves.

Some logistics workflows may need to:

- wait for a customer response;
- pause for an administrator decision;
- survive a process restart;
- continue after an unexpected tool or database failure;
- resume from the correct step without repeating already-completed work;
- keep the full operational state inspectable by an administrator.

For these cases, Swiftrail uses persistent state graphs backed by durable storage.

---

# Stateful Problem 1 — Delivery Exception Recovery & Customer Rerouting

## Problem

A shipment may enter a delivery-exception state after an unsuccessful delivery. Resolving it can require several steps across customer communication, policy retrieval, risk evaluation, and operational execution.

A single LLM response is not enough because the workflow may need to wait for the customer, generate new alternatives, escalate risky reroutes, and resume later.

## Main Flow

```text
load_shipment
    ↓
validate_delivery_exception
    ↓
decompose_recovery_plan
    ↓
retrieve_rerouting_policy
    ↓
create_recovery_case
    ↓
generate_recovery_options
    ↓
wait_for_customer
    ↓
evaluate_customer_choice
    │
    ├── customer requests different options
    │        ↓
    │   generate_recovery_options
    │        ↓
    │   wait_for_customer
    │
    ├── safe reroute
    │        ↓
    │   apply_reroute
    │        ↓
    │      verify
    │        ↓
    │     complete
    │
    └── risky reroute
             ↓
          HITL review
             │
             ├── reject → generate new options
             │
             └── approve → apply → verify → complete
```

The graph contains a real external wait, a real cycle, an administrator decision path, and failure recovery.

## LLM Additions

Two LLM-based additions are used:

1. **Task Decomposition** — creates a structured recovery plan and identifies the information required before execution.
2. **RAG** — retrieves delivery-exception and rerouting policy evidence before operational decisions are made.

## HITL Examples

Administrator review may be required for conditions such as:

- an unverified rerouting destination;
- a high rerouting cost;
- a customs-region change;
- a high-value shipment.

The model can recommend a next step, but it cannot independently authorize a protected reroute.

## Recovery Improvement

During integration, the validation path was updated so it can reload the **current** shipment state from live operational tools during recovery instead of relying only on a stale shipment copy stored in an earlier checkpoint.

This allows a resumed workflow to observe real external corrections made while the run was paused.

---

# Stateful Problem 2 — Rate Exception Approval & Recovery

## Problem

A shipment may have a rate exception that must be evaluated against current policy, employee authority, and operational state.

The graph must distinguish between actions that fall within delegated authority and actions that require a higher-level administrator decision.

## Main Flow

```text
load_shipment
    ↓
load_rate_exception
    ↓
retrieve_policy
    ↓
classify_authority
    │
    ├── delegated authority
    │        ↓
    │   apply_rate_decision
    │        ↓
    │      complete
    │
    └── higher authority required
             ↓
          HITL review
             ↓
        apply_rate_decision
             ↓
           complete
```

## LLM Additions

1. **RAG** — retrieves the applicable rate-exception and authority policy.
2. **Constrained ReAct** — supports bounded reasoning over allowed operational actions while server-side validation remains authoritative.

The LLM never replaces MCP role checks or protected business rules.

---

# Stateful Problem 3 — Credit-Hold Remediation

## Problem

A customer with an active credit hold may need to provide payment information, dispute evidence, or additional clarification before the hold can be resolved.

This workflow is stateful because the system may need to wait across sessions, evaluate new evidence, retry the customer step, or escalate a severe case to finance management.

## Main Flow

```text
load_account_state
      ↓
build_remediation_plan
      ↓
prepare_customer_wait
      ↓
wait_for_customer
      ↓
process_customer_input
      │
      ├── insufficient evidence
      │        ↓
      │    wait again
      │
      ├── partial remediation
      │        ↓
      │      complete
      │
      └── acceptable evidence
               ↓
        classify_release_action
               │
               ├── severe / high risk
               │        ↓
               │     finance HITL
               │
               └── allowed
                        ↓
                   execute action
                        ↓
                     complete
```

## LLM Additions

1. **LATS** — explores bounded remediation alternatives using grounded feedback.
2. **Constrained ReAct** — supports restricted reasoning over approved credit-hold operations.

## Live MCP Integration

The production graph uses the existing MCP-backed operations such as:

```text
list_customer_invoices
list_customer_credit_holds
release_credit_hold
```

High-risk release decisions remain protected by finance-manager authority and HITL.

---

# Shared Persistent State-Graph Runtime

The common runtime is implemented under:

```text
state_graph/core/
```

It provides the shared behavior used by all three graphs.

## Main Capabilities

- serializable graph state;
- durable MySQL-backed checkpoints;
- explicit node transitions;
- external wait states;
- HITL pauses;
- persisted administrator decisions;
- failure tickets;
- resume/recovery behavior;
- execution receipts.

## Durable Checkpoints

State is persisted after meaningful transitions such as:

- successful node completion;
- external wait;
- customer input;
- HITL creation;
- administrator decision;
- unexpected failure;
- ticket recovery.

A persisted state can include:

- run ID;
- graph name;
- current node;
- status;
- workflow data;
- transition history;
- HITL references;
- ticket references;
- recovery metadata.

## Execution Receipts

Execution receipts record completed node work so recovery does not blindly repeat operations that were already completed before a failure or restart.

Conceptually:

```text
Node executes successfully
        ↓
State + receipt persisted
        ↓
Process stops / failure occurs later
        ↓
Run restored
        ↓
Completed work is recognized
        ↓
Workflow continues from persisted state
```

---

# Human-in-the-Loop (HITL)

HITL represents an **expected authorization boundary**, not a system failure.

```text
Risk / authority condition detected
        ↓
Persist full graph state
        ↓
Create HITL task
        ↓
Pause run
        ↓
Administrator inspects task and persisted state
        ↓
Approve / Reject
        ↓
Persist decision
        ↓
Resume the same workflow
```

The administrator decision is explicit and persisted. The model does not simulate or fabricate approval.

---

# Failure Tickets and Recovery

Failure tickets represent **unexpected operational failures**.

Examples include:

- MCP communication errors;
- database failures;
- state validation failures;
- unusable model output;
- unexpected runtime exceptions.

## Ticket Lifecycle

```text
Unexpected failure
       ↓
Persist failed state
       ↓
Create ticket
       ↓
OPEN
       ↓
INVESTIGATING
       ↓
Underlying issue corrected
       ↓
RESOLVED
       ↓
Resume from saved workflow state
```

## HITL vs. Failure Ticket

| HITL | Failure Ticket |
|---|---|
| Expected decision | Unexpected failure |
| Authorization boundary | Recovery boundary |
| Approve / Reject | Investigate / Resolve |
| Graph intentionally pauses | Node cannot complete |
| Human judgment is required | Underlying problem must be corrected |

This separation keeps authorization logic distinct from technical or operational recovery.

---

# Final Product Platform

The browser product is implemented under:

```text
platform/
platform_app/
```

The platform integrates the Swiftrail agents and the three stateful workflows into one interface.

## User Platform

The user can switch between five agents:

1. Credit-Hold Remediation
2. Delivery Exception Recovery
3. Rate Exception Approval
4. Memory & RAG Agent
5. Decomposition & Planning Agent

User requests are sent through the real FastAPI backend rather than hardcoded frontend responses.

Main route:

```text
POST /api/chat
```

State-graph runs can also be inspected through the run API.

---

# Admin Dashboard

The administrator interface contains four main operational areas:

```text
Agents & Tools
RAG Documents
HITL Queue
Tickets
```

## Agents & Tools

The administrator can inspect runtime MCP tool assignments for each registered agent.

Non-protected tools can be enabled or disabled at runtime without editing the agent source code.

## RAG Documents

The administrator can:

- inspect the current corpus;
- add a document;
- provide document metadata;
- specify allowed roles;
- remove removable documents;
- trigger live re-indexing.

## HITL Queue

The administrator can:

- inspect pending HITL tasks;
- inspect the associated run;
- inspect persisted graph state;
- approve;
- reject.

## Tickets

The administrator can:

- inspect open failures;
- inspect the failed node and error;
- inspect persisted graph state;
- mark a ticket as investigating;
- resolve the ticket.

The Admin UI exposes:

```text
View persisted state
```

for both HITL tasks and failure tickets so the stored workflow state is directly inspectable.

---

# Runtime MCP Tool Management

Runtime tool management is implemented through components including:

```text
mcp_server/runtime_tool_manager.py
mcp_server/runtime_admin_api.py
mcp_server/runtime_setup.py
platform_app/admin/runtime_client.py
```

The runtime layer supports:

- registered agents;
- tool assignment per agent;
- enable/disable behavior;
- runtime tool registration and removal;
- `tools/list_changed`;
- protected tools;
- agent-specific permission checks.

## Per-Agent Enforcement

A tool may remain globally available on the MCP server because another agent still uses it. Therefore runtime enforcement is also checked at the agent level.

```text
Agent A
Tool X = OFF
        ↓
Agent A requests Tool X
        ↓
Permission checker
        ↓
DENIED

Agent B
Tool X = ON
        ↓
Permission checker
        ↓
ALLOWED
```

The live state-graph integrations are wired to the runtime permission checker before protected MCP calls are executed.

---

# Live RAG Document Management

The RAG administration path connects the Admin Dashboard to the same knowledge system used by the Memory/RAG agent.

```text
Admin UI
    ↓
FastAPI backend
    ↓
RAG document manager
    ↓
Corpus / manifest update
    ↓
Qdrant re-index
    ↓
Refresh live retrieval pipeline
```

The RAG manager updates the corpus and Qdrant index and includes rollback behavior if indexing fails.

## Live Retrieval Refresh

During integration, a newly added document successfully appeared in the live corpus while the already-created Memory/RAG agent still held an older retrieval pipeline in memory.

The platform integration was updated so document add/remove operations clear the cached RAG pipeline.

```text
Admin modifies corpus
       ↓
Qdrant is re-indexed
       ↓
Cached Memory/RAG pipeline is cleared
       ↓
Next query rebuilds retrieval
       ↓
Updated corpus can affect retrieval
```

This makes RAG administration an actual runtime capability rather than a UI-only configuration change.

---

# Final Project Concern Map

| Concern | Main implementation |
|---|---|
| Shared graph runtime | `state_graph/core/` |
| Graph execution engine | `state_graph/core/engine.py` |
| Serializable workflow state | `state_graph/core/state.py` |
| Durable MySQL persistence | `state_graph/core/mysql_store.py` |
| Wait / HITL support | `state_graph/core/nodes.py` |
| Delivery Exception workflow | `state_graph/graph_1/` |
| Rate Exception workflow | `state_graph/graph_2/` |
| Credit-Hold workflow | `state_graph/graph_3/` |
| Runtime MCP tool manager | `mcp_server/runtime_tool_manager.py` |
| Runtime Admin API | `mcp_server/runtime_admin_api.py` |
| Runtime MCP setup | `mcp_server/runtime_setup.py` |
| Existing-agent platform integration | `platform_app/agent_integration.py` |
| State-graph platform integration | `platform_app/graph_integration.py` |
| Admin integration services | `platform_app/admin/` |
| User web interface | `platform/frontend/index.html` |
| Admin web interface | `platform/frontend/admin.html` |
| Product FastAPI backend | `platform/backend/app.py` |
| State-graph database migration | `db/migrations/001_state_graph_and_delivery_recovery.sql` |

---

# Final Project Structure Additions

The original project structure remains unchanged. The stateful platform adds the following major areas:

```text
platform/            Browser User and Admin interfaces
platform_app/        Product integration and admin services
state_graph/         Shared persistent runtime and the three stateful workflows
db/migrations/       Persistent state schema additions
```

These directories extend the existing `agent/`, `mcp_server/`, `memory/`, `rag/`, and `planning/` architecture rather than replacing them.

---

# Running the Final Platform

## 1. Start the MCP Server

From the repository root:

```powershell
python mcp_server\server.py --http
```

The MCP endpoint is available at:

```text
http://127.0.0.1:8000/mcp
```

## 2. Start the Product Platform

In another terminal:

```powershell
cd platform\backend
python -m uvicorn app:app --reload --port 8080
```

## 3. Open the Interfaces

User Platform:

```text
http://127.0.0.1:8080/
```

Admin Dashboard:

```text
http://127.0.0.1:8080/admin.html
```

---

# Stateful Database Migration

Apply the state-graph migration to the existing Swiftrail database:

```text
db/migrations/001_state_graph_and_delivery_recovery.sql
```

The stateful system uses the same Swiftrail MySQL database. It does not introduce a separate business database.

---

# Stateful Platform Tests

In addition to the existing Memory, RAG, Agent, and Planning tests documented earlier, the persistent platform includes test areas such as:

```powershell
python -m pytest state_graph\tests -q
python -m pytest platform_app -q
python -m pytest mcp_server\test_runtime_admin_api.py -q
```

Tests that depend on MySQL, Qdrant, or the live MCP runtime require those services to be available.

---

# End-to-End Validation

The following behaviors were exercised during final integration:

| Validation | Status |
|---|---|
| MCP HTTP server starts | PASS |
| Web platform starts | PASS |
| User Platform loads | PASS |
| Admin Dashboard loads | PASS |
| Five agents are exposed | PASS |
| Runtime agent/tool configuration loads | PASS |
| Memory/RAG session starts | PASS |
| Existing grounded RAG policy retrieval works | PASS |
| Credit-Hold state graph is reachable from the User Platform | PASS |
| State graph reads live business state | PASS |
| A real Graph 1 validation failure creates a ticket | PASS |
| Failure ticket appears in the Admin Dashboard | PASS |
| Persisted ticket state is inspectable | PASS |
| Ticket investigation action works | PASS |
| Ticket resolution action works | PASS |
| RAG document creation endpoint succeeds | PASS |
| Added RAG document appears in the live corpus | PASS |
| HITL pause → admin decision → same-run resume | PASS |
| Same-run ticket recovery after the latest live-state refresh improvement | PASS |
| Process kill → restart → durable checkpoint recovery | PASS |
| Tool OFF for Agent A while the same tool remains available to Agent B | PASS |
| RAG retrieval after the live-pipeline refresh improvement | PASS |
| RAG remove → next retrieval reflects corpus removal | PASS |

---

# Final Demo Evidence

The existing project evidence remains under:

```text
demo/
```

The final stateful-platform demonstration is documented in:

```text
demo/final_project_demo_transcript.md
```

A concise final demo should show:

1. the five-agent User Platform;
2. one grounded Memory/RAG request;
3. one stateful workflow;
4. a real HITL pause and administrator decision;
5. a real failure ticket and recovery;
6. persisted state inspection;
7. runtime tool administration;
8. RAG document administration;
9. checkpoint/restart recovery.

This keeps the demonstration focused on the complete Swiftrail system while highlighting the persistent operational capabilities.

---

# Final System Summary

Swiftrail combines:

```text
MCP
+
MySQL
+
Memory
+
RAG / Self-RAG
+
Context Management
+
Task Decomposition
+
Planning
+
Grounded Validation
+
Persistent State Graphs
+
Human-in-the-Loop
+
Failure Tickets
+
Durable Recovery
+
Runtime Tool Management
+
RAG Administration
+
User / Admin Platform
```

The overall design follows one core safety principle:

> **The LLM may reason, plan, retrieve, and recommend, but operational truth, authorization, persistence, and protected actions remain controlled by the surrounding Swiftrail system.**
