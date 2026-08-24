# Swiftrail Logistics Agent

## MCP, Memory & RAG, Planning, Persistent State Graphs, HITL, Recovery, and Platform Integration

Swiftrail is an enterprise logistics agent that has been extended across multiple project stages.

The same system now combines:

- MCP-based logistics operations
- MySQL business data
- Short-term, episodic, and semantic memory
- Retrieval-Augmented Generation
- Self-RAG verification
- Context-management strategies
- Task decomposition
- Plan-and-Solve
- Tree of Thoughts
- LATS
- Self-Refine and Reflexion
- Persistent state graphs
- Durable checkpoints
- Human-in-the-Loop
- Failure tickets and recovery
- Runtime MCP tool management
- RAG document administration
- User and Admin web interfaces

The final project extends the existing Swiftrail system instead of replacing the previous MCP, Memory/RAG, or Planning implementations.

---

# Problem Statement

Enterprise logistics agents operate across long and complex interactions involving:

- customers;
- shipments;
- invoices;
- credit holds;
- rate exceptions;
- employee authority;
- operational policies.

A simple conversation history is not enough.

The agent must be able to:

1. remember useful information across sessions;
2. retrieve verified business knowledge;
3. avoid stale, conflicting, or unauthorized facts;
4. decide what actions should happen and in what order;
5. respect operational authority boundaries;
6. pause when a customer or administrator must respond;
7. survive runtime failures and process restarts;
8. resume workflows without starting again;
9. expose real administrative control over tools and knowledge.

The Swiftrail project therefore combines memory, retrieval, planning, MCP tools, persistent workflows, human governance, and recovery in one logistics system.

---

# Overall Architecture

```text
                          SWIFTRAIL PLATFORM
                 ┌───────────────┴────────────────┐
                 │                                │
            User Platform                    Admin Platform
         chat + agent switching        tools / RAG / HITL / tickets
                 │                                │
                 └───────────────┬────────────────┘
                                 │
                         FastAPI Backend
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      │                          │                          │
 Memory & RAG Agent         Planning Agent            State Graphs
                                                      G1 / G2 / G3
      │                          │                          │
      └──────────────────────────┼──────────────────────────┘
                                 │
                           Existing MCP Server
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
             MySQL            Qdrant          Mistral
                │
         Logistics Data
```

The final project does not create a second logistics backend.

All stages reuse the same Swiftrail domain and operational infrastructure.

---

# MCP Server

The operational layer is implemented using FastMCP and MySQL.

## MCP Concerns

| MCP Concern | Implementation |
|---|---|
| Capability negotiation | Client reads server-declared capabilities before using protocol features |
| Notifications | Tool-list changes can trigger `tools/list_changed` |
| Elicitation | Protected decisions require explicit human confirmation |
| Sampling | Selected workflows can request model-generated summaries |
| Resources | Operational policies are exposed through MCP resources |
| Prompts | Parameterized operational prompts are supported |
| Transport | stdio during local development and Streamable HTTP for live integration |
| Progress tracking | Long operations can report progress |
| Defensive tools | Typed schemas, server-side validation, role checks, state checks, and safe failures |

## Main MCP Tools

The server includes operational tools such as:

```text
authenticate
search_customer
get_shipment_status
list_customer_invoices
approve_rate_exception
release_credit_hold
run_portfolio_risk_sweep
list_portfolio_credit_exposure
```

Additional final-project tools are reused by the state-graph workflows where required.

The MCP implementation remains authoritative for business validation even when an LLM proposes an action.

---

# Database

The application uses the existing Swiftrail MySQL database.

The original schema and seed data are located under:

```text
db/
```

The final project adds persistent workflow tables through:

```text
db/migrations/001_state_graph_and_delivery_recovery.sql
```

The migration extends the same database with persistent state information such as:

- graph runs;
- checkpoints;
- HITL tasks;
- failure tickets;
- execution receipts;
- delivery-recovery records.

The final project does not use a parallel business database.

---

# Memory System

## Problem

Swiftrail employees may work with the same customers across multiple sessions.

Without explicit memory:

- important events disappear after a conversation ends;
- previously rejected actions may be reconsidered incorrectly;
- old account state can be confused with current state;
- the agent may repeat work already completed.

The memory system separates temporary conversation context from durable events and long-term facts.

---

## Memory Architecture

```text
ShortTermBuffer
      │
      └── overflow
             ↓
      PromoteDropRouter
             │
             ├── Drop
             │
             └── Promote
                    ↓
              EpisodicMemory
                    │
              consolidation
                    ↓
              SemanticMemory

Scratchpad remains separate from pruning.
```

## Components

| Component | Purpose |
|---|---|
| `ShortTermBuffer` | Keeps recent conversation turns |
| `Scratchpad` | Stores current goal, sub-goals, and working state |
| `PromoteDropRouter` | Decides whether an evicted turn should be forgotten or stored |
| `EpisodicMemory` | Stores important historical events |
| `ConsolidationLayer` | Converts repeated events into stable facts |
| `SemanticMemory` | Stores versioned long-term facts |

---

# Semantic Conflict Resolution

Long-term facts are not simply overwritten.

If a customer's state changes, the old fact can be marked as superseded and a new version becomes active.

Example:

```text
Customer in good standing
        ↓
new severe credit hold
        ↓
old semantic fact = superseded
new semantic fact = active
```

This preserves an auditable history while preventing stale facts from remaining authoritative.

Semantic facts can also expire when they are no longer reinforced.

---

# Context Management

The project implements multiple context-management strategies under:

```text
context_eval/
```

Implemented strategies include:

| Strategy | Purpose |
|---|---|
| Sliding Window | Keep recent messages |
| Recursive Summarization | Compress older conversation history |
| Tool Output Masking | Remove unnecessary old tool output |
| Zone-Based Pruning | Preserve important context while pruning low-value regions |

The live agent uses a selected context strategy while the remaining strategies remain independently testable.

---

# RAG System

## Knowledge Corpus

The Swiftrail RAG corpus contains operational documents covering areas such as:

- credit holds;
- rate exceptions;
- portfolio risk;
- invoice collection;
- employee access;
- shipment pricing;
- delivery-exception recovery.

Documents are converted into retrieval units using section-aware processing and metadata.

---

## Ingestion Pipeline

```text
Documents
    ↓
Validated Loading
    ↓
Section-Aware Chunking
    ↓
Metadata Validation
    ↓
Embeddings
    ↓
Qdrant
```

The vector database is Qdrant.

The retrieval system applies metadata such as:

- role;
- document;
- section;
- status;
- business domain.

---

# Retrieval Architectures

## Naive RAG

```text
Query
→ Dense retrieval
→ Qdrant
→ Grounded generation
```

## Hybrid RAG

```text
Query
   ├── Dense retrieval
   └── BM25 lexical retrieval
             ↓
   Reciprocal Rank Fusion
             ↓
       Grounded answer
```

Hybrid RAG is used by the live Memory/RAG agent.

## Agentic RAG

```text
Plan
→ Retrieve
→ Grade evidence
→ Rewrite query if needed
→ Retrieve again
→ Generate grounded answer
```

Agentic RAG is retained for more complex multi-part retrieval cases.

---

# Self-RAG Verification

The RAG system does not immediately trust retrieved information.

The verification flow includes:

1. evidence relevance checking;
2. grounded answer generation;
3. citation validation;
4. factual-support validation;
5. numeric-claim checking;
6. safe abstention when evidence is insufficient.

This reduces unsupported or unauthorized answers.

---

# Agent Integration

The existing agent layer routes requests between different capabilities.

Conceptually:

```text
User Request
      ↓
Intent / task routing
      │
      ├── Policy / knowledge → RAG
      ├── Historical recall → Memory
      ├── Logistics operation → MCP
      └── Complex action sequence → Planning
```

The final state-graph agents are added as sibling agents rather than replacing the existing Memory/RAG or Planning agents.

---

# Planning and Decomposition System

## Planning Problem

A shipment may be blocked by several connected operational constraints.

For example, resolving one shipment may require checking:

- shipment state;
- customer account;
- overdue invoices;
- credit-hold severity;
- rate exception;
- employee role;
- company policy.

A single fixed tool call cannot safely solve this problem.

An early observation may change the rest of the plan.

---

# Planning Architecture

```text
Complex Swiftrail Request
          ↓
    DAG Decomposition
          ↓
 Sub-task Planning Router
      │       │       │
      ↓       ↓       ↓
 Plan-and-   Tree of   LATS
 Solve       Thoughts
      \       |       /
       Validated Result
```

---

# Planning Methods

## Plan-and-Solve

Used when the task is mostly linear and one explicit reasoning path is sufficient.

## Tree of Thoughts

Used when several alternatives need to be explored and compared before selecting one.

## LATS

Used for high-stakes branching tasks where external grounded feedback is available.

## Dynamic Decomposition

The project also supports dynamic decomposition where later planning steps can change after new observations.

---

# Self-Correction

## Self-Refine

The model:

```text
Generate
→ Evaluate
→ Revise
```

within the same trial.

## Reflexion

Reflexion stores a failure lesson and can reuse it in a later trial.

This provides cross-trial correction rather than only rewriting the current answer.

---

# Grounded Planning

The planning system does not rely only on the model to judge whether a plan is safe.

Candidate plans can be evaluated against:

- MCP tool results;
- database state;
- business rules;
- employee authority;
- shipment status;
- customer financial state.

The Swiftrail grounded validator acts as an external source of truth.

---

# Planning Concern Map

| Concern | Main Implementation |
|---|---|
| DAG construction | `planning/models.py`, `planning/algorithms/decomposition.py` |
| Dynamic decomposition | `planning/algorithms/dynamic_decomposition.py` |
| Plan-and-Solve | `planning/algorithms/plan_and_solve.py` |
| Tree of Thoughts | `planning/algorithms/tree_of_thoughts.py` |
| LATS | `planning/algorithms/lats.py` |
| Planning router | `planning/planning_router.py` |
| Grounded feedback | `planning/algorithms/environment.py`, `planning/swiftrail_validator.py` |
| Self-Refine | `planning/algorithms/self_refine.py` |
| Reflexion | `planning/algorithms/reflexion.py` |
| Episodic reflection buffer | `planning/episodic_buffer.py` |
| Real action execution | `planning/execution_adapter.py`, `planning/action_executor.py` |
| Planning orchestration | `planning/orchestrator.py` |
| Evaluation | `planning_eval/`, `artifacts/` |

---

# Planning Evaluation

The project contains both deterministic and live-provider planning evaluations.

The deterministic evaluation is used to make important behavioral cases reproducible, including:

- routing between planning methods;
- DAG validation;
- grounded vs. ungrounded planning;
- self-correction;
- Reflexion carry-over;
- dynamic replanning.

Live provider evaluation is stored separately under:

```text
artifacts/
```

Relevant artifacts include:

```text
artifacts/final_planning_evaluation.json
artifacts/final_planning_evaluation.md

artifacts/full_planning_benchmark.json
artifacts/full_planning_benchmark.md
artifacts/full_planning_benchmark_summary.json

artifacts/live_planning_benchmark.json
artifacts/live_planning_benchmark.md
```

Run the deterministic benchmark with:

```powershell
python -m planning_eval.full_benchmark
```

Run the live benchmark with:

```powershell
python -m planning_eval.live_benchmark
```

---

# Final Project — Persistent Stateful Workflows

The final project introduces three persistent state-graph workflows.

They reuse:

- the existing MCP server;
- the same MySQL database;
- the existing RAG system;
- Mistral;
- existing operational tools;
- previous Memory/RAG and Planning capabilities.

The shared runtime is implemented under:

```text
state_graph/core/
```

---

# Shared State-Graph Runtime

The shared runtime provides:

- serializable graph state;
- node transitions;
- durable checkpoints;
- external wait states;
- HITL pauses;
- persisted human decisions;
- failure tickets;
- recovery;
- node execution receipts.

Production state is persisted in MySQL.

Tests may use isolated local stores where appropriate.

---

# State Graph 1 — Delivery Exception Recovery & Customer Rerouting

## Problem

A shipment may enter a delivery-exception state after an unsuccessful delivery.

The system must:

- inspect the shipment;
- validate the exception;
- retrieve the relevant policy;
- generate recovery options;
- wait for the customer;
- evaluate the customer choice;
- request administrator approval for risky reroutes;
- execute and verify the final change.

---

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
    ├── request new options
    │       ↓
    │   generate_recovery_options
    │       ↓
    │   wait_for_customer
    │
    ├── safe reroute
    │       ↓
    │   apply_reroute
    │       ↓
    │   verify
    │       ↓
    │   complete
    │
    └── risky reroute
            ↓
         wait_for_admin
            │
            ├── reject
            │      ↓
            │   generate new options
            │
            └── approve
                   ↓
                apply reroute
                   ↓
                verify
                   ↓
                complete
```

---

## Graph 1 LLM Additions

### Task Decomposition

The model creates a structured recovery plan and identifies the policy information required by the workflow.

### RAG

The graph retrieves delivery-exception and rerouting policy evidence before continuing.

---

## Graph 1 HITL Conditions

A customer selection can require administrator review when the reroute is considered risky.

Examples include:

- unverified destination;
- rerouting cost above the configured threshold;
- customs-region change;
- high-value shipment.

The LLM cannot autonomously authorize these actions.

---

# State Graph 2 — Rate Exception Approval & Recovery

## Problem

A shipment may have a requested rate exception that must be evaluated against:

- policy;
- employee authority;
- current operational state.

---

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
         wait_for_admin
             ↓
        apply_rate_decision
             ↓
           complete
```

---

## Graph 2 LLM Additions

### RAG

Rate-exception policy is retrieved before authority classification.

### Constrained ReAct

The model is limited to approved operational actions while server-side rules remain authoritative.

---

# State Graph 3 — Credit-Hold Remediation

## Problem

A customer with an active credit hold may need to provide:

- payment information;
- dispute evidence;
- additional proof.

The workflow may need to wait across sessions and may require finance-manager intervention.

---

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
      ├── weak evidence
      │      ↓
      │   wait again
      │
      ├── partial remediation
      │      ↓
      │   complete with hold unchanged
      │
      └── acceptable input
              ↓
         classify_release_action
              │
              ├── high risk
              │      ↓
              │   finance HITL
              │
              └── allowed
                     ↓
                execute action
                     ↓
                  complete
```

---

## Graph 3 LLM Additions

### LATS

A bounded search compares remediation alternatives against grounded customer/account state.

### Constrained ReAct

The model is restricted to approved credit-hold operations.

Severe or risky releases require finance HITL.

---

## Graph 3 MCP Integration

The production graph uses real MCP-backed operations including:

```text
list_customer_invoices
list_customer_credit_holds
release_credit_hold
```

The final production path does not rely on the old local mock payment/dispute implementation.

---

# Durable Checkpointing

The shared GraphEngine persists workflow state at meaningful points.

Examples include:

- node completion;
- external wait;
- external input;
- HITL request;
- administrator decision;
- failure;
- recovery.

The persisted state contains information such as:

- run ID;
- graph name;
- current node;
- status;
- workflow data;
- transition history;
- HITL references;
- ticket references;
- recovery metadata.

---

# Execution Receipts

The state runtime stores node-execution receipts.

These receipts help prevent already-completed work from being blindly executed again during recovery.

---

# Human-in-the-Loop

HITL represents an expected authorization boundary.

It is not treated as a failure.

```text
High-Risk Condition
        ↓
Save Checkpoint
        ↓
Create HITL Task
        ↓
Pause Graph
        ↓
Admin Reviews Persisted State
        ↓
Approve / Reject
        ↓
Persist Decision
        ↓
Resume Workflow
```

The administrator's decision is stored rather than inferred by the model.

---

# Failure Tickets

Failure tickets represent unexpected execution failures.

Examples include:

- MCP communication failure;
- database failure;
- validation failure;
- unusable model output;
- runtime exception.

---

## Ticket Lifecycle

```text
Unexpected Failure
       ↓
Persist Failed State
       ↓
Create Ticket
       ↓
OPEN
       ↓
INVESTIGATING
       ↓
Correct Underlying Problem
       ↓
RESOLVED
       ↓
Resume from Saved Workflow State
```

HITL and tickets therefore serve different purposes.

| HITL | Failure Ticket |
|---|---|
| Expected human decision | Unexpected failure |
| Authorization boundary | Recovery boundary |
| Approve / Reject | Investigate / Resolve |
| Model intentionally pauses | Node cannot complete |

---

# Recovery Improvement

During final integration, Graph 1 recovery exposed an important stale-state issue.

The `validate_delivery_exception` recovery path was updated so that it reloads the current shipment state through the live operational tools instead of trusting only the shipment copy stored in an older checkpoint.

This allows recovery logic to observe real corrections made to the external system.

---

# Product Platform

The final browser platform is located under:

```text
platform/
```

Structure:

```text
platform/
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── admin.html
│   └── style.css
│
└── requirements.txt
```

The FastAPI backend serves the user and administrator interfaces.

---

# User Platform

The user interface exposes five agents:

1. Credit-Hold Remediation
2. Delivery Exception Recovery
3. Rate Exception Approval
4. Memory & RAG Agent
5. Decomposition & Planning Agent

Users can switch between agents through the same browser interface.

Requests are sent through the real backend:

```text
/api/chat
```

Stateful runs are exposed through:

```text
/api/runs
```

---

# Admin Dashboard

The administrator interface contains four main sections:

```text
Agents & Tools
RAG Documents
HITL Queue
Tickets
```

---

# Agents & Tools

Administrators can inspect the tools assigned to each registered agent.

Runtime configuration supports enabling and disabling non-protected MCP tools.

Protected tools remain required where needed.

---

# Runtime MCP Tool Management

Runtime tool management is implemented through:

```text
mcp_server/runtime_tool_manager.py
mcp_server/runtime_admin_api.py
mcp_server/runtime_setup.py
platform_app/admin/runtime_client.py
```

The runtime layer supports:

- registered agents;
- tool assignments;
- tool enable/disable;
- runtime tool registration;
- runtime tool removal;
- `tools/list_changed`;
- agent-specific permission checks.

---

# Per-Agent Runtime Enforcement

A shared tool may still exist on the MCP server because another agent uses it.

Therefore tool removal cannot rely only on deleting the tool globally.

The live graph MCP clients also perform agent-specific permission checks.

Conceptually:

```text
Agent A
Tool X = OFF
        ↓
Agent A attempts Tool X
        ↓
Permission Checker
        ↓
DENIED

Agent B
Tool X = ON
        ↓
Permission Checker
        ↓
ALLOWED
```

The live graph integration was updated so the graph MCP clients receive the runtime permission checker.

---

# RAG Document Administration

The Admin Dashboard allows administrators to:

- list RAG documents;
- add a document;
- specify metadata;
- specify allowed roles;
- remove removable documents;
- rebuild Qdrant.

---

## RAG Update Path

```text
Admin UI
    ↓
Platform Backend
    ↓
RAG Document Manager
    ↓
Corpus / Manifest Update
    ↓
Qdrant Re-index
    ↓
Refresh Live Retrieval Pipeline
```

The RAG manager can also roll back a corpus change if indexing fails.

---

# Live RAG Refresh

During final integration, document addition successfully updated the live corpus but the existing Memory/RAG agent still held an older retrieval pipeline in memory.

The platform integration was updated with a RAG refresh path.

After document addition or removal:

```text
Corpus changes
      ↓
Qdrant changes
      ↓
Cached Memory/RAG pipeline is cleared
      ↓
Next query rebuilds retrieval
```

This ensures future retrieval can use the updated knowledge base.

---

# HITL Queue

The Admin Dashboard exposes pending HITL requests.

For each task, the administrator can inspect information such as:

- run ID;
- graph;
- reason;
- decision status.

The final integration also exposes:

```text
View persisted state
```

so the administrator can inspect the graph checkpoint associated with the pause.

---

# Failure Tickets UI

The Tickets page exposes:

- status;
- run ID;
- agent;
- failed node;
- error;
- persisted state.

The administrator can move a ticket through:

```text
OPEN
→ INVESTIGATING
→ RESOLVED
```

The UI also provides:

```text
View persisted state
```

for inspection of the failed checkpoint.

---

# Final Project Concern Map

| Concern | Main Implementation |
|---|---|
| Shared state runtime | `state_graph/core/` |
| Graph engine | `state_graph/core/engine.py` |
| Serializable state | `state_graph/core/state.py` |
| Durable MySQL store | `state_graph/core/mysql_store.py` |
| Wait/HITL nodes | `state_graph/core/nodes.py` |
| Delivery Exception workflow | `state_graph/graph_1/` |
| Rate Exception workflow | `state_graph/graph_2/` |
| Credit-Hold workflow | `state_graph/graph_3/` |
| Runtime tool management | `mcp_server/runtime_tool_manager.py` |
| Runtime Admin API | `mcp_server/runtime_admin_api.py` |
| Runtime setup | `mcp_server/runtime_setup.py` |
| Previous-agent platform integration | `platform_app/agent_integration.py` |
| State-graph platform integration | `platform_app/graph_integration.py` |
| Admin services | `platform_app/admin/` |
| User web interface | `platform/frontend/index.html` |
| Admin web interface | `platform/frontend/admin.html` |
| FastAPI product backend | `platform/backend/app.py` |
| State-graph migration | `db/migrations/001_state_graph_and_delivery_recovery.sql` |

---

# Project Structure

```text
agent/
    Existing agent loop, MCP clients, sessions

artifacts/
    Planning and evaluation artifacts

context_eval/
    Context-management strategies and evaluation

db/
    MySQL schema, seed data, ERD, final-project migrations

demo/
    MCP, Memory/RAG, Planning, and final-project evidence

mcp_server/
    FastMCP server, operational tools, runtime tool management

memory/
    Short-term, episodic, semantic memory

planning/
    Decomposition and planning methods

planning_eval/
    Planning evaluation and benchmarks

platform/
    Runnable browser product

platform_app/
    Platform integration and admin services

rag/
    Corpus, ingestion, retrieval, Qdrant, verification

retrieval_eval/
    Retrieval evaluation

state_graph/
    Shared state runtime and three final-project graphs
```

---

# Setup

## 1. Create a Virtual Environment

Example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

# 2. Install Dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

For the platform:

```powershell
python -m pip install -r platform\requirements.txt
```

---

# 3. Configure MySQL

Create:

```text
swiftrail_db
```

Then apply the original schema and seed data.

Example:

```text
db/schema.sql
db/seed.sql
```

Apply the final-project migration:

```text
db/migrations/001_state_graph_and_delivery_recovery.sql
```

---

# 4. Environment Variables

Create a local `.env`.

Example:

```env
MISTRAL_API_KEY=your_key
MISTRAL_MODEL=mistral-small-latest

DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=swiftrail_db

QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=swiftrail

SWIFTRAIL_ADMIN_TOKEN=replace_with_a_secure_local_token
SWIFTRAIL_MCP_ADMIN_URL=http://127.0.0.1:8000/admin/runtime
```

Do not commit real secrets.

---

# 5. Start Qdrant

```powershell
docker compose -f rag\vector_store\docker-compose.yml up -d
```

If required, rebuild the corpus:

```powershell
python -m rag.ingestion.pipeline --recreate
```

---

# Running the MCP Server

## stdio

```powershell
python mcp_server\server.py
```

## Streamable HTTP

```powershell
python mcp_server\server.py --http
```

The HTTP server runs on:

```text
http://127.0.0.1:8000
```

with MCP available at:

```text
http://127.0.0.1:8000/mcp
```

---

# Running the Final Platform

Keep the MCP server running.

In another terminal:

```powershell
cd platform\backend
python -m uvicorn app:app --reload --port 8080
```

Open:

## User Platform

```text
http://127.0.0.1:8080/
```

## Admin Dashboard

```text
http://127.0.0.1:8080/admin.html
```

---

# Other Run Commands

## Memory Demo

```powershell
python -m memory.demo_memory
```

## Hybrid RAG Example

```powershell
python -m rag.hybrid_rag.cli --query "RE-2" --role finance_manager
```

## Planning Agent

```powershell
python -m planning.run_swiftrail --shipment-id 3 --customer-id 3 --employee-id 1
```

Dynamic decomposition:

```powershell
python -m planning.run_swiftrail --shipment-id 3 --customer-id 3 --employee-id 1 --method dynamic
```

---

# Evaluation Commands

## Retrieval Architecture Evaluation

```powershell
python -m retrieval_eval.evaluate_architectures
```

## Planning Evaluation

```powershell
python -m planning_eval.final_evaluation
```

## Deterministic Planning Benchmark

```powershell
python -m planning_eval.full_benchmark
```

## Live Planning Benchmark

```powershell
python -m planning_eval.live_benchmark
```

---

# Tests

Relevant test suites include:

```powershell
python -m pytest memory -q
python -m pytest context_eval -q
python -m pytest rag\tests -q
python -m pytest retrieval_eval\test_evaluate_architectures.py -q

python -m pytest agent -q

python -m pytest planning -q
python -m pytest planning_eval -q

python -m pytest state_graph\tests -q

python -m pytest platform_app -q

python -m pytest mcp_server\test_runtime_admin_api.py -q
```

Tests requiring MySQL or Qdrant require the corresponding services to be running.

---

# Final End-to-End Validation

The final platform was exercised through the browser, FastAPI backend, MCP runtime, MySQL, and RAG administration layer.

---

## Platform Startup

The MCP server successfully started using:

```powershell
python mcp_server\server.py --http
```

The final platform successfully started using:

```powershell
cd platform\backend
python -m uvicorn app:app --reload --port 8080
```

The User and Admin interfaces both loaded successfully.

The backend returned successful responses for routes including:

```text
GET /api/agents
GET /api/admin/agents
GET /api/admin/rag/documents
GET /api/admin/hitl
GET /api/admin/tickets
```

---

# Agent Discovery

The User Platform displayed the five expected live agents:

```text
Credit-Hold Remediation
Delivery Exception Recovery
Rate Exception Approval
Memory & RAG Agent
Decomposition & Planning Agent
```

The Admin Dashboard also successfully loaded runtime agent/tool information.

---

# Memory/RAG Validation

A live customer-scoped session was started using:

```text
start customer 3, role finance_manager
```

The Memory/RAG agent then answered:

```text
What is the policy for delivery exceptions?
```

The answer returned grounded delivery-exception policy information.

The User Platform displayed:

```text
Customer session started
Verified RAG retrieval
```

This verified that the previous Memory/RAG implementation remained accessible through the final platform.

---

# Credit-Hold Graph Validation

The Credit-Hold Remediation graph was successfully reached from the User Platform.

Example:

```text
start customer 3, claim: one invoice is incorrect
```

The live workflow checked the current customer state.

For the selected customer, the workflow correctly returned:

```text
no_active_hold
```

and completed.

---

# Failure Ticket Validation

Graph 1 was started with a shipment whose current database state did not satisfy the Delivery Exception workflow.

The graph reached:

```text
validate_delivery_exception
```

and failed safely.

The User Platform reported that Graph 1 had opened an administrator ticket.

A real failure ticket was automatically persisted and appeared in:

```text
Admin Dashboard → Tickets
```

No ticket was manually inserted.

---

# Ticket Administration Validation

The administrator could inspect:

- graph;
- run ID;
- failed node;
- error;
- persisted state.

The ticket controls supported:

```text
OPEN
→ INVESTIGATING
→ RESOLVED
```

The underlying shipment state was corrected in MySQL during investigation.

The recovery implementation was subsequently improved so the validation node can reload current shipment state instead of relying only on stale checkpoint data.

---

# Persisted State Validation

The Admin Dashboard now exposes:

```text
View persisted state
```

for:

- HITL tasks;
- failure tickets.

This gives the administrator direct visibility into the saved graph state associated with the pause/failure.

---

# RAG Administration Validation

A new document was submitted through:

```text
Admin Dashboard → RAG Documents
```

The backend returned:

```text
POST /api/admin/rag/documents → 200 OK
```

and the document appeared in the Live Corpus.

During this validation, the system exposed a stale live-retrieval cache issue.

The integration was updated so add/remove operations clear the existing Memory/RAG retrieval pipeline and allow the next query to rebuild against the updated corpus.

---

# Runtime Tool Management Validation

The Admin Dashboard successfully retrieved real runtime MCP agent/tool assignments.

The runtime implementation supports:

- per-agent assignments;
- enable/disable;
- protected tools;
- server-side tool management;
- permission checks.

The final graph integration was updated so live Graph 1, Graph 2, and Graph 3 MCP clients receive the agent-specific permission checker.

---

# Current End-to-End Status

The following items were directly exercised during final integration:

# Current End-to-End Status

| Validation | Status |
|---|---|
| MCP HTTP server starts | PASS |
| Web platform starts | PASS |
| User Platform loads | PASS |
| Admin Dashboard loads | PASS |
| Five agents are exposed | PASS |
| Runtime agents/tools load | PASS |
| Memory/RAG session starts | PASS |
| Existing RAG policy retrieval works | PASS |
| Credit-Hold graph reachable from UI | PASS |
| State graph reads live business state | PASS |
| Genuine Graph 1 failure creates ticket | PASS |
| Ticket appears in admin platform | PASS |
| Persisted ticket state visible | PASS |
| Ticket investigation control works | PASS |
| Ticket resolution control works | PASS |
| RAG document POST succeeds | PASS |
| Added RAG document appears in live corpus | PASS |
| HITL pause → admin decision → same-run resume | PASS |
| Same-run ticket recovery after latest fix | PASS |
| Process kill → restart → checkpoint recovery | PASS |
| Tool OFF for Agent A → denied while Agent B remains allowed | PASS |
| RAG query after live-pipeline refresh | PASS |
| RAG remove → next retrieval changes | PASS |

---

# Demo Evidence

Existing project evidence is stored under:

```text
demo/
```

Earlier project evidence includes:

```text
demo/demo_transcript.md
demo/memory_demo_transcript.md
demo/planning_demo_transcript.md
```

The final-project demo is documented in:

```text
demo/final_project_demo_transcript.md
```

---

# Recommended Final Demo Flow

For a short live presentation:

```text
1. Show architecture and five agents
2. Show Memory/RAG retrieval
3. Show Delivery Exception workflow
4. Show HITL queue and persisted state
5. Show Failure Tickets and persisted state
6. Show runtime tool administration
7. Show RAG document administration
8. Explain persistent checkpoint recovery
```

The prepared presentation/demo should remain short enough to leave time for discussion and questions.

---

# Repository Safety

The repository should not contain:

- real API keys;
- database passwords;
- admin tokens;
- committed `.env` secrets;
- hardcoded credentials.

Use:

```text
.env.example
```

for placeholders.

Keep the real:

```text
.env
```

ignored by Git.

---

# Final Summary

Swiftrail evolved from a basic logistics MCP agent into a multi-agent operational platform.

The complete architecture now combines:

```text
MCP
+
MySQL
+
Memory
+
RAG
+
Self-RAG
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
Failure Recovery
+
Runtime Tool Management
+
RAG Administration
+
User/Admin Platform
```

The core design principle across every stage is the same:

> The LLM may reason and propose actions, but real logistics state, policy, authorization, persistence, and safety are enforced by the surrounding system.
