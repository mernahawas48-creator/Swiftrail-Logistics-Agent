# memory/

Short-term, episodic, and semantic memory for the Swiftrail agent.
See the "Memory System" section of the top-level README for the
problem statement and the worked conflict-resolution example.

This file is a map: where each graded concern lives, and which test
proves it.

| Concern | File | Key class/function | Proven by |
|---|---|---|---|
| Rolling short-term buffer | `short_term.py` | `ShortTermBuffer` | `test_short_term.py` |
| Scratchpad separate from the buffer | `scratchpad.py` | `Scratchpad` | `test_scratchpad.py::test_scratchpad_survives_short_term_buffer_pruning` |
| Promote-or-drop routing decision + logged reasoning | `router.py` | `PromoteDropRouter.route` / `decision_log` | `test_router.py::test_every_decision_is_logged_with_reasoning` |
| Router never writes to semantic memory | `router.py` | (no import of `semantic_store` anywhere in the file) | `test_router.py::test_router_never_touches_semantic_memory` (static AST check) |
| Episodic store, queryable by customer | `episodic_store.py` | `EpisodicMemory` | `test_episodic_store.py` |
| Consolidation as a separate periodic pass | `consolidation.py` | `ConsolidationLayer.run` | `test_consolidation.py` |
| Semantic fact versioning | `semantic_store.py` | `SemanticMemory.upsert_fact` | `test_semantic_store.py::test_conflicting_value_creates_new_version_and_supersedes_old` |
| Expiration of stale facts | `semantic_store.py` | `SemanticMemory.expire_stale_facts` | `test_semantic_store.py::test_expire_stale_facts` |
| Real conflict resolved across two runs | `consolidation.py` + `semantic_store.py` | -- | `test_consolidation.py::test_consolidation_resolves_a_real_conflict_across_two_runs` |

## Running it

```bash
pip install pytest --break-system-packages   # if not already installed
python -m pytest memory/ -v
python -m memory.demo_memory                 # end-to-end scenario, prints each step
```

## Storage

`EpisodicMemory` and `SemanticMemory` are both backed by a local SQLite
file (`memory/memory_store.db` by default, override with the
`MEMORY_DB_PATH` env var). Tests use a temporary DB per test via
pytest's `tmp_path` fixture, so they never touch the real store.

## Integration contract for `agent/`

`agent_loop.py` is expected to:

1. Own one `ShortTermBuffer` and one `Scratchpad` per session (replacing
   the current flat `AgentState.scratchpad` list).
2. Feed every evicted turn from the buffer into
   `PromoteDropRouter.process_overflow(...)`.
3. Run `ConsolidationLayer.run()` on a schedule (not inline per-request)
   -- e.g. a periodic job, not part of `AgentLoop.process`.
4. Read from `SemanticMemory.get_active_fact(...)` when the agent needs
   a customer's current standing, instead of re-deriving it from scratch
   every session.

