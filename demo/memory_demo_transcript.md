# Memory demo transcript (Person 1's part)

Extracted from `demo/memory_rag_demo_transcript.md` -- this file covers
only the Memory concerns: short-term buffer, scratchpad, promote-or-drop
router, episodic memory, semantic memory consolidation, conflict
resolution, and Self-RAG-style verification on memory recall.

No API key needed. This is real, captured output from running the repo,
not hand-written.

```
$ python -m memory.demo_memory
```

```
=== 1. Scratchpad set for the current task ===
{'goal': 'Review customer 12 before approving shipment 512', 'sub_goal': '', 'working_state': {'customer_id': 12}, 'updated_at': '2026-08-08T05:58:11.419542+00:00'}

=== 2. Short-term buffer fills up with a long triage call ===
  evicted -> forget: No operationally significant event detected in this turn (routine chatter or a lookup with no state change).
  evicted -> forget: No operationally significant event detected in this turn (routine chatter or a lookup with no state change).
  evicted -> episodic: balance settled

=== 3. First consolidation pass ===
  episodes processed: 1
  wrote fact: customer_risk_level=good_standing (v1)

=== 4. Weeks later: a severe credit hold lands on the same customer ===
  routed -> episodic: 90+ days overdue on shipment 512

=== 5. Second consolidation pass hits a real conflict ===
  CONFLICT RESOLVED: customer_risk_level -> high_risk (v2)
    reason: Superseded version 1 ('good_standing' -> 'high_risk') based on episode 2.

=== 6. Full fact history is preserved, nothing silently lost ===
  v1 [superseded] good_standing (created 2026-08-08T05:58:11.426141+00:00)
  v2 [active] high_risk (created 2026-08-08T05:58:11.440873+00:00)

=== 7. Self-RAG-style verification on memory recall ===
  -- a relevant, on-topic query --
  query: 'any credit hold history for this customer'
  passed: True
  reason: Relevance: Recalled memory passed the relevance check (2/5 content terms matched). Freshness: At least one recalled item is current (not superseded/expired).

  -- an off-topic query against the same recalled episodes --
  query: 'what is the weather forecast for tomorrow'
  passed: False
  reason: Relevance: Recalled memory failed the relevance check (0/3 content terms matched). Freshness: At least one recalled item is current (not superseded/expired).

  -- the superseded fact from step 5/6, recalled on its own --
  passed: False
  reason: Relevance: Recalled memory passed the relevance check (3/4 content terms matched). Freshness: All recalled items are stale: fact:customer_risk_level [superseded].
```

## What this shows

- **Short-term buffer + scratchpad (concern 1):** the scratchpad
  survives independently of the rolling buffer -- it isn't touched
  when the buffer prunes old turns.
- **Promote-or-drop router (concern 3):** two routine turns get
  dropped, one gets promoted to episodic, with the router's reasoning
  printed for each decision.
- **Semantic memory consolidation (concern 4):** a first consolidation
  pass turns the promoted episode into a semantic fact.
- **Conflict resolution:** a later, contradicting episode triggers a
  second consolidation pass that resolves the conflict by versioning
  (`good_standing` -> `high_risk`) instead of silently overwriting --
  the old value stays queryable as `v1 [superseded]`.
- **Self-RAG-style verification on memory recall:** passes a relevant,
  current recall, but catches and rejects both an off-topic query and
  a superseded fact recalled on its own, instead of handing either
  back as if it were trustworthy.
