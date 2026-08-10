"""End-to-end demo of the memory system for one Swiftrail scenario.

Run with: python -m memory.demo_memory

Walks through:
  1. Short-term buffer overflowing during a long conversation.
  2. The promote-or-drop router deciding forget vs. episodic, with
     reasoning, for each evicted turn.
  3. A first consolidation pass turning episodes into a semantic fact.
  4. A second, later consolidation pass hitting a real conflict and
     resolving it with versioning instead of a silent overwrite.
"""

from __future__ import annotations

import os
import tempfile

from memory.consolidation import ConsolidationLayer
from memory.episodic_store import EpisodicMemory
from memory.router import PromoteDropRouter
from memory.scratchpad import Scratchpad
from memory.semantic_store import SemanticMemory
from memory.short_term import ShortTermBuffer, Turn
from memory.verification import MemoryVerifier


def main() -> None:
    db_path = os.path.join(tempfile.mkdtemp(), "demo_memory.db")
    episodic = EpisodicMemory(db_path=db_path)
    semantic = SemanticMemory(db_path=db_path)
    router = PromoteDropRouter(episodic)
    consolidation = ConsolidationLayer(episodic, semantic)

    print("=== 1. Scratchpad set for the current task ===")
    pad = Scratchpad()
    pad.set_goal("Review customer 12 before approving shipment 512")
    pad.update_state("customer_id", 12)
    print(pad.snapshot())

    print("\n=== 2. Short-term buffer fills up with a long triage call ===")
    buffer = ShortTermBuffer(max_turns=2)
    conversation = [
        Turn(role="employee", customer_id=12, session_id="sess-1",
             content="hi, checking on customer 12"),
        Turn(role="tool", customer_id=12, session_id="sess-1",
             content={"event_type": "shipment_status_lookup", "status": "in_transit"}),
        Turn(role="tool", customer_id=12, session_id="sess-1",
             content={"event_type": "credit_hold_released", "note": "balance settled"}),
        Turn(role="employee", customer_id=12, session_id="sess-1",
             content="ok thanks"),
        Turn(role="tool", customer_id=12, session_id="sess-1",
             content={"event_type": "rate_exception_approved", "discount_pct": 15,
                       "justification": "within standard 15% authority, no manager needed"}),
    ]

    for turn in conversation:
        evicted = buffer.add(turn)
        decision = router.process_overflow(evicted)
        if decision:
            print(f"  evicted -> {decision.action}: {decision.reason}")

    print("\n=== 3. First consolidation pass ===")
    result = consolidation.run()
    print(f"  episodes processed: {result.episodes_processed}")
    for fact in result.facts_written:
        print(f"  wrote fact: {fact.fact_key}={fact.fact_value} (v{fact.version})")

    print("\n=== 4. Weeks later: a severe credit hold lands on the same customer ===")
    late_turn = Turn(
        role="tool", customer_id=12, session_id="sess-9",
        content={"event_type": "credit_hold_placed", "severity": "severe",
                 "note": "90+ days overdue on shipment 512"},
    )
    decision = router.route(late_turn)
    print(f"  routed -> {decision.action}: {decision.reason}")

    print("\n=== 5. Second consolidation pass hits a real conflict ===")
    result2 = consolidation.run()
    for fact in result2.conflicts_resolved:
        print(f"  CONFLICT RESOLVED: {fact.fact_key} -> {fact.fact_value} "
              f"(v{fact.version})")
        print(f"    reason: {fact.conflict_reason}")

    print("\n=== 6. Full fact history is preserved, nothing silently lost ===")
    for fact in semantic.fact_history(12, "customer_risk_level"):
        print(f"  v{fact.version} [{fact.status}] {fact.fact_value} "
              f"(created {fact.created_at})")

    print("\n=== 7. Self-RAG-style verification on memory recall ===")
    verifier = MemoryVerifier()
    all_episodes = episodic.get_by_customer(12)

    print("  -- a relevant, on-topic query --")
    relevant_query = "any credit hold history for this customer"
    good = verifier.verify(relevant_query, all_episodes)
    print(f"  query: {relevant_query!r}")
    print(f"  passed: {good.passed}")
    print(f"  reason: {good.reason}")

    print("\n  -- an off-topic query against the same recalled episodes --")
    off_topic_query = "what is the weather forecast for tomorrow"
    bad = verifier.verify(off_topic_query, all_episodes)
    print(f"  query: {off_topic_query!r}")
    print(f"  passed: {bad.passed}")
    print(f"  reason: {bad.reason}")

    print("\n  -- the superseded fact from step 5/6, recalled on its own --")
    stale_facts = [f for f in semantic.fact_history(12, "customer_risk_level")
                   if f.status == "superseded"]
    stale_check = verifier.verify("what is the credit risk level for this customer",
                                   stale_facts)
    print(f"  passed: {stale_check.passed}")
    print(f"  reason: {stale_check.reason}")


if __name__ == "__main__":
    main()
