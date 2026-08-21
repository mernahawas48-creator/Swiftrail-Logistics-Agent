"""Promote-or-drop router.

Fires when the short-term buffer overflows (ShortTermBuffer.add returns
an evicted Turn). For each aging turn, decides:

    - "forget": the turn is discarded, nothing else happens.
    - "episodic": the turn is written into EpisodicMemory.

IMPORTANT: this router NEVER writes to semantic memory. It has no
import of, or reference to, SemanticMemory anywhere in this file.
Semantic facts are only ever produced by ConsolidationLayer's separate,
periodic pass over EpisodicMemory (see consolidation.py).

Every decision is logged with its reasoning so a grader can see, for
any given turn, exactly why it was kept or dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory.episodic_store import EpisodicMemory
from memory.short_term import Turn

# Event types worth promoting for Swiftrail: anything that changes the
# operational or financial state of a customer relationship. Small talk
# and routine status lookups are not.
PROMOTABLE_EVENT_TYPES = {
    "credit_hold_placed",
    "credit_hold_released",
    "rate_exception_requested",
    "rate_exception_approved",
    "rate_exception_rejected",
    "customer_commitment",  # e.g. "customer agreed to pay within 30 days"
}


@dataclass
class RoutingDecision:
    action: str  # "forget" | "episodic"
    reason: str
    turn: Turn
    decided_at: str


class PromoteDropRouter:
    """Decides forget vs. episodic for each turn aged out of short-term
    memory, and writes the decision + reasoning to a visible log."""

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        classifier: Callable[[Turn], tuple[str, str] | None] | None = None,
    ):
        self.episodic_memory = episodic_memory
        # Pluggable classifier so this can be swapped for an LLM-based
        # one later without changing the routing/logging contract.
        # Returns (event_type, extracted_reason) or None if this turn
        # doesn't look like a promotable event at all.
        self._classify = classifier or self._default_classifier
        self.decision_log: list[RoutingDecision] = []

    def route(self, turn: Turn) -> RoutingDecision:
        classification = self._classify(turn)

        if classification is None:
            decision = RoutingDecision(
                action="forget",
                reason="No operationally significant event detected "
                "in this turn (routine chatter or a lookup with no "
                "state change).",
                turn=turn,
                decided_at=_now(),
            )
            self.decision_log.append(decision)
            return decision

        event_type, extracted_reason = classification

        self.episodic_memory.add_episode(
            event_type=event_type,
            content=_turn_content(turn),
            reason=extracted_reason,
            customer_id=turn.customer_id,
            source_session_id=turn.session_id,
        )

        decision = RoutingDecision(
            action="episodic",
            reason=extracted_reason,
            turn=turn,
            decided_at=_now(),
        )
        self.decision_log.append(decision)
        return decision

    def process_overflow(self, evicted_turn: Turn | None) -> RoutingDecision | None:
        """Convenience wrapper for the agent loop: call this with
        whatever ShortTermBuffer.add(...) returned."""

        if evicted_turn is None:
            return None
        return self.route(evicted_turn)

    # ------------------------------------------------------------------
    # Default, rule-based classifier for Swiftrail turns. A turn's
    # content is expected to be a dict with at least an "event_type"
    # key when it represents a tool result / structured event, e.g.:
    #   {"event_type": "credit_hold_placed", "customer_id": 12,
    #    "severity": "severe", "note": "90+ days overdue"}
    # ------------------------------------------------------------------
    @staticmethod
    def _default_classifier(turn: Turn) -> tuple[str, str] | None:
        content = turn.content
        if not isinstance(content, dict):
            return None

        event_type = content.get("event_type")
        if event_type not in PROMOTABLE_EVENT_TYPES:
            return None

        reason = content.get("note") or content.get("justification") or (
            f"Event type '{event_type}' changes the customer's "
            "operational or financial state and must be recoverable "
            "in future sessions."
        )
        return event_type, reason


def _turn_content(turn: Turn) -> dict[str, Any]:
    if isinstance(turn.content, dict):
        return turn.content
    return {"role": turn.role, "content": turn.content}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
