"""Consolidation layer.

This is the ONLY code path allowed to write to SemanticMemory. It runs
as a separate, periodic pass over EpisodicMemory -- never inline at
write time, and never triggered directly by PromoteDropRouter.

Run it on a schedule (cron / APScheduler / a Swiftrail nightly job) or
call `run()` manually / from a test. Either way it must be invoked
*after* episodes have already landed in EpisodicMemory, not as part of
routing them there.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory.episodic_store import EpisodicMemory
from memory.semantic_store import SemanticFact, SemanticMemory


# Maps an episode's event_type to the semantic fact_key it feeds, and
# how to derive the fact_value from the episode's content. This is the
# "extraction" step -- rule-based here, swappable for an LLM extractor
# later without touching the versioning/conflict logic in
# SemanticMemory.
def _extract_customer_risk(content: dict) -> str | None:
    if content.get("event_type") == "credit_hold_placed":
        severity = content.get("severity", "minor")
        return "high_risk" if severity == "severe" else "watch"
    if content.get("event_type") == "credit_hold_released":
        return "good_standing"
    return None


def _extract_discount_authority(content: dict) -> str | None:
    if content.get("event_type") == "rate_exception_rejected":
        return "standard_discount_only"
    if content.get("event_type") == "rate_exception_approved":
        pct = content.get("discount_pct")
        return f"approved_up_to_{pct}pct" if pct is not None else None
    return None


EXTRACTORS: dict[str, callable] = {
    "customer_risk_level": _extract_customer_risk,
    "discount_authority": _extract_discount_authority,
}


@dataclass
class ConsolidationResult:
    episodes_processed: int
    facts_written: list[SemanticFact]
    conflicts_resolved: list[SemanticFact]


class ConsolidationLayer:
    def __init__(self, episodic_memory: EpisodicMemory, semantic_memory: SemanticMemory):
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory

    def run(self) -> ConsolidationResult:
        """One consolidation pass: pull unconsolidated episodes, derive
        semantic facts, upsert them (handles versioning/conflict), mark
        the episodes consolidated, expire stale facts."""

        episodes = self.episodic_memory.get_unconsolidated()
        facts_written: list[SemanticFact] = []
        conflicts_resolved: list[SemanticFact] = []
        processed_ids: list[int] = []

        for episode in episodes:
            if episode.customer_id is None:
                processed_ids.append(episode.id)
                continue

            for fact_key, extractor in EXTRACTORS.items():
                value = extractor(episode.content)
                if value is None:
                    continue

                fact = self.semantic_memory.upsert_fact(
                    customer_id=episode.customer_id,
                    fact_key=fact_key,
                    fact_value=value,
                    source_episode_id=episode.id,
                )
                facts_written.append(fact)
                if fact.conflict_reason:
                    conflicts_resolved.append(fact)

            processed_ids.append(episode.id)

        self.episodic_memory.mark_consolidated(processed_ids)
        self.semantic_memory.expire_stale_facts()

        return ConsolidationResult(
            episodes_processed=len(processed_ids),
            facts_written=facts_written,
            conflicts_resolved=conflicts_resolved,
        )
