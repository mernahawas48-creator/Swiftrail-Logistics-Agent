"""Self-RAG-style verification for memory recall.

rag/verification/verifier.py checks RAG answers against retrieved
document chunks. This module is the equivalent for the *other* half of
the lab's Self-RAG requirement -- "This applies to both your RAG
answers and to memories recalled from the episodic and semantic
store" -- which had no code path anywhere before this file.

Two checks, deterministic and auditable like the RAG verifier:

  - check_relevance: does a recalled item actually relate to the query
    that triggered the recall, via content-term overlap? A customer_id
    match alone is not enough -- an unrelated fact/episode for the
    right customer should not be handed back as if it answered the
    question.
  - check_freshness: is the item still something we should act on?
    A SemanticFact that's "superseded" or "expired" (see
    semantic_store.py) or an Episode that's stale must not be surfaced
    as current truth just because it matched on relevance.

A failure has a visible consequence wherever this is called from (see
agent/agent_loop.py's "memory" branch): the agent returns a safe
abstention instead of confidently repeating a stale or off-topic
memory.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from memory.episodic_store import Episode
from memory.semantic_store import SemanticFact

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can",
    "could", "do", "does", "for", "from", "has", "have", "how", "i",
    "if", "in", "is", "it", "may", "must", "of", "on", "or", "should",
    "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would",
}

MemoryItem = SemanticFact | Episode

STALE_STATUSES = {"superseded", "expired"}


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class MemoryVerificationSummary:
    relevant: bool
    fresh: bool
    reason: str

    @property
    def passed(self) -> bool:
        return self.relevant and self.fresh


class MemoryVerifier:
    """Checks a recalled episodic/semantic item before it's handed back
    to the user as if it were current, trustworthy information."""

    def __init__(self, minimum_relevance_coverage: float = 0.2):
        if not 0.0 <= minimum_relevance_coverage <= 1.0:
            raise ValueError(
                "minimum_relevance_coverage must be between 0 and 1."
            )
        self.minimum_relevance_coverage = minimum_relevance_coverage

    def check_relevance(
        self,
        query: str,
        items: Sequence[MemoryItem],
    ) -> VerificationCheck:
        if not items:
            return VerificationCheck(
                passed=False,
                reason="No memory items were recalled.",
            )

        query_terms = self._content_terms(query)
        if not query_terms:
            return VerificationCheck(
                passed=False,
                reason="The query has no content terms that can be verified.",
            )

        evidence_terms: set[str] = set()
        for item in items:
            evidence_terms.update(self._content_terms(self._item_text(item)))

        matched = query_terms.intersection(evidence_terms)
        coverage = len(matched) / len(query_terms)

        if coverage >= self.minimum_relevance_coverage:
            return VerificationCheck(
                passed=True,
                reason=(
                    "Recalled memory passed the relevance check "
                    f"({len(matched)}/{len(query_terms)} content terms matched)."
                ),
            )

        return VerificationCheck(
            passed=False,
            reason=(
                "Recalled memory failed the relevance check "
                f"({len(matched)}/{len(query_terms)} content terms matched)."
            ),
        )

    def check_freshness(self, items: Sequence[MemoryItem]) -> VerificationCheck:
        if not items:
            return VerificationCheck(
                passed=False,
                reason="No memory items were recalled.",
            )

        stale = [item for item in items if self._is_stale(item)]
        if stale:
            stale_desc = ", ".join(
                f"{self._item_label(item)} [{getattr(item, 'status', 'n/a')}]"
                for item in stale
            )
            if len(stale) == len(items):
                return VerificationCheck(
                    passed=False,
                    reason=f"All recalled items are stale: {stale_desc}.",
                )

        return VerificationCheck(
            passed=True,
            reason="At least one recalled item is current (not superseded/expired).",
        )

    def summarize(
        self,
        relevance: VerificationCheck,
        freshness: VerificationCheck,
    ) -> MemoryVerificationSummary:
        return MemoryVerificationSummary(
            relevant=relevance.passed,
            fresh=freshness.passed,
            reason=f"Relevance: {relevance.reason} Freshness: {freshness.reason}",
        )

    def verify(self, query: str, items: Sequence[MemoryItem]) -> MemoryVerificationSummary:
        """Convenience entry point: relevance check, then freshness check."""
        relevance = self.check_relevance(query, items)
        freshness = self.check_freshness(items)
        return self.summarize(relevance, freshness)

    @staticmethod
    def _is_stale(item: MemoryItem) -> bool:
        status = getattr(item, "status", None)
        return status in STALE_STATUSES

    @staticmethod
    def _item_text(item: MemoryItem) -> str:
        if isinstance(item, SemanticFact):
            return f"{item.fact_key} {item.fact_value}"
        if isinstance(item, Episode):
            return f"{item.event_type} {item.reason} {item.content}"
        return str(item)

    @staticmethod
    def _item_label(item: MemoryItem) -> str:
        if isinstance(item, SemanticFact):
            return f"fact:{item.fact_key}"
        if isinstance(item, Episode):
            return f"episode:{item.event_type}"
        return "item"

    @classmethod
    def _content_terms(cls, text: str) -> set[str]:
        return {
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if token.lower() not in STOPWORDS and len(token) > 1
        }
