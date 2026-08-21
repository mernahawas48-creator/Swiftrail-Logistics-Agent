"""Deterministic routing for Swiftrail agent requests."""

from __future__ import annotations

import re

SECTION_ID_PATTERN = re.compile(r"\b[A-Z]{2,5}-\d+(?:\.\d+)?\b", re.IGNORECASE)


class QueryRouter:
    """Decide whether a request needs RAG, memory, or an MCP tool path."""

    def route(self, query: str) -> str:
        normalized = query.strip().lower()

        # Company knowledge: exact policy identifiers and policy/authority questions.
        if SECTION_ID_PATTERN.search(query) or any(
            term in normalized
            for term in (
                "policy",
                "manual",
                "procedure",
                "rule",
                "guideline",
                "authority",
                "who can release",
                "who may release",
                "who must approve",
            )
        ):
            return "rag"

        # Cross-session recall from episodic / semantic memory.
        if any(
            term in normalized
            for term in (
                "remember",
                "previous",
                "last time",
                "history",
                "before",
                "earlier session",
            )
        ):
            return "memory"

        # Existing MCP operational paths. Names match AgentLoop.call_mcp_tool.
        if any(
            term in normalized
            for term in ("shipment", "container", "tracking", "delivery", "package")
        ):
            return "shipment"

        if any(
            term in normalized
            for term in ("invoice", "bill", "payment", "amount due")
        ):
            return "invoice"

        if any(term in normalized for term in ("credit", "limit", "hold", "blocked")):
            return "credit"

        if any(term in normalized for term in ("customer", "client", "account")):
            return "customer"

        return "context"
