"""Self-RAG-style verification for episodic and semantic memory recall.

This is the memory-side counterpart to the RAG verification path.  Recalled
memory is checked for relevance before generation, and the generated answer is
checked against the cited memory evidence before it is returned.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memory.episodic_store import EpisodicMemory
from memory.semantic_store import SemanticMemory
from rag.naive_rag.generator import MistralTextGenerator, TextGenerator
from rag.verification.verifier import SelfRAGVerifier, VerificationSummary

SAFE_MEMORY_ANSWER = (
    "I could not find enough verified memory to answer this question."
)
_MEMORY_CUE_PATTERN = re.compile(
    r"\b(?:remember|previous|previously|last\s+time|history|historical|before|earlier(?:\s+session)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    """Adapter exposing recalled memory in the verifier's evidence shape."""

    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MemoryRecallSource:
    number: int
    memory_type: str
    memory_id: str
    label: str


@dataclass(frozen=True, slots=True)
class VerifiedMemoryAnswer:
    query: str
    answer: str
    sources: tuple[MemoryRecallSource, ...]
    verification: VerificationSummary


class VerifiedMemoryRecall:
    """Recall episodic + semantic memory with explicit pre/post checks."""

    def __init__(
        self,
        *,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        generator: TextGenerator | None = None,
        verifier: SelfRAGVerifier | None = None,
        max_episodes: int = 5,
    ):
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.generator = generator or MistralTextGenerator()
        self.verifier = verifier or SelfRAGVerifier()
        self.max_episodes = max_episodes

    def answer(self, query: str, *, customer_id: int) -> VerifiedMemoryAnswer:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("The memory query cannot be empty.")

        evidence = self._collect_evidence(customer_id)
        sources = self._build_sources(evidence)
        verification_query = self._verification_query(normalized_query)
        relevance = self.verifier.check_relevance(verification_query, evidence)

        if not relevance.passed:
            return VerifiedMemoryAnswer(
                query=normalized_query,
                answer=SAFE_MEMORY_ANSWER,
                sources=sources,
                verification=VerificationSummary(
                    retrieval_relevant=False,
                    answer_supported=True,
                    citations_valid=True,
                    reason=relevance.reason,
                ),
            )

        prompt = self._build_prompt(normalized_query, evidence)
        answer = self.generator.generate(prompt).strip()
        support = self.verifier.check_support(answer, evidence)
        verification = self.verifier.summarize(relevance, support)

        if not support.passed:
            answer = SAFE_MEMORY_ANSWER

        return VerifiedMemoryAnswer(
            query=normalized_query,
            answer=answer,
            sources=sources,
            verification=verification,
        )

    def _collect_evidence(self, customer_id: int) -> list[MemoryEvidence]:
        evidence: list[MemoryEvidence] = []

        for fact in self.semantic_memory.get_active_facts(customer_id):
            evidence.append(
                MemoryEvidence(
                    text=f"{fact.fact_key}: {fact.fact_value}",
                    metadata={
                        "memory_type": "semantic",
                        "memory_id": f"semantic-{fact.id}",
                        "title": "Semantic memory",
                        "section_id": "",
                        "section_title": fact.fact_key,
                        "keywords": [fact.fact_key, fact.fact_value],
                    },
                )
            )

        episodes = self.episodic_memory.get_by_customer(customer_id)
        for episode in reversed(episodes[-self.max_episodes :]):
            content_text = json.dumps(
                episode.content,
                sort_keys=True,
                ensure_ascii=False,
            )
            evidence.append(
                MemoryEvidence(
                    text=(
                        f"event_type={episode.event_type}; "
                        f"content={content_text}; reason={episode.reason}"
                    ),
                    metadata={
                        "memory_type": "episodic",
                        "memory_id": f"episode-{episode.id}",
                        "title": "Episodic memory",
                        "section_id": "",
                        "section_title": episode.event_type,
                        "keywords": [episode.event_type],
                    },
                )
            )

        return evidence

    @staticmethod
    def _verification_query(query: str) -> str:
        stripped = _MEMORY_CUE_PATTERN.sub(" ", query)
        return " ".join(stripped.split()) or query

    @staticmethod
    def _build_sources(
        evidence: Sequence[MemoryEvidence],
    ) -> tuple[MemoryRecallSource, ...]:
        return tuple(
            MemoryRecallSource(
                number=number,
                memory_type=str(item.metadata["memory_type"]),
                memory_id=str(item.metadata["memory_id"]),
                label=str(item.metadata["section_title"]),
            )
            for number, item in enumerate(evidence, start=1)
        )

    @staticmethod
    def _build_prompt(query: str, evidence: Sequence[MemoryEvidence]) -> str:
        blocks = []
        for number, item in enumerate(evidence, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[{number}]",
                        f"Type: {item.metadata['memory_type']}",
                        f"Label: {item.metadata['section_title']}",
                        f"Memory: {item.text}",
                    ]
                )
            )
        context = "\n\n".join(blocks)
        return f"""
You are the Swiftrail memory assistant.

Answer using ONLY the recalled memory evidence below.
Rules:
1. Cite every factual sentence with source numbers such as [1] or [2].
2. Do not add facts that are not present in the recalled memories.
3. If the memories do not support the requested detail, say so without inventing it.

RECALLED MEMORY
---------------
{context}

QUESTION
--------
{query}

ANSWER
------
""".strip()
