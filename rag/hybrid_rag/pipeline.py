"""Hybrid RAG: dense + BM25 retrieval, verification, and generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag.hybrid_search.search import HybridSearch
from rag.naive_rag.generator import (
    MistralTextGenerator,
    TextGenerator,
)
from rag.naive_rag.pipeline import NO_CONTEXT_ANSWER
from rag.verification.verifier import (
    SelfRAGVerifier,
    VerificationSummary,
)


@dataclass(frozen=True, slots=True)
class HybridRAGSource:
    number: int
    chunk_id: str
    doc_id: str
    title: str
    section_id: str
    section_title: str
    fused_score: float
    dense_rank: int | None
    sparse_rank: int | None


@dataclass(frozen=True, slots=True)
class HybridRAGAnswer:
    query: str
    answer: str
    sources: tuple[HybridRAGSource, ...]
    retrieved_count: int
    model_name: str
    verification: VerificationSummary | None = None


class HybridRAG:
    """Generate a grounded answer from RRF-fused dense and BM25 evidence."""

    def __init__(
        self,
        *,
        searcher: Any | None = None,
        generator: TextGenerator | None = None,
        verifier: SelfRAGVerifier | None = None,
    ):
        self.searcher = searcher or HybridSearch()
        self.generator = generator or MistralTextGenerator()
        self.verifier = verifier or SelfRAGVerifier()

    def answer(
        self,
        query: str,
        *,
        role: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        statuses: Sequence[str] = ("active",),
        departments: Sequence[str] | None = None,
        document_types: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
    ) -> HybridRAGAnswer:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("The query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        results = self.searcher.search(
            normalized_query,
            role=role,
            top_k=top_k,
            candidate_k=candidate_k,
            statuses=tuple(statuses),
            departments=departments,
            document_types=document_types,
            doc_ids=doc_ids,
            section_ids=section_ids,
        )

        sources = self._build_sources(results)

        if not results:
            return HybridRAGAnswer(
                query=normalized_query,
                answer=NO_CONTEXT_ANSWER,
                sources=(),
                retrieved_count=0,
                model_name=self._model_name(),
                verification=VerificationSummary(
                    retrieval_relevant=False,
                    answer_supported=True,
                    citations_valid=True,
                    reason="No authorized evidence was retrieved.",
                ),
            )

        relevance = self.verifier.check_relevance(
            normalized_query,
            results,
        )

        if not relevance.passed:
            return HybridRAGAnswer(
                query=normalized_query,
                answer=NO_CONTEXT_ANSWER,
                sources=sources,
                retrieved_count=len(results),
                model_name=self._model_name(),
                verification=VerificationSummary(
                    retrieval_relevant=False,
                    answer_supported=True,
                    citations_valid=True,
                    reason=relevance.reason,
                ),
            )

        prompt = self._build_prompt(
            query=normalized_query,
            role=role,
            results=results,
        )
        answer = self.generator.generate(prompt).strip()

        support = self.verifier.check_support(
            answer,
            results,
        )
        verification = self.verifier.summarize(
            relevance,
            support,
        )

        if not support.passed:
            answer = NO_CONTEXT_ANSWER

        return HybridRAGAnswer(
            query=normalized_query,
            answer=answer,
            sources=sources,
            retrieved_count=len(results),
            model_name=self._model_name(),
            verification=verification,
        )

    def _model_name(self) -> str:
        return str(
            getattr(
                self.generator,
                "model_name",
                self.generator.__class__.__name__,
            )
        )

    @staticmethod
    def _build_sources(
        results: Sequence[Any],
    ) -> tuple[HybridRAGSource, ...]:
        return tuple(
            HybridRAGSource(
                number=number,
                chunk_id=str(result.chunk_id),
                doc_id=str(result.metadata["doc_id"]),
                title=str(result.metadata["title"]),
                section_id=str(result.metadata["section_id"]),
                section_title=str(
                    result.metadata["section_title"]
                ),
                fused_score=float(result.fused_score),
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank,
            )
            for number, result in enumerate(
                results,
                start=1,
            )
        )

    @staticmethod
    def _build_prompt(
        *,
        query: str,
        role: str,
        results: Sequence[Any],
    ) -> str:
        context_blocks: list[str] = []

        for number, result in enumerate(
            results,
            start=1,
        ):
            metadata = result.metadata
            context_blocks.append(
                "\n".join(
                    [
                        f"[{number}]",
                        f"Document: {metadata['title']}",
                        f"Document ID: {metadata['doc_id']}",
                        (
                            "Section: "
                            f"{metadata['section_id']} — "
                            f"{metadata['section_title']}"
                        ),
                        (
                            "Hybrid score: "
                            f"{result.fused_score:.6f}"
                        ),
                        "Text:",
                        result.text.strip(),
                    ]
                )
            )

        context = "\n\n".join(context_blocks)

        return f"""
You are the Swiftrail Logistics knowledge assistant.

Answer using ONLY the authorized hybrid-retrieval evidence below.

Rules:
1. Cite every factual claim using source numbers such as [1] or [2].
2. Do not use outside knowledge.
3. Do not invent a policy, threshold, role, permission, or procedure.
4. If the evidence is insufficient, return the safe abstention message.
5. The authenticated role is: {role}

AUTHORIZED EVIDENCE
-------------------
{context}

USER QUESTION
-------------
{query}

ANSWER
------
""".strip()
