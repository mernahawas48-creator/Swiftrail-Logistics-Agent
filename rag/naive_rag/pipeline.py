"""Single-retrieval, single-generation Naive RAG pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag.embeddings.embedder import ChunkEmbedder
from rag.metadata.schema import SearchFilters
from rag.naive_rag.generator import (
    MistralTextGenerator,
    TextGenerator,
)
from rag.verification.verifier import (
    SelfRAGVerifier,
    VerificationSummary,
)

NO_CONTEXT_ANSWER = (
    "I could not find enough authorized information in the "
    "Swiftrail knowledge base to answer this question."
)


@dataclass(frozen=True, slots=True)
class RAGSource:
    """Source information exposed with the generated answer."""

    number: int
    chunk_id: str
    doc_id: str
    title: str
    section_id: str
    section_title: str
    score: float


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Answer, retrieval evidence, and verification result."""

    query: str
    answer: str
    sources: tuple[RAGSource, ...]
    retrieved_count: int
    model_name: str
    verification: VerificationSummary | None = None


class NaiveRAG:
    """Retrieve once from Qdrant, verify, then make one grounded LLM call."""

    def __init__(
        self,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        generator: TextGenerator | None = None,
        verifier: SelfRAGVerifier | None = None,
    ):
        self.embedder = embedder or ChunkEmbedder()

        if vector_store is None:
            # Lazy import keeps unit tests independent from qdrant-client.
            from rag.vector_store.qdrant_store import (
                QdrantVectorStore,
            )
            vector_store = QdrantVectorStore()

        self.vector_store = vector_store
        self.generator = (
            generator or MistralTextGenerator()
        )
        self.verifier = (
            verifier or SelfRAGVerifier()
        )

    def answer(
        self,
        query: str,
        *,
        role: str,
        top_k: int = 5,
        statuses: Sequence[str] = ("active",),
        departments: Sequence[str] | None = None,
        document_types: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        section_ids: Sequence[str] | None = None,
    ) -> RAGAnswer:
        """Answer one question using authorized and verified chunks."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("The query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        filters = SearchFilters(
            role=role,
            statuses=tuple(statuses),
            departments=(
                tuple(departments)
                if departments is not None
                else None
            ),
            document_types=(
                tuple(document_types)
                if document_types is not None
                else None
            ),
            doc_ids=(
                tuple(doc_ids)
                if doc_ids is not None
                else None
            ),
            section_ids=(
                tuple(section_ids)
                if section_ids is not None
                else None
            ),
        )

        query_vector = self.embedder.embed_query(
            normalized_query
        )

        results = self.vector_store.search(
            query_vector,
            filters,
            top_k=top_k,
        )

        sources = self._build_sources(results)

        if not results:
            return RAGAnswer(
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
            return RAGAnswer(
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

        if not answer:
            raise RuntimeError(
                "The text generator returned an empty answer."
            )

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

        return RAGAnswer(
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
    ) -> tuple[RAGSource, ...]:
        sources: list[RAGSource] = []

        for number, result in enumerate(
            results,
            start=1,
        ):
            metadata = result.metadata

            sources.append(
                RAGSource(
                    number=number,
                    chunk_id=result.chunk_id,
                    doc_id=str(metadata["doc_id"]),
                    title=str(metadata["title"]),
                    section_id=str(
                        metadata["section_id"]
                    ),
                    section_title=str(
                        metadata["section_title"]
                    ),
                    score=float(result.score),
                )
            )

        return tuple(sources)

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
                        f"Retrieval score: {result.score:.4f}",
                        "Text:",
                        result.text.strip(),
                    ]
                )
            )

        context = "\n\n".join(context_blocks)

        return f"""
You are the Swiftrail Logistics knowledge assistant.

Answer the user's question using ONLY the authorized context below.

Rules:
1. Do not use outside knowledge.
2. Cite every factual claim using source numbers such as [1] or [2].
3. Never invent a policy, threshold, role, permission, or procedure.
4. If the context is insufficient, say exactly:
   "I could not find enough authorized information in the Swiftrail knowledge base to answer this question."
5. Keep the answer direct and operational.
6. The requesting role is: {role}

AUTHORIZED CONTEXT
------------------
{context}

USER QUESTION
-------------
{query}

ANSWER
------
""".strip()
