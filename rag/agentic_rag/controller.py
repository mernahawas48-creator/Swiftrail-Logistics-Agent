"""Plan, retrieve, grade, retry, and answer with authorized evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, ClassVar, Protocol

from rag.agentic_rag.models import (
    AgenticRAGAnswer,
    AgenticRAGSource,
    AgentTraceStep,
    EvidenceAssessment,
    RetrievalPlan,
)
from rag.naive_rag.generator import (
    MistralTextGenerator,
    TextGenerator,
)
from rag.verification.verifier import (
    SelfRAGVerifier,
)

SECTION_ID_PATTERN = re.compile(
    r"^[A-Z]{2,5}-\d+(?:\.\d+)?$",
    re.IGNORECASE,
)

WORD_PATTERN = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*",
    re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

SAFE_NO_EVIDENCE_ANSWER = (
    "I could not find enough authorized evidence in the "
    "Swiftrail knowledge base to answer this question."
)


class Retriever(Protocol):
    """Interface required from the hybrid-search tool."""

    def search(self, query: str, **kwargs: Any) -> list[Any]:
        ...


class Planner(Protocol):
    def plan(
        self,
        query: str,
        *,
        attempt: int,
        top_k: int,
    ) -> RetrievalPlan:
        ...


class Grader(Protocol):
    def grade(
        self,
        original_query: str,
        results: Sequence[Any],
    ) -> EvidenceAssessment:
        ...


class Rewriter(Protocol):
    def rewrite(
        self,
        original_query: str,
        results: Sequence[Any],
    ) -> str:
        ...


class HeuristicPlanner:
    """Choose an exact-ID or balanced hybrid-search strategy."""

    def plan(
        self,
        query: str,
        *,
        attempt: int,
        top_k: int,
    ) -> RetrievalPlan:
        normalized_query = query.strip()

        if SECTION_ID_PATTERN.fullmatch(
            normalized_query
        ):
            return RetrievalPlan(
                query=normalized_query.upper(),
                top_k=top_k,
                candidate_k=max(top_k * 4, 20),
                dense_weight=0.5,
                sparse_weight=1.5,
                section_ids=(
                    normalized_query.upper(),
                ),
                reason=(
                    "The query is an exact section identifier, "
                    "so lexical retrieval and an exact metadata "
                    "filter receive priority."
                ),
            )

        if attempt == 1:
            return RetrievalPlan(
                query=normalized_query,
                top_k=top_k,
                candidate_k=max(top_k * 4, 20),
                dense_weight=1.0,
                sparse_weight=1.0,
                section_ids=None,
                reason=(
                    "Use balanced dense and BM25 retrieval "
                    "for the first attempt."
                ),
            )

        return RetrievalPlan(
            query=normalized_query,
            top_k=max(top_k, 5),
            candidate_k=max(top_k * 6, 30),
            dense_weight=0.75,
            sparse_weight=1.25,
            section_ids=None,
            reason=(
                "Use a broader candidate pool and slightly "
                "stronger lexical retrieval after rewriting."
            ),
        )


class KeywordEvidenceGrader:
    """Grade evidence using exact IDs and query-term coverage."""

    def __init__(
        self,
        *,
        minimum_matches: int = 2,
        minimum_coverage: float = 0.35,
    ):
        if minimum_matches < 1:
            raise ValueError(
                "minimum_matches must be positive."
            )

        if not 0.0 <= minimum_coverage <= 1.0:
            raise ValueError(
                "minimum_coverage must be between 0 and 1."
            )

        self.minimum_matches = minimum_matches
        self.minimum_coverage = minimum_coverage

    def grade(
        self,
        original_query: str,
        results: Sequence[Any],
    ) -> EvidenceAssessment:
        if not results:
            return EvidenceAssessment(
                sufficient=False,
                reason="No authorized results were retrieved.",
            )

        normalized_query = original_query.strip()

        if SECTION_ID_PATTERN.fullmatch(
            normalized_query
        ):
            expected = normalized_query.upper()

            for result in results:
                if (
                    str(
                        result.metadata.get(
                            "section_id",
                            "",
                        )
                    ).upper()
                    == expected
                ):
                    return EvidenceAssessment(
                        sufficient=True,
                        reason=(
                            "The exact requested section ID "
                            "was retrieved."
                        ),
                        matched_terms=(expected,),
                    )

            return EvidenceAssessment(
                sufficient=False,
                reason=(
                    "The exact requested section ID was not "
                    "present in the retrieved evidence."
                ),
            )

        query_terms = self._content_terms(
            normalized_query
        )

        if not query_terms:
            return EvidenceAssessment(
                sufficient=False,
                reason=(
                    "The query did not contain enough content "
                    "terms to grade the evidence."
                ),
            )

        evidence_text = " ".join(
            self._result_text(result)
            for result in results[:3]
        ).lower()
        full_evidence_text = " ".join(
            self._result_text(result)
            for result in results
        ).lower()

        matched_terms = tuple(
            sorted(
                term
                for term in query_terms
                if term in evidence_text
            )
        )

        coverage = (
            len(matched_terms)
            / len(query_terms)
        )

        multi_part = (
            " and " in normalized_query.lower()
            or ";" in normalized_query
            or normalized_query.count("?") > 1
        )
        required_coverage = (
            max(self.minimum_coverage, 0.45)
            if multi_part
            else self.minimum_coverage
        )

        broad_multi_part = (
            multi_part
            and len(query_terms) >= 8
        )
        enough_evidence = (
            len(results) >= 3
            if broad_multi_part
            else True
        )

        sufficient = (
            enough_evidence
            and len(matched_terms) >= self.minimum_matches
            and coverage >= required_coverage
        )

        # Multi-part questions must have evidence for the operational
        # facets they ask about, not merely broad lexical overlap. In
        # particular, a severe-hold release question is incomplete until
        # the evidence states the release authority and confirmation step.
        query_lower = normalized_query.lower()
        missing_facets: list[str] = []

        if (
            multi_part
            and "severe" in query_lower
            and "hold" in query_lower
            and "release" in query_lower
        ):
            has_release_authority = (
                "finance manager" in full_evidence_text
                and "release" in full_evidence_text
                and (
                    "human confirmation" in full_evidence_text
                    or "authorization note" in full_evidence_text
                )
            )
            if not has_release_authority:
                missing_facets.append("severe-hold release authority")

        if missing_facets:
            sufficient = False

        if sufficient:
            reason = (
                "The top evidence covers enough of the "
                "question's content terms."
            )
        elif missing_facets:
            reason = (
                "The retrieved evidence is missing required multi-part "
                "policy evidence: " + ", ".join(missing_facets) + "."
            )
        else:
            reason = (
                "The retrieved evidence has weak lexical "
                "coverage of the original question."
            )

        return EvidenceAssessment(
            sufficient=sufficient,
            reason=reason,
            matched_terms=matched_terms,
        )

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        return {
            token.lower()
            for token in WORD_PATTERN.findall(text)
            if (
                token.lower() not in STOPWORDS
                and len(token) > 1
            )
        }

    @staticmethod
    def _result_text(result: Any) -> str:
        metadata = result.metadata

        return " ".join(
            [
                str(result.text),
                str(metadata.get("title", "")),
                str(
                    metadata.get(
                        "section_title",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "section_id",
                        "",
                    )
                ),
                " ".join(
                    str(value)
                    for value in metadata.get(
                        "keywords",
                        [],
                    )
                ),
            ]
        )


class CorpusQueryRewriter:
    """Expand weak queries using corpus-specific operational terms."""

    EXPANSIONS: ClassVar[dict[str, str]] = {
        "access": (
            "employee access role permission authentication"
        ),
        "approve": (
            "approval authority authorization delegated limit"
        ),
        "discount": (
            "discount rate exception delegated authority approval"
        ),
        "hold": (
            "credit hold severity release authorization"
        ),
        "invoice": (
            "invoice collection payment overdue escalation"
        ),
        "portfolio": (
            "portfolio risk review priority exposure"
        ),
        "release": (
            "release permission authorization confirmation"
        ),
        "shipment": (
            "shipment pricing rate reference exception"
        ),
    }

    def rewrite(
        self,
        original_query: str,
        results: Sequence[Any],
    ) -> str:
        normalized_query = original_query.lower()
        multi_part = (
            " and " in normalized_query
            or ";" in original_query
            or original_query.count("?") > 1
        )

        # Multi-part operational questions are decomposed into the corpus
        # facets that must be retrieved together. This intentionally removes
        # narrative filler and makes the retry target the missing policy
        # evidence rather than repeating the original long question.
        if multi_part:
            facet_terms: list[str] = []
            if "discount" in normalized_query:
                facet_terms.extend([
                    "above authority discount",
                    "rate exception",
                    "finance manager",
                    "human approval",
                ])
            if (
                "shipment" in normalized_query
                or "pricing" in normalized_query
            ):
                facet_terms.extend([
                    "shipment pricing adjustment",
                    "pricing approval",
                    "separate workflow",
                ])
            if (
                "hold" in normalized_query
                or "release" in normalized_query
                or "released" in normalized_query
            ):
                facet_terms.extend([
                    "severe credit hold release",
                    "finance manager",
                    "human confirmation",
                    "authorization note",
                    "separate workflow",
                ])

            if facet_terms:
                return " ".join(dict.fromkeys(facet_terms))

        tokens = [
            token.lower()
            for token in WORD_PATTERN.findall(
                original_query
            )
            if token.lower() not in STOPWORDS
        ]

        parts: list[str] = []
        seen: set[str] = set()

        for token in tokens:
            if token not in seen:
                parts.append(token)
                seen.add(token)

            expansion = self.EXPANSIONS.get(token)
            if expansion:
                for expanded_token in (
                    expansion.split()
                ):
                    if expanded_token not in seen:
                        parts.append(expanded_token)
                        seen.add(expanded_token)

        for result in results[:2]:
            metadata = result.metadata

            for value in (
                metadata.get("section_id"),
                metadata.get("section_title"),
                metadata.get("title"),
            ):
                if value:
                    for token in WORD_PATTERN.findall(
                        str(value)
                    ):
                        normalized = token.lower()
                        if normalized not in seen:
                            parts.append(normalized)
                            seen.add(normalized)

        rewritten = " ".join(parts).strip()

        return rewritten or original_query.strip()


class AgenticRAG:
    """Autonomous RAG controller with evidence grading and retry."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        generator: TextGenerator | None = None,
        planner: Planner | None = None,
        grader: Grader | None = None,
        rewriter: Rewriter | None = None,
        verifier: SelfRAGVerifier | None = None,
        max_attempts: int = 2,
    ):
        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        if retriever is None:
            from rag.hybrid_search.search import (
                HybridSearch,
            )

            retriever = HybridSearch()

        self.retriever = retriever
        self.generator = (
            generator or MistralTextGenerator()
        )
        self.planner = (
            planner or HeuristicPlanner()
        )
        self.grader = (
            grader or KeywordEvidenceGrader()
        )
        self.rewriter = (
            rewriter or CorpusQueryRewriter()
        )
        self.verifier = (
            verifier or SelfRAGVerifier()
        )
        self.max_attempts = max_attempts

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
    ) -> AgenticRAGAnswer:
        """Run the complete agent loop for one user question."""

        original_query = query.strip()

        if not original_query:
            raise ValueError(
                "The query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        trace: list[AgentTraceStep] = []
        step_number = 1
        retrieval_query = original_query
        final_results: list[Any] = []
        accumulated_results: list[Any] = []
        final_assessment = EvidenceAssessment(
            sufficient=False,
            reason="No retrieval attempt has run.",
        )
        attempts_used = 0

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            attempts_used = attempt

            plan = self.planner.plan(
                retrieval_query,
                attempt=attempt,
                top_k=top_k,
            )

            trace.append(
                AgentTraceStep(
                    step=step_number,
                    action="plan",
                    details=asdict(plan),
                )
            )
            step_number += 1

            results = self.retriever.search(
                plan.query,
                role=role,
                top_k=plan.top_k,
                candidate_k=plan.candidate_k,
                dense_weight=plan.dense_weight,
                sparse_weight=plan.sparse_weight,
                statuses=tuple(statuses),
                departments=departments,
                document_types=document_types,
                doc_ids=doc_ids,
                section_ids=plan.section_ids,
            )

            accumulated_results = self._merge_results(
                accumulated_results,
                results,
            )
            final_results = accumulated_results

            trace.append(
                AgentTraceStep(
                    step=step_number,
                    action="retrieve",
                    details={
                        "attempt": attempt,
                        "query": plan.query,
                        "result_count": len(results),
                        "top_sections": [
                            result.metadata.get(
                                "section_id"
                            )
                            for result in results[:3]
                        ],
                        "accumulated_result_count": len(
                            final_results
                        ),
                        "accumulated_sections": [
                            result.metadata.get(
                                "section_id"
                            )
                            for result in final_results
                        ],
                    },
                )
            )
            step_number += 1

            assessment = self.grader.grade(
                original_query,
                final_results,
            )
            final_assessment = assessment

            trace.append(
                AgentTraceStep(
                    step=step_number,
                    action="grade_evidence",
                    details=asdict(assessment),
                )
            )
            step_number += 1

            if assessment.sufficient:
                retrieval_query = plan.query
                break

            if attempt < self.max_attempts:
                rewritten_query = self.rewriter.rewrite(
                    original_query,
                    final_results,
                )

                trace.append(
                    AgentTraceStep(
                        step=step_number,
                        action="rewrite_query",
                        details={
                            "original_query": (
                                original_query
                            ),
                            "previous_query": (
                                plan.query
                            ),
                            "rewritten_query": (
                                rewritten_query
                            ),
                            "reason": (
                                assessment.reason
                            ),
                        },
                    )
                )
                step_number += 1
                retrieval_query = rewritten_query

        sources = self._build_sources(
            final_results
        )

        if not final_assessment.sufficient:
            trace.append(
                AgentTraceStep(
                    step=step_number,
                    action="stop",
                    details={
                        "reason": (
                            final_assessment.reason
                        ),
                        "generated_answer": False,
                    },
                )
            )

            return AgenticRAGAnswer(
                query=original_query,
                answer=SAFE_NO_EVIDENCE_ANSWER,
                sources=sources,
                attempts=attempts_used,
                final_retrieval_query=retrieval_query,
                model_name=self._model_name(),
                trace=tuple(trace),
                verification_passed=False,
                verification_reason=final_assessment.reason,
            )

        relevance = self.verifier.check_relevance(
            original_query,
            final_results,
        )
        trace.append(
            AgentTraceStep(
                step=step_number,
                action="verify_retrieval",
                details={
                    "passed": relevance.passed,
                    "reason": relevance.reason,
                },
            )
        )
        step_number += 1

        if not relevance.passed:
            trace.append(
                AgentTraceStep(
                    step=step_number,
                    action="stop",
                    details={
                        "reason": relevance.reason,
                        "generated_answer": False,
                    },
                )
            )
            return AgenticRAGAnswer(
                query=original_query,
                answer=SAFE_NO_EVIDENCE_ANSWER,
                sources=sources,
                attempts=attempts_used,
                final_retrieval_query=retrieval_query,
                model_name=self._model_name(),
                trace=tuple(trace),
                verification_passed=False,
                verification_reason=relevance.reason,
            )

        prompt = self._build_prompt(
            query=original_query,
            role=role,
            retrieval_query=retrieval_query,
            results=final_results,
            trace=trace,
        )

        answer = self.generator.generate(
            prompt
        ).strip()

        if not answer:
            raise RuntimeError(
                "The text generator returned an empty answer."
            )

        trace.append(
            AgentTraceStep(
                step=step_number,
                action="generate_answer",
                details={
                    "model": self._model_name(),
                    "source_count": len(sources),
                },
            )
        )
        step_number += 1

        support = self.verifier.check_support(
            answer,
            final_results,
            query=original_query,
        )
        trace.append(
            AgentTraceStep(
                step=step_number,
                action="verify_answer",
                details={
                    "passed": support.passed,
                    "reason": support.reason,
                },
            )
        )

        if not support.passed:
            answer = SAFE_NO_EVIDENCE_ANSWER

        return AgenticRAGAnswer(
            query=original_query,
            answer=answer,
            sources=sources,
            attempts=attempts_used,
            final_retrieval_query=retrieval_query,
            model_name=self._model_name(),
            trace=tuple(trace),
            verification_passed=support.passed,
            verification_reason=support.reason,
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
    def _merge_results(
        existing: Sequence[Any],
        new_results: Sequence[Any],
    ) -> list[Any]:
        """Keep evidence from every retrieval round without duplicates."""
        merged: list[Any] = []
        seen: set[str] = set()

        for result in [*existing, *new_results]:
            key = str(
                getattr(
                    result,
                    "chunk_id",
                    (
                        result.metadata.get("doc_id", ""),
                        result.metadata.get("section_id", ""),
                        getattr(result, "text", ""),
                    ),
                )
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)

        return merged

    @staticmethod
    def _build_sources(
        results: Sequence[Any],
    ) -> tuple[AgenticRAGSource, ...]:
        sources: list[AgenticRAGSource] = []

        for number, result in enumerate(
            results,
            start=1,
        ):
            metadata = result.metadata

            sources.append(
                AgenticRAGSource(
                    number=number,
                    chunk_id=str(
                        result.chunk_id
                    ),
                    doc_id=str(
                        metadata["doc_id"]
                    ),
                    title=str(
                        metadata["title"]
                    ),
                    section_id=str(
                        metadata["section_id"]
                    ),
                    section_title=str(
                        metadata["section_title"]
                    ),
                    fused_score=float(
                        result.fused_score
                    ),
                    dense_rank=(
                        result.dense_rank
                    ),
                    sparse_rank=(
                        result.sparse_rank
                    ),
                )
            )

        return tuple(sources)

    @staticmethod
    def _build_prompt(
        *,
        query: str,
        role: str,
        retrieval_query: str,
        results: Sequence[Any],
        trace: Sequence[AgentTraceStep],
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

        context = "\n\n".join(
            context_blocks
        )

        trace_summary = "\n".join(
            f"- {item.action}: {item.details}"
            for item in trace
        )

        return f"""
You are the Swiftrail Logistics knowledge assistant.

The retrieval agent has already planned, searched, graded the evidence, and
possibly rewritten the query. Answer using ONLY the authorized evidence.

Rules:
1. Cite every factual claim with source numbers such as [1] or [2].
2. Do not invent a policy, threshold, role, permission, or procedure.
3. Respect the authenticated role and do not reveal unauthorized content.
4. If the evidence does not directly support a detail, do not include it.
5. Keep the answer direct and operational.

Authenticated role: {role}
Original question: {query}
Final retrieval query: {retrieval_query}

AGENT TRACE
-----------
{trace_summary}

AUTHORIZED EVIDENCE
-------------------
{context}

FINAL ANSWER
------------
""".strip()
