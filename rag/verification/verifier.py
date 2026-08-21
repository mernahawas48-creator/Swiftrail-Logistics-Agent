"""Deterministic Self-RAG-style verification for retrieved evidence and answers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SECTION_ID_PATTERN = re.compile(
    r"^[A-Z]{2,5}-\d+(?:\.\d+)?$",
    re.IGNORECASE,
)
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?<!\[)\b\d+(?:\.\d+)?%?\b(?!\])")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can",
    "could", "do", "does", "for", "from", "has", "have", "how", "i", "if",
    "in", "is", "it", "may", "must", "of", "on", "or", "should", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would",
}

SAFE_ANSWER_PREFIXES = (
    "i could not find enough authorized",
    "i could not find enough authorized evidence",
)


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """Result of one explicit verification stage."""

    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    """Post-retrieval and post-generation verification results."""

    retrieval_relevant: bool
    answer_supported: bool
    citations_valid: bool
    reason: str

    @property
    def passed(self) -> bool:
        return (
            self.retrieval_relevant
            and self.answer_supported
            and self.citations_valid
        )


class SelfRAGVerifier:
    """Check relevance before generation and support after generation.

    The verifier is intentionally deterministic and auditable:
    - retrieval relevance uses exact section-ID matching or content-term
      coverage;
    - answer support is checked sentence by sentence against the cited
      evidence;
    - numeric claims must occur in the cited evidence or be scenario values explicitly supplied by the user;
    - every factual sentence must contain valid numbered citations.

    A failure has a visible consequence in the RAG pipelines: generation is
    skipped for irrelevant retrieval, or an unsupported generated answer is
    replaced with a safe abstention.
    """

    def __init__(
        self,
        *,
        minimum_relevance_coverage: float = 0.35,
        minimum_sentence_support: float = 0.30,
    ):
        if not 0.0 <= minimum_relevance_coverage <= 1.0:
            raise ValueError(
                "minimum_relevance_coverage must be between 0 and 1."
            )
        if not 0.0 <= minimum_sentence_support <= 1.0:
            raise ValueError(
                "minimum_sentence_support must be between 0 and 1."
            )

        self.minimum_relevance_coverage = minimum_relevance_coverage
        self.minimum_sentence_support = minimum_sentence_support

    def check_relevance(
        self,
        query: str,
        results: Sequence[Any],
    ) -> VerificationCheck:
        """Verify that retrieved content is relevant to the user query."""

        normalized_query = query.strip()

        if not results:
            return VerificationCheck(
                passed=False,
                reason="No authorized evidence was retrieved.",
            )

        if SECTION_ID_PATTERN.fullmatch(normalized_query):
            requested = normalized_query.upper()
            section_ids = {
                str(result.metadata.get("section_id", "")).upper()
                for result in results
            }
            if requested in section_ids:
                return VerificationCheck(
                    passed=True,
                    reason=f"Exact requested section {requested} was retrieved.",
                )
            return VerificationCheck(
                passed=False,
                reason=f"Exact requested section {requested} was not retrieved.",
            )

        query_terms = self._content_terms(normalized_query)
        if not query_terms:
            return VerificationCheck(
                passed=False,
                reason="The query has no content terms that can be verified.",
            )

        evidence_terms: set[str] = set()
        for result in results:
            evidence_terms.update(
                self._content_terms(self._result_text(result))
            )

        matched = query_terms.intersection(evidence_terms)
        coverage = len(matched) / len(query_terms)

        if coverage >= self.minimum_relevance_coverage:
            return VerificationCheck(
                passed=True,
                reason=(
                    "Retrieved evidence passed the relevance check "
                    f"({len(matched)}/{len(query_terms)} content terms matched)."
                ),
            )

        return VerificationCheck(
            passed=False,
            reason=(
                "Retrieved evidence failed the relevance check "
                f"({len(matched)}/{len(query_terms)} content terms matched)."
            ),
        )

    def check_support(
        self,
        answer: str,
        results: Sequence[Any],
        *,
        query: str | None = None,
    ) -> VerificationCheck:
        """Verify generated factual sentences against their cited chunks.

        Numeric values explicitly supplied in the user's query are treated
        as scenario inputs, not as facts that the knowledge base must repeat.
        """

        normalized_answer = answer.strip()
        if not normalized_answer:
            return VerificationCheck(
                passed=False,
                reason="The generated answer is empty.",
            )

        if normalized_answer.lower().startswith(SAFE_ANSWER_PREFIXES):
            return VerificationCheck(
                passed=True,
                reason="The model returned a safe abstention.",
            )

        if not results:
            return VerificationCheck(
                passed=False,
                reason="An answer was generated without retrieved evidence.",
            )

        sentences = self._sentences(normalized_answer)
        if not sentences:
            return VerificationCheck(
                passed=False,
                reason="No verifiable answer sentence was found.",
            )

        for sentence in sentences:
            citation_numbers = [
                int(value)
                for value in CITATION_PATTERN.findall(sentence)
            ]

            if not citation_numbers:
                return VerificationCheck(
                    passed=False,
                    reason=(
                        "A factual sentence has no numbered citation: "
                        f"{sentence}"
                    ),
                )

            invalid = [
                number
                for number in citation_numbers
                if number < 1 or number > len(results)
            ]
            if invalid:
                return VerificationCheck(
                    passed=False,
                    reason=(
                        "The answer contains an invalid citation number: "
                        + ", ".join(str(value) for value in invalid)
                    ),
                )

            cited_results = [
                results[number - 1]
                for number in citation_numbers
            ]
            evidence_text = " ".join(
                self._result_text(result)
                for result in cited_results
            )

            sentence_without_citations = CITATION_PATTERN.sub(
                " ",
                sentence,
            )
            claim_terms = self._content_terms(
                sentence_without_citations
            )
            evidence_terms = self._content_terms(evidence_text)

            if claim_terms:
                supported_terms = claim_terms.intersection(evidence_terms)
                coverage = len(supported_terms) / len(claim_terms)
                if coverage < self.minimum_sentence_support:
                    return VerificationCheck(
                        passed=False,
                        reason=(
                            "A cited sentence has weak support in its evidence "
                            f"({len(supported_terms)}/{len(claim_terms)} "
                            f"content terms matched): {sentence}"
                        ),
                    )

            claim_numbers = {
                value.lower()
                for value in NUMBER_PATTERN.findall(
                    sentence_without_citations
                )
            }
            evidence_numbers = {
                value.lower()
                for value in NUMBER_PATTERN.findall(evidence_text)
            }
            query_numbers = {
                value.lower()
                for value in NUMBER_PATTERN.findall(query or "")
            }
            allowed_numbers = evidence_numbers.union(query_numbers)
            missing_numbers = claim_numbers.difference(allowed_numbers)

            if missing_numbers:
                return VerificationCheck(
                    passed=False,
                    reason=(
                        "A numeric claim is not present in the cited evidence: "
                        + ", ".join(sorted(missing_numbers))
                    ),
                )

        return VerificationCheck(
            passed=True,
            reason="Every factual sentence is supported by valid cited evidence.",
        )

    def summarize(
        self,
        relevance: VerificationCheck,
        support: VerificationCheck,
    ) -> VerificationSummary:
        """Create a compact response-level verification record."""

        citations_valid = (
            support.passed
            or "citation" not in support.reason.lower()
        )

        reason = (
            f"Retrieval: {relevance.reason} "
            f"Answer: {support.reason}"
        )

        return VerificationSummary(
            retrieval_relevant=relevance.passed,
            answer_supported=support.passed,
            citations_valid=citations_valid,
            reason=reason,
        )

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        return {
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if (
                token.lower() not in STOPWORDS
                and len(token) > 1
            )
        }

    @staticmethod
    def _result_text(result: Any) -> str:
        metadata = result.metadata
        keywords = metadata.get("keywords", [])
        return " ".join(
            [
                str(result.text),
                str(metadata.get("title", "")),
                str(metadata.get("section_id", "")),
                str(metadata.get("section_title", "")),
                " ".join(str(value) for value in keywords),
            ]
        )

    @staticmethod
    def _sentences(answer: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", answer).strip()
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
        return [piece.strip() for piece in pieces if piece.strip()]
