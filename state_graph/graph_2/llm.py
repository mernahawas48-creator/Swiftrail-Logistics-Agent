from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rag.naive_rag.generator import MistralTextGenerator, TextGenerator


def json_object(text: str, *, operation: str) -> dict[str, Any]:
    normalized = text.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        normalized = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Mistral {operation} did not return valid JSON.") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Mistral {operation} must return one JSON object.")
    return value


@dataclass(frozen=True, slots=True)
class PolicyAnalysis:
    summary: str
    delegated_limit_pct: float
    requires_human_above_limit: bool
    citations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "delegated_limit_pct": self.delegated_limit_pct,
            "requires_human_above_limit": self.requires_human_above_limit,
            "citations": list(self.citations),
        }


class MistralPolicyAnalyst:
    """RAG-grounded Mistral addition for the Graph 2 policy node."""

    def __init__(self, generator: TextGenerator | None = None) -> None:
        self.generator = generator or MistralTextGenerator(temperature=0.0)

    def analyze(self, *, evidence: list[dict[str, Any]]) -> PolicyAnalysis:
        if not evidence:
            raise ValueError("Policy analysis requires retrieved evidence.")
        prompt = f"""
You are the grounded policy-analysis node in a durable logistics state graph.
Use only the retrieved rate-exception evidence below.

Evidence: {json.dumps(evidence, default=str)}

Return one JSON object only. It must contain a short string `summary`, numeric
`delegated_limit_pct`, boolean `requires_human_above_limit`, and a non-empty
array of retrieved chunk IDs named `citations`.

Every citation must be a chunk_id from the supplied evidence. Do not infer a
numeric threshold that is absent from the evidence.
""".strip()
        value = json_object(
            self.generator.generate(prompt), operation="policy analysis"
        )
        summary = value.get("summary")
        limit = value.get("delegated_limit_pct")
        requires_human = value.get("requires_human_above_limit")
        citations = value.get("citations")
        if not isinstance(summary, str) or len(summary.strip()) < 10:
            raise RuntimeError("Mistral policy analysis returned an invalid summary.")
        if not isinstance(limit, (int, float)) or isinstance(limit, bool):
            raise TypeError("Mistral policy analysis returned an invalid limit.")
        normalized_limit = float(limit)
        if not 0 <= normalized_limit <= 100:
            raise RuntimeError("Mistral policy limit must be between 0 and 100.")
        if not isinstance(requires_human, bool):
            raise TypeError(
                "Mistral policy analysis returned an invalid escalation rule."
            )
        if requires_human is not True:
            raise RuntimeError(
                "Rate exceptions above delegated authority must require human review."
            )
        if not isinstance(citations, list) or not citations:
            raise RuntimeError("Mistral policy analysis must cite retrieved evidence.")
        available_ids = {str(item.get("chunk_id")) for item in evidence}
        normalized_citations = tuple(str(item) for item in citations)
        if any(item not in available_ids for item in normalized_citations):
            raise RuntimeError(
                "Mistral policy analysis cited evidence that was not retrieved."
            )
        evidence_text = " ".join(str(item.get("text", "")) for item in evidence)
        number = re.escape(f"{normalized_limit:g}")
        grounded_limit = re.search(
            rf"(?<![\d.]){number}(?:\.0+)?\s*(?:%|percent)?(?![\d.])",
            evidence_text,
            flags=re.IGNORECASE,
        )
        if grounded_limit is None:
            raise RuntimeError(
                "Mistral policy limit is not grounded in retrieved evidence."
            )
        return PolicyAnalysis(
            summary=summary.strip(),
            delegated_limit_pct=normalized_limit,
            requires_human_above_limit=requires_human,
            citations=normalized_citations,
        )
