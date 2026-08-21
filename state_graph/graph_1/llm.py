from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rag.naive_rag.generator import MistralTextGenerator, TextGenerator


def _json_object(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        normalized = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Mistral decomposition did not return valid JSON.") from exc
    if not isinstance(value, dict):
        raise TypeError("Mistral decomposition must return one JSON object.")
    return value


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    steps: tuple[str, ...]
    customer_question: str
    policy_query: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": list(self.steps),
            "customer_question": self.customer_question,
            "policy_query": self.policy_query,
        }


class MistralRecoveryDecomposer:
    """Task-decomposition LLM addition used by Graph 1."""

    def __init__(self, generator: TextGenerator | None = None) -> None:
        self.generator = generator or MistralTextGenerator(temperature=0.0)

    def decompose(
        self,
        *,
        shipment: dict[str, Any],
        failure_reason: str,
    ) -> RecoveryPlan:
        prompt = f"""
You are the task-decomposition node in a durable logistics state graph.
Build a recovery plan for this delivery exception.

Shipment: {json.dumps(shipment, default=str)}
Failure reason: {failure_reason}

Return JSON only with exactly these fields:
{{
  "steps": ["4 to 10 short ordered recovery steps"],
  "customer_question": "one question that collects the customer's choice",
  "policy_query": "one focused query for delivery and rerouting policy"
}}

The steps must include policy verification, customer input, safe execution,
and final verification. Do not claim that an action already happened.
""".strip()
        value = _json_object(self.generator.generate(prompt))
        steps = value.get("steps")
        question = value.get("customer_question")
        policy_query = value.get("policy_query")
        if (
            not isinstance(steps, list)
            or not 4 <= len(steps) <= 10
            or any(not isinstance(step, str) or len(step.strip()) < 4 for step in steps)
        ):
            raise RuntimeError("Mistral decomposition returned invalid steps.")
        if not isinstance(question, str) or len(question.strip()) < 10:
            raise RuntimeError("Mistral decomposition returned an invalid question.")
        if not isinstance(policy_query, str) or len(policy_query.strip()) < 10:
            raise RuntimeError("Mistral decomposition returned an invalid policy query.")
        return RecoveryPlan(
            steps=tuple(step.strip() for step in steps),
            customer_question=question.strip(),
            policy_query=policy_query.strip(),
        )
