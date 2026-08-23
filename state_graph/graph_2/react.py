from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rag.naive_rag.generator import MistralTextGenerator, TextGenerator
from state_graph.graph_2.llm import json_object


@dataclass(frozen=True)
class ReActDecision:
    decision: str
    rationale: str
    tool: str


class ConstrainedReActPlanner:
    """Mistral decision node constrained by policy and an MCP allow-list."""

    def __init__(
        self,
        generator: TextGenerator | None = None,
        *,
        llm: Any | None = None,
    ) -> None:
        if generator is not None and llm is not None:
            raise ValueError("Pass either generator or llm, not both.")
        if llm is not None:
            generator = _InvokeGenerator(llm)
        self.generator = generator or MistralTextGenerator(temperature=0.0)

    def decide(
        self,
        *,
        shipment: dict[str, Any],
        exception: dict[str, Any],
        policy: list[dict[str, Any]],
        analysis: dict[str, Any] | None = None,
    ) -> ReActDecision:
        allowed_tools = ["approve_rate_exception"]
        policy_analysis = analysis or {
            "delegated_limit_pct": 15.0,
            "requires_human_above_limit": True,
        }
        prompt = f"""
You are the constrained ReAct decision node in a logistics approval graph.
The observation phase has already collected the shipment, exception, and
grounded policy evidence. Select only a safe next decision and allowed action.

Allowed MCP tools: {json.dumps(allowed_tools)}
Shipment: {json.dumps(shipment, default=str)}
Rate exception: {json.dumps(exception, default=str)}
Policy evidence: {json.dumps(policy, default=str)}
Grounded policy analysis: {json.dumps(policy_analysis, default=str)}

Return JSON only with:
{{"decision":"auto_approve|human_review", "rationale":"short grounded reason", "tool":"one allowed MCP tool"}}
Never invent a tool or approve above the grounded delegated limit.
""".strip()
        data = json_object(
            self.generator.generate(prompt), operation="constrained ReAct decision"
        )
        if data.get("tool") not in allowed_tools:
            raise RuntimeError(
                "Constrained ReAct selected a tool outside the MCP allow-list"
            )
        if data.get("decision") not in {"auto_approve", "human_review"}:
            raise RuntimeError("Constrained ReAct returned an invalid decision")
        rationale = data.get("rationale")
        if not isinstance(rationale, str) or len(rationale.strip()) < 10:
            raise RuntimeError("Constrained ReAct returned an invalid rationale")
        discount = float(exception["discount_pct"])
        limit = float(policy_analysis["delegated_limit_pct"])
        requires_human = bool(policy_analysis["requires_human_above_limit"])
        if requires_human and discount > limit and data["decision"] != "human_review":
            raise RuntimeError(
                "Constrained ReAct attempted to bypass the grounded authority limit"
            )
        return ReActDecision(
            decision=data["decision"],
            rationale=rationale.strip(),
            tool=data["tool"],
        )


class _InvokeGenerator:
    """Compatibility adapter for callers that still inject a LangChain model."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            raise TypeError("Constrained ReAct model returned unsupported content")
        return content
