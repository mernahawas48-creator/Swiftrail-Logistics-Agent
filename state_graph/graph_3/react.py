from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rag.naive_rag.generator import MistralTextGenerator, TextGenerator
from state_graph.graph_2.llm import json_object


@dataclass(frozen=True, slots=True)
class CreditHoldReActDecision:
    decision: str
    rationale: str
    tool: str


class ConstrainedCreditHoldReActPlanner:
    """Choose the release path while enforcing MCP and authority constraints."""

    def __init__(self, generator: TextGenerator | None = None) -> None:
        self.generator = generator or MistralTextGenerator(temperature=0.0)

    def decide(
        self,
        *,
        hold: dict[str, Any],
        overdue_amount: float,
        plan: dict[str, Any],
        customer_evidence: str | None,
        payment_confirmed: float | None,
    ) -> CreditHoldReActDecision:
        allowed_tools = ["release_credit_hold"]
        prompt = f"""
You are the constrained ReAct decision node in a credit-hold state graph.
The observation phase and LATS planning phase are complete. Choose the next
safe action without executing it.

Allowed MCP tools: {json.dumps(allowed_tools)}
Active hold: {json.dumps(hold, default=str)}
Overdue amount: {overdue_amount}
Grounded LATS plan: {json.dumps(plan, default=str)}
Accepted dispute evidence: {json.dumps(customer_evidence)}
Confirmed payment: {json.dumps(payment_confirmed)}

Return JSON only:
{{"decision":"release|human_review", "rationale":"grounded reason", "tool":"release_credit_hold"}}

A severe hold must always use human_review. Never invent a tool, claim that a
release already succeeded, or bypass finance-manager approval.
""".strip()
        value = json_object(
            self.generator.generate(prompt),
            operation="Graph 3 constrained ReAct decision",
        )
        if value.get("tool") not in allowed_tools:
            raise RuntimeError(
                "Graph 3 constrained ReAct selected a tool outside the MCP allow-list."
            )
        if value.get("decision") not in {"release", "human_review"}:
            raise RuntimeError(
                "Graph 3 constrained ReAct returned an invalid decision."
            )
        rationale = value.get("rationale")
        if not isinstance(rationale, str) or len(rationale.strip()) < 10:
            raise RuntimeError(
                "Graph 3 constrained ReAct returned an invalid rationale."
            )
        if (
            str(hold.get("severity", "")).lower() == "severe"
            and value["decision"] != "human_review"
        ):
            raise RuntimeError(
                "Graph 3 constrained ReAct attempted to bypass severe-hold HITL."
            )
        return CreditHoldReActDecision(
            decision=value["decision"],
            rationale=rationale.strip(),
            tool=value["tool"],
        )
