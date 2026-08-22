from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReActDecision:
    decision: str
    rationale: str
    tool: str


class ConstrainedReActPlanner:
    """Small constrained ReAct planner for Graph 2.

    The model is never allowed to invent a tool name. The final action must be
    one of the registered MCP operations supplied to this planner.
    """

    def __init__(self, llm: Any | None = None):
        self.llm = llm

    def decide(self, *, shipment: dict[str, Any], exception: dict[str, Any], policy: list[dict[str, Any]]) -> ReActDecision:
        allowed_tools = ["get_shipment_status", "get_shipment_rate_exception", "approve_rate_exception"]
        if self.llm is None:
            discount = float(exception["discount_pct"])
            return ReActDecision(
                decision="human_review" if discount > 15 else "auto_approve",
                rationale="The decision is constrained by the 15% delegated authority boundary.",
                tool="approve_rate_exception",
            )

        prompt = f"""You are the constrained decision node in a logistics approval workflow.
Allowed tools: {json.dumps(allowed_tools)}
Never invent another tool.
Shipment: {json.dumps(shipment, default=str)}
Rate exception: {json.dumps(exception, default=str)}
Policy evidence: {json.dumps(policy, default=str)}

Return JSON only with:
{{"decision":"auto_approve|human_review", "rationale":"...", "tool":"one allowed tool"}}
"""
        response = self.llm.invoke(prompt)
        text = getattr(response, "content", response)
        if not isinstance(text, str):
            raise TypeError("Constrained ReAct model returned unsupported content")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Constrained ReAct model did not return valid JSON") from exc
        if data.get("tool") not in allowed_tools:
            raise RuntimeError("Constrained ReAct selected a tool outside the MCP allow-list")
        if data.get("decision") not in {"auto_approve", "human_review"}:
            raise RuntimeError("Constrained ReAct returned an invalid decision")
        return ReActDecision(
            decision=data["decision"],
            rationale=str(data.get("rationale", "")),
            tool=data["tool"],
        )
