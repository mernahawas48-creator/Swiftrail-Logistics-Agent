import json

import pytest

from state_graph.graph_2.llm import MistralPolicyAnalyst
from state_graph.graph_2.react import ConstrainedReActPlanner


class SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def evidence():
    return [
        {
            "chunk_id": "chunk-RE-2",
            "doc_id": "rate_exception_policy",
            "section_id": "RE-2",
            "text": "Discounts above 15 percent require finance-manager review.",
        }
    ]


def analysis_json(**overrides):
    value = {
        "summary": "Finance review is required above delegated authority.",
        "delegated_limit_pct": 15,
        "requires_human_above_limit": True,
        "citations": ["chunk-RE-2"],
    }
    value.update(overrides)
    return json.dumps(value)


def decision_json(**overrides):
    value = {
        "decision": "human_review",
        "rationale": "The 25 percent discount exceeds delegated authority.",
        "tool": "approve_rate_exception",
    }
    value.update(overrides)
    return json.dumps(value)


def test_graph_2_makes_grounded_rag_and_constrained_react_calls():
    generator = SequenceGenerator([analysis_json(), decision_json()])
    policy_analysis = MistralPolicyAnalyst(generator).analyze(evidence=evidence())
    decision = ConstrainedReActPlanner(generator).decide(
        shipment={"id": 5, "status": "pending"},
        exception={"id": 2, "discount_pct": 25},
        policy=evidence(),
        analysis=policy_analysis.to_dict(),
    )

    assert len(generator.prompts) == 2
    assert "retrieved rate-exception evidence" in generator.prompts[0]
    assert "Allowed MCP tools" in generator.prompts[1]
    assert policy_analysis.citations == ("chunk-RE-2",)
    assert decision.decision == "human_review"
    assert decision.tool == "approve_rate_exception"


def test_rag_policy_analysis_rejects_unretrieved_citation():
    generator = SequenceGenerator([analysis_json(citations=["invented-chunk"])])

    with pytest.raises(RuntimeError, match="was not retrieved"):
        MistralPolicyAnalyst(generator).analyze(evidence=evidence())


def test_constrained_react_cannot_bypass_authority_limit():
    generator = SequenceGenerator([decision_json(decision="auto_approve")])

    with pytest.raises(RuntimeError, match="bypass"):
        ConstrainedReActPlanner(generator).decide(
            shipment={"id": 5},
            exception={"id": 2, "discount_pct": 25},
            policy=evidence(),
            analysis={
                "delegated_limit_pct": 15,
                "requires_human_above_limit": True,
            },
        )


def test_constrained_react_rejects_tool_outside_mcp_allow_list():
    generator = SequenceGenerator([decision_json(tool="delete_shipment")])

    with pytest.raises(RuntimeError, match="allow-list"):
        ConstrainedReActPlanner(generator).decide(
            shipment={"id": 5},
            exception={"id": 2, "discount_pct": 10},
            policy=evidence(),
            analysis={
                "delegated_limit_pct": 15,
                "requires_human_above_limit": True,
            },
        )
