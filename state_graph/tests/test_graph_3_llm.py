import json
from types import SimpleNamespace

import pytest

from state_graph.graph_3.llm import (
    CreditHoldPlanningEnvironment,
    MistralLATSRemediationPlanner,
)
from state_graph.graph_3.react import ConstrainedCreditHoldReActPlanner


class SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, prompt):
        assert "release_credit_hold" in prompt
        self.calls += 1
        return self.outputs.pop(0)


def test_credit_hold_environment_accepts_grounded_severe_plan():
    environment = CreditHoldPlanningEnvironment(
        customer_claim="invoice is duplicated",
        hold={"severity": "severe"},
    )

    result = environment.evaluate(
        "Request invoice evidence, then obtain finance approval. "
        "ACTION: dispute_review"
    )

    assert result.success is True
    assert result.score == 0.95


def test_credit_hold_environment_rejects_wrong_or_unsafe_branch():
    environment = CreditHoldPlanningEnvironment(
        customer_claim="invoice is duplicated",
        hold={"severity": "severe"},
    )

    result = environment.evaluate("Release succeeded. ACTION: payment_confirmation")

    assert result.success is False
    assert len(result.details) == 3


def test_lats_adapter_returns_auditable_grounded_plan():
    calls = []

    def runner(task, llm, environment, **options):
        calls.append((task, llm, options))
        output = "Collect payment and preserve finance approval. ACTION: payment_confirmation"
        assert environment.evaluate(output).success is True
        return SimpleNamespace(
            success=True,
            output=output,
            best_score=0.95,
            iterations=1,
            root=None,
        )

    marker_llm = object()
    planner = MistralLATSRemediationPlanner(marker_llm, runner=runner)
    plan = planner.plan(
        invoices=[{"id": 5, "amount": "12000.00", "paid_status": "overdue"}],
        hold={"id": 3, "severity": "severe"},
        overdue_amount=12000,
        customer_claim=None,
    )

    assert plan.action == "payment_confirmation"
    assert plan.score == 0.95
    assert calls[0][1] is marker_llm
    assert calls[0][2] == {"iterations": 2, "n_actions": 2}


def test_constrained_react_accepts_severe_human_review():
    generator = SequenceGenerator(
        [
            json.dumps(
                {
                    "decision": "human_review",
                    "rationale": "Severe holds require finance-manager approval.",
                    "tool": "release_credit_hold",
                }
            )
        ]
    )
    planner = ConstrainedCreditHoldReActPlanner(generator)

    decision = planner.decide(
        hold={"id": 3, "severity": "severe"},
        overdue_amount=12000,
        plan={"action": "payment_confirmation"},
        customer_evidence=None,
        payment_confirmed=12000,
    )

    assert generator.calls == 1
    assert decision.decision == "human_review"


def test_constrained_react_blocks_severe_hold_bypass():
    generator = SequenceGenerator(
        [
            json.dumps(
                {
                    "decision": "release",
                    "rationale": "The customer supplied a complete payment confirmation.",
                    "tool": "release_credit_hold",
                }
            )
        ]
    )
    planner = ConstrainedCreditHoldReActPlanner(generator)

    with pytest.raises(RuntimeError, match="bypass severe-hold HITL"):
        planner.decide(
            hold={"id": 3, "severity": "severe"},
            overdue_amount=12000,
            plan={"action": "payment_confirmation"},
            customer_evidence=None,
            payment_confirmed=12000,
        )


def test_constrained_react_blocks_unregistered_tool():
    generator = SequenceGenerator(
        [
            json.dumps(
                {
                    "decision": "release",
                    "rationale": "The customer supplied the required confirmation.",
                    "tool": "update_customer_balance",
                }
            )
        ]
    )
    planner = ConstrainedCreditHoldReActPlanner(generator)

    with pytest.raises(RuntimeError, match="outside the MCP allow-list"):
        planner.decide(
            hold={"id": 2, "severity": "minor"},
            overdue_amount=5000,
            plan={"action": "payment_confirmation"},
            customer_evidence=None,
            payment_confirmed=5000,
        )
