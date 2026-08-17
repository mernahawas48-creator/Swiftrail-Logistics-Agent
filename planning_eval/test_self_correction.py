from __future__ import annotations

from dataclasses import dataclass

from planning.algorithms.reflexion import reflexion
from planning.algorithms.self_refine import reflect_and_refine
from planning_eval.test_cases import severe_hold_sales_rep_case
from planning.environment import Environment

@dataclass
class Response:
    content: str


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []

    def invoke(self, messages, temperature=0.2):
        self.calls += 1
        self.messages.append(messages)
        return Response(self.responses.pop(0))


def test_self_refine_revises_after_grounded_failure():
    _, employee_id, snapshot = severe_hold_sales_rep_case()
    env = Environment(shipment_id=3, employee_id=employee_id, snapshot_provider=lambda: snapshot)
    draft = """Shipment resolution plan\nACTION: check_shipment\nACTION: check_customer\nACTION: check_invoices\nACTION: check_credit_hold\nACTION: release_credit_hold hold_id=2\nACTION: release_shipment"""
    revised = """Shipment resolution plan\nACTION: check_shipment\nACTION: check_customer\nACTION: check_invoices\nACTION: check_credit_hold\nACTION: check_rate_exception\nACTION: escalate role=finance_manager"""
    llm = FakeLLM(["The plan attempts an unauthorized severe credit-hold release.", revised])
    result = reflect_and_refine(
        "Resolve blocked shipment 3 safely for employee 1",
        draft,
        llm,
        critic_llm=llm,
        environment=env,
    )
    assert result.revision_feedback is not None
    assert result.revision_feedback.success
    assert "Sales representatives" in " ".join(result.grounded_issues)


def test_reflexion_carries_reflection_across_trials():
    _, employee_id, snapshot = severe_hold_sales_rep_case()
    env = Environment(shipment_id=3, employee_id=employee_id, snapshot_provider=lambda: snapshot)
    bad = """ACTION: check_shipment\nACTION: check_customer\nACTION: check_invoices\nACTION: check_credit_hold\nACTION: release_credit_hold hold_id=2\nACTION: release_shipment"""
    good = """ACTION: check_shipment\nACTION: check_customer\nACTION: check_invoices\nACTION: check_credit_hold\nACTION: check_rate_exception\nACTION: escalate role=finance_manager"""
    llm = FakeLLM([bad, "I must verify employee authority before releasing a severe hold.", good])
    result = reflexion("Resolve blocked shipment 3 safely for employee 1", llm, env, max_trials=2, memory_size=1, critic_llm=llm)
    assert result.success
    assert len(result.trials) == 2
    assert len(result.memory) == 1
    assert result.memory[0].startswith("I must verify")
