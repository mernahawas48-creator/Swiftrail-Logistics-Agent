from __future__ import annotations

from planning.environment import Environment
from planning_eval.test_cases import (
    above_authority_rate_case,
    severe_hold_sales_rep_case,
)


def test_grounded_catches_sales_rep_severe_hold_release():
    _, employee_id, snapshot = severe_hold_sales_rep_case()
    env = Environment(shipment_id=3, employee_id=employee_id, snapshot_provider=lambda: snapshot)
    bad = """ACTION: check_shipment
ACTION: check_customer
ACTION: check_invoices
ACTION: check_credit_hold
ACTION: release_credit_hold hold_id=2
ACTION: release_shipment"""
    feedback = env.evaluate(bad)
    assert not feedback.success
    assert any("Sales representatives" in detail for detail in feedback.details)


def test_grounded_accepts_escalation_plan():
    _, employee_id, snapshot = severe_hold_sales_rep_case()
    env = Environment(shipment_id=3, employee_id=employee_id, snapshot_provider=lambda: snapshot)
    good = """ACTION: check_shipment
ACTION: check_customer
ACTION: check_invoices
ACTION: check_credit_hold
ACTION: check_rate_exception
ACTION: escalate role=finance_manager"""
    feedback = env.evaluate(good)
    assert feedback.success
    assert feedback.score == 1.0


def test_grounded_catches_above_authority_rate_approval():
    _, employee_id, snapshot = above_authority_rate_case()
    env = Environment(shipment_id=5, employee_id=employee_id, snapshot_provider=lambda: snapshot)
    bad = """ACTION: approve_rate_exception exception_id=2"""
    feedback = env.evaluate(bad)
    assert not feedback.success
    assert any("Above-15%" in detail for detail in feedback.details)
