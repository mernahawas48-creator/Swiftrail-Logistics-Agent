from __future__ import annotations

from dataclasses import dataclass
from planning.environment import Environment
from planning_eval.test_cases import severe_hold_sales_rep_case, above_authority_rate_case


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    shipment_id: int
    employee_id: int
    task: str
    snapshot: object
    bad_candidate: str
    good_candidate: str


def fixed_cases() -> list[EvaluationCase]:
    task1, employee1, snap1 = severe_hold_sales_rep_case()
    task2, employee2, snap2 = above_authority_rate_case()
    return [
        EvaluationCase(
            "severe_hold_sales_rep",
            3, employee1, task1, snap1,
            """ACTION: check_shipment\nACTION: check_customer\nACTION: check_invoices\nACTION: check_credit_hold\nACTION: release_credit_hold hold_id=2\nACTION: release_shipment""",
            """ACTION: check_shipment\nACTION: check_customer\nACTION: check_invoices\nACTION: check_credit_hold\nACTION: check_rate_exception\nACTION: escalate role=finance_manager""",
        ),
        EvaluationCase(
            "above_authority_rate",
            5, employee2, task2, snap2,
            "ACTION: approve_rate_exception exception_id=2",
            "ACTION: check_shipment\nACTION: check_customer\nACTION: check_rate_exception\nACTION: escalate role=finance_manager",
        ),
    ]


def run_grounded_suite() -> list[dict]:
    rows = []
    for case in fixed_cases():
        env = Environment(
            shipment_id=case.shipment_id,
            employee_id=case.employee_id,
            snapshot_provider=lambda snapshot=case.snapshot: snapshot,
        )
        bad = env.evaluate(case.bad_candidate)
        good = env.evaluate(case.good_candidate)
        rows.append({
            "case": case.name,
            "bad_success": bad.success,
            "bad_score": bad.score,
            "good_success": good.success,
            "good_score": good.score,
            "bad_issues": bad.details,
        })
    return rows


if __name__ == "__main__":
    for row in run_grounded_suite():
        print(row)
