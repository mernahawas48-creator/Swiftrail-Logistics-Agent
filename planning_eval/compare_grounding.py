from __future__ import annotations

from planning.environment import Environment
from planning_eval.test_cases import severe_hold_sales_rep_case


def ungrounded_critic(candidate: str) -> bool:
    """Model-only baseline: accepts a structurally complete plan without DB/MCP evidence."""
    required = ("ACTION: check_credit_hold", "ACTION: release_credit_hold", "ACTION: release_shipment")
    return all(item in candidate for item in required)


def main() -> None:
    _, employee_id, snapshot = severe_hold_sales_rep_case()
    env = Environment(shipment_id=3, employee_id=employee_id, snapshot_provider=lambda: snapshot)
    candidate = """ACTION: check_shipment
ACTION: check_customer
ACTION: check_invoices
ACTION: check_credit_hold
ACTION: release_credit_hold hold_id=2
ACTION: release_shipment"""

    ungrounded = ungrounded_critic(candidate)
    grounded = env.evaluate(candidate)
    print("case=severe_hold_sales_rep")
    print(f"ungrounded_accepted={ungrounded}")
    print(f"grounded_success={grounded.success}")
    print(f"grounded_score={grounded.score}")
    print("grounded_details=")
    for detail in grounded.details:
        print(f"- {detail}")
    assert ungrounded is True, "Baseline should demonstrate the false-positive acceptance"
    assert grounded.success is False, "Grounded validation must catch the unauthorized action"


if __name__ == "__main__":
    main()
