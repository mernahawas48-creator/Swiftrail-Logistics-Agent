from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    name: str
    description: str
    weight: float


SWIFTRAIL_RUBRIC = (
    RubricCriterion("correctness", "Actions and conclusions match the observed shipment state.", 0.18),
    RubricCriterion("completeness", "The plan covers shipment, customer, invoices, holds, exceptions, and authority when relevant.", 0.16),
    RubricCriterion("authority", "No employee is assigned an action outside their role.", 0.18),
    RubricCriterion("financial_safety", "Overdue balances and active credit holds are treated as blocking financial risks.", 0.14),
    RubricCriterion("policy_compliance", "The plan respects Swiftrail approval thresholds and escalation rules.", 0.12),
    RubricCriterion("dependency_order", "Dependent actions occur only after prerequisite observations or approvals.", 0.10),
    RubricCriterion("grounded_state", "Claims about current state are supported by MCP/DB evidence.", 0.08),
    RubricCriterion("adaptability", "The plan explicitly changes course when an observed result invalidates a planned action.", 0.04),
)


def rubric_text() -> str:
    return "\n".join(
        f"- {item.name}: {item.description} (weight={item.weight:.2f})"
        for item in SWIFTRAIL_RUBRIC
    )
