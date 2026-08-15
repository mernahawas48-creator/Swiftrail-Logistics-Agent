from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .models import EnvironmentFeedback


@dataclass(slots=True)
class SwiftrailSnapshot:
    employee: dict[str, Any]
    shipment: dict[str, Any]
    customer: dict[str, Any]
    invoices: list[dict[str, Any]]
    holds: list[dict[str, Any]]
    rate_exceptions: list[dict[str, Any]]


_ACTION_RE = re.compile(r"^ACTION:[ \t]*(?P<action>[a-z_]+)(?:[ \t]+(?P<args>[^\r\n]+))?", re.I | re.M)


def parse_actions(candidate: str) -> list[tuple[str, dict[str, str]]]:
    actions: list[tuple[str, dict[str, str]]] = []
    for match in _ACTION_RE.finditer(candidate):
        args: dict[str, str] = {}
        for part in (match.group("args") or "").split():
            if "=" in part:
                key, value = part.split("=", 1)
                args[key.strip()] = value.strip().strip(",.")
        actions.append((match.group("action").lower(), args))
    return actions


class SwiftrailGroundedValidator:
    """Ground candidate plans in the real Swiftrail database.

    The database is the source of truth; this class does not use random scores.
    A callable snapshot_provider can be injected for deterministic tests.
    """

    def __init__(self, *, shipment_id: int, employee_id: int, snapshot_provider: Callable[[], SwiftrailSnapshot] | None = None) -> None:
        self.shipment_id = shipment_id
        self.employee_id = employee_id
        self.snapshot_provider = snapshot_provider

    def _load_snapshot(self) -> SwiftrailSnapshot:
        if self.snapshot_provider is not None:
            return self.snapshot_provider()
        from mcp_server.db import db_cursor
        with db_cursor() as (_, cursor):
            cursor.execute("SELECT id, name, role FROM employees WHERE id=%s", (self.employee_id,))
            employee = cursor.fetchone()
            cursor.execute(
                "SELECT s.*, c.name AS customer_name, c.credit_limit, c.balance_due, c.credit_status "
                "FROM shipments s JOIN customers c ON c.id=s.customer_id WHERE s.id=%s",
                (self.shipment_id,),
            )
            shipment = cursor.fetchone()
            if shipment is None:
                raise ValueError(f"Shipment #{self.shipment_id} not found")
            customer_id = shipment["customer_id"]
            cursor.execute("SELECT * FROM invoices WHERE customer_id=%s ORDER BY id", (customer_id,))
            invoices = list(cursor.fetchall())
            cursor.execute("SELECT * FROM credit_holds WHERE customer_id=%s AND status='active' ORDER BY id", (customer_id,))
            holds = list(cursor.fetchall())
            cursor.execute("SELECT re.* FROM rate_exceptions re WHERE re.shipment_id=%s ORDER BY id", (self.shipment_id,))
            rate_exceptions = list(cursor.fetchall())
        if employee is None:
            raise ValueError(f"Employee #{self.employee_id} not found")
        customer = {k: shipment[k] for k in ("customer_id", "customer_name", "credit_limit", "balance_due", "credit_status")}
        return SwiftrailSnapshot(employee, shipment, customer, invoices, holds, rate_exceptions)

    def evaluate(self, candidate: str) -> EnvironmentFeedback:
        state = self._load_snapshot()
        actions = parse_actions(candidate)
        names = [name for name, _ in actions]
        details: list[str] = []
        evidence: dict[str, Any] = {
            "employee": state.employee,
            "shipment": {k: state.shipment.get(k) for k in ("id", "customer_id", "status", "base_rate", "final_rate")},
            "customer": state.customer,
            "active_holds": state.holds,
            "invoices": state.invoices,
            "rate_exceptions": state.rate_exceptions,
            "actions": actions,
        }

        if not actions:
            details.append("No explicit ACTION lines were found; the plan cannot be validated safely.")

        severe_holds = [h for h in state.holds if h.get("severity") == "severe"]
        overdue = [i for i in state.invoices if i.get("paid_status") == "overdue"]
        pending_exceptions = [r for r in state.rate_exceptions if r.get("status") == "pending"]

        if state.shipment.get("status") == "blocked":
            required_observations = {"check_shipment", "check_customer", "check_invoices", "check_credit_hold", "check_rate_exception"}
            missing = sorted(required_observations.difference(names))
            if missing:
                details.append(f"Blocked shipment plan is missing required observations: {', '.join(missing)}.")

        role = state.employee["role"]
        if severe_holds:
            if "release_credit_hold" in names and role != "finance_manager":
                details.append("Sales representatives cannot release a severe credit hold; escalation to finance_manager is required.")
            if "escalate" not in names:
                details.append("An active severe credit hold requires escalation to finance_manager before release.")
            else:
                if not any(args.get("role") == "finance_manager" for action, args in actions if action == "escalate"):
                    details.append("The escalation action must explicitly target role=finance_manager.")

        for exception in pending_exceptions:
            discount = float(exception["discount_pct"])
            if discount > 15 and "approve_rate_exception" in names and role != "finance_manager":
                details.append("Above-15% rate exceptions require a finance_manager; the current sales_rep cannot approve them.")
            if discount > 15 and role == "sales_rep" and "escalate" not in names:
                details.append("The pending above-authority rate exception requires finance_manager escalation.")

        if overdue and state.customer.get("credit_status") == "hold" and "release_shipment" in names:
            details.append("The shipment cannot be safely released while the customer remains on credit hold.")

        if "release_credit_hold" in names and not severe_holds:
            details.append("The plan requests a credit-hold release, but no active credit hold exists in the database.")

        if "approve_rate_exception" in names and not pending_exceptions:
            details.append("The plan requests rate-exception approval, but no pending exception exists for this shipment.")

        # Score is evidence-derived: start at 1 and subtract deterministic violations.
        score = max(0.0, 1.0 - min(1.0, 0.15 * len(details)))
        success = not details
        return EnvironmentFeedback(success=success, score=round(score, 4), details=details, evidence=evidence)
