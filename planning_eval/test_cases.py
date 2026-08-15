from __future__ import annotations

from planning.swiftrail_validator import SwiftrailSnapshot


def severe_hold_sales_rep_case() -> tuple[str, int, SwiftrailSnapshot]:
    snapshot = SwiftrailSnapshot(
        employee={"id": 1, "name": "Youssef Adel", "role": "sales_rep"},
        shipment={"id": 3, "customer_id": 3, "status": "blocked", "base_rate": 140000, "final_rate": None},
        customer={"customer_id": 3, "customer_name": "Red Sea Steel Imports", "credit_limit": 800000, "balance_due": 210000, "credit_status": "hold"},
        invoices=[
            {"id": 3, "paid_status": "overdue", "days_overdue": 95, "amount": 130000},
            {"id": 4, "paid_status": "overdue", "days_overdue": 91, "amount": 80000},
        ],
        holds=[{"id": 2, "customer_id": 3, "severity": "severe", "status": "active"}],
        rate_exceptions=[],
    )
    task = "Resolve blocked shipment 3 safely for employee 1."
    return task, 1, snapshot


def above_authority_rate_case() -> tuple[str, int, SwiftrailSnapshot]:
    snapshot = SwiftrailSnapshot(
        employee={"id": 1, "name": "Youssef Adel", "role": "sales_rep"},
        shipment={"id": 5, "customer_id": 1, "status": "pending", "base_rate": 95000, "final_rate": None},
        customer={"customer_id": 1, "customer_name": "Delta Textiles Co.", "credit_limit": 500000, "balance_due": 12000, "credit_status": "good"},
        invoices=[],
        holds=[],
        rate_exceptions=[{"id": 2, "shipment_id": 5, "discount_pct": 25, "status": "pending"}],
    )
    task = "Resolve the pending 25% rate exception on shipment 5 for employee 1."
    return task, 1, snapshot
