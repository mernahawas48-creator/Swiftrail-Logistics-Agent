from __future__ import annotations

from typing import Annotated

from app_instance import app
from pydantic import Field


@app.resource("policy://credit-and-discount-authority")
def credit_and_discount_authority_policy() -> str:
    """Return the current internal credit and discount authority policy."""

    return (
        "SWIFTRAIL LOGISTICS -- CREDIT HOLD & DISCOUNT AUTHORITY POLICY\n"
        "(internal reference, v1.2)\n\n"
        "1. CREDIT HOLDS\n"
        "   - MINOR severity: invoice 30-89 days overdue. An authenticated "
        "sales_rep or finance_manager may release these directly.\n"
        "   - SEVERE severity: invoice 90+ days overdue, OR overdue balance "
        "exceeds 25% of the customer's credit limit. Release requires explicit "
        "human confirmation and a finance_manager session.\n\n"
        "2. RATE EXCEPTIONS (DISCOUNTS)\n"
        "   - Up to 15%: within delegated authority and may be auto-approved.\n"
        "   - Above 15% up to the 50% hard ceiling: requires explicit human "
        "confirmation and a finance_manager session.\n"
    )


@app.prompt()
def draft_rate_exception_justification(
    shipment_id: Annotated[int, Field(gt=0, description="Positive shipment ID")],
    discount_pct: Annotated[
        float,
        Field(gt=15, le=50, description="Requested above-authority discount"),
    ],
    reason_summary: Annotated[
        str,
        Field(min_length=10, max_length=500, description="Commercial context"),
    ],
) -> str:
    """Draft a concise rate-exception justification template."""

    return (
        "Write a concise, specific justification of at least 20 characters, "
        f"without unsupported claims, for requesting a {discount_pct}% discount "
        f"on shipment {shipment_id}. Requester context: {reason_summary}."
    )
