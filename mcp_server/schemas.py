from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,63}$"


class StrictInputModel(BaseModel):
    """Base model for every registered MCP tool input.

    ``extra='forbid'`` produces ``additionalProperties: false`` in the JSON
    Schema and prevents undeclared arguments from reaching a handler.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class SessionScopedInput(StrictInputModel):
    session_id: str = Field(
        ...,
        min_length=8,
        max_length=64,
        pattern=SESSION_ID_PATTERN,
        description=(
            "Client-generated session identifier used to isolate employee "
            "authorization and, in the Memory extension, conversation memory."
        ),
        examples=["demo-session-001"],
    )


class AuthenticateInput(SessionScopedInput):
    employee_id: int = Field(
        ...,
        gt=0,
        description="Positive employee identifier to authenticate for this session.",
        examples=[1],
    )


class SearchCustomerInput(SessionScopedInput):
    customer_id: int = Field(
        ...,
        gt=0,
        description="Positive unique identifier of the customer to retrieve.",
        examples=[3],
    )


class ShipmentStatusInput(SessionScopedInput):
    shipment_id: int = Field(
        ...,
        gt=0,
        description="Positive unique identifier of the shipment to retrieve.",
        examples=[500],
    )


class CustomerInvoicesInput(SessionScopedInput):
    customer_id: int = Field(
        ...,
        gt=0,
        description="Positive customer identifier whose invoices will be listed.",
        examples=[3],
    )


class CustomerCreditHoldsInput(SessionScopedInput):
    customer_id: int = Field(
        ...,
        gt=0,
        description="Positive customer identifier whose credit holds will be listed.",
        examples=[3],
    )


class ShipmentRateExceptionInput(SessionScopedInput):
    shipment_id: int = Field(
        ...,
        gt=0,
        description="Positive shipment identifier whose pending rate exception, if any, is returned.",
        examples=[500],
    )


class ApproveRateExceptionInput(SessionScopedInput):
    exception_id: int = Field(
        ...,
        gt=0,
        description="Positive identifier of a pending rate-exception request.",
        examples=[2],
    )


class ReleaseCreditHoldInput(SessionScopedInput):
    hold_id: int = Field(
        ...,
        gt=0,
        description="Positive identifier of an active credit hold.",
        examples=[2],
    )


class PortfolioExposureInput(SessionScopedInput):
    """Input for the finance-manager-only portfolio exposure tool."""


class PortfolioRiskSweepInput(SessionScopedInput):
    include_good_accounts: bool = Field(
        ...,
        description=(
            "Whether accounts with no active hold should still be included in "
            "the scored portfolio output."
        ),
    )


class RateExceptionDecision(StrictInputModel):
    approve: bool = Field(
        ...,
        description="True to approve the above-authority discount; false to reject it.",
    )
    reviewer_note: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Human finance-review rationale used in the returned audit result.",
    )

    @field_validator("reviewer_note")
    @classmethod
    def reject_placeholder_note(cls, value: str) -> str:
        if value.lower() in {"test note", "no reason", "n/a", "unknown"}:
            raise ValueError("reviewer_note must contain a meaningful rationale")
        return value


class CreditHoldReleaseDecision(StrictInputModel):
    confirm_release: bool = Field(
        ...,
        description="True only when the human explicitly authorizes release.",
    )
    authorization_note: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Human finance authorization rationale for the severe release.",
    )

    @field_validator("authorization_note")
    @classmethod
    def reject_placeholder_note(cls, value: str) -> str:
        if value.lower() in {"test note", "no reason", "n/a", "unknown"}:
            raise ValueError("authorization_note must contain a meaningful rationale")
        return value


EmployeeRole = Literal["sales_rep", "finance_manager"]
RateExceptionStatus = Literal["pending", "auto_approved", "approved", "rejected"]
CreditHoldStatus = Literal["active", "released"]
CreditHoldSeverity = Literal["minor", "severe"]
