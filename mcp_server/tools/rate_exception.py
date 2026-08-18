from __future__ import annotations

from mcp.server.fastmcp import Context

from app_instance import app
from db import SwiftrailDatabaseError, db_cursor
from schemas import ApproveRateExceptionInput, RateExceptionDecision
from tool_support import (
    authorize_session,
    database_failure,
    fail,
    ok,
    unexpected_failure,
    validate_request,
)


AUTO_APPROVAL_LIMIT = 15.0
HARD_DISCOUNT_LIMIT = 50.0


@app.tool()
async def approve_rate_exception(
    request: ApproveRateExceptionInput,
    ctx: Context,
) -> dict:
    """Resolve a pending rate exception safely.

    Discounts at or below 15% may be auto-approved by an authenticated sales
    representative or finance manager. Above-authority discounts require both
    a finance-manager session and a human elicitation decision.
    """

    validated, error = validate_request(ApproveRateExceptionInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            identity, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                """
                SELECT re.*, s.customer_id
                FROM rate_exceptions AS re
                JOIN shipments AS s ON s.id = re.shipment_id
                WHERE re.id = %s
                """,
                (validated.exception_id,),
            )
            exception = cursor.fetchone()

        if exception is None:
            return fail(
                "RATE_EXCEPTION_NOT_FOUND",
                f"Rate exception #{validated.exception_id} was not found.",
            )
        if exception["status"] != "pending":
            return fail(
                "RATE_EXCEPTION_ALREADY_RESOLVED",
                f"Rate exception #{validated.exception_id} is already '{exception['status']}'.",
                {"current_status": exception["status"]},
            )

        discount = float(exception["discount_pct"])
        if not 0 < discount <= HARD_DISCOUNT_LIMIT:
            return fail(
                "INVALID_DISCOUNT_STATE",
                "The stored discount is outside the allowed business range.",
            )

        if discount <= AUTO_APPROVAL_LIMIT:
            with db_cursor(transactional=True) as (_, cursor):
                identity, auth_error = authorize_session(
                    cursor,
                    session_id=validated.session_id,
                    allowed_roles={"sales_rep", "finance_manager"},
                )
                if auth_error:
                    return auth_error

                cursor.execute(
                    """
                    UPDATE rate_exceptions
                    SET status = 'auto_approved',
                        approved_by = %s,
                        resolved_at = NOW()
                    WHERE id = %s AND status = 'pending'
                    """,
                    (identity.employee_id, validated.exception_id),
                )
                if cursor.rowcount != 1:
                    return fail(
                        "CONCURRENT_OR_IDEMPOTENT_UPDATE",
                        "The rate exception changed before the update; no write was applied.",
                    )

            return ok(
                "RATE_EXCEPTION_AUTO_APPROVED",
                "The discount is within delegated authority and was auto-approved.",
                {
                    "exception_id": validated.exception_id,
                    "discount_pct": discount,
                    "approved_by": identity.employee_id,
                },
            )

        if identity.role != "finance_manager":
            return fail(
                "FINANCE_MANAGER_REQUIRED",
                "Above-authority discounts require a finance-manager session before elicitation.",
                {"discount_pct": discount, "current_role": identity.role},
            )

        if validated.decision is not None:
            decision = validated.decision
        else:
            result = await ctx.elicit(
                message=(
                    f"Rate exception {validated.exception_id} requests a {discount}% "
                    f"discount (justification: {exception['justification']}). This "
                    "exceeds the 15% delegated limit. Approve or reject?"
                ),
                schema=RateExceptionDecision,
            )
            if result.action != "accept" or result.data is None:
                return fail(
                    "HUMAN_DECISION_NOT_ACCEPTED",
                    "The above-authority rate exception was not finalized.",
                    {"elicitation_action": str(result.action)},
                )
            decision = result.data

        final_status = "approved" if decision.approve else "rejected"
        with db_cursor(transactional=True) as (_, cursor):
            identity, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                """
                UPDATE rate_exceptions
                SET status = %s,
                    approved_by = %s,
                    resolved_at = NOW()
                WHERE id = %s AND status = 'pending'
                """,
                (final_status, identity.employee_id, validated.exception_id),
            )
            if cursor.rowcount != 1:
                return fail(
                    "CONCURRENT_OR_IDEMPOTENT_UPDATE",
                    "The rate exception changed during human review; no duplicate write was applied.",
                )

        return ok(
            "RATE_EXCEPTION_RESOLVED",
            f"The rate exception was {final_status} by the finance manager.",
            {
                "exception_id": validated.exception_id,
                "discount_pct": discount,
                "status": final_status,
                "approved_by": identity.employee_id,
                "reviewer_note": decision.reviewer_note,
            },
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("rate-exception resolution")

        return ok(
            "RATE_EXCEPTION_RESOLVED",
            f"The rate exception was {final_status} by the finance manager.",
            {
                "exception_id": validated.exception_id,
                "discount_pct": discount,
                "status": final_status,
                "approved_by": identity.employee_id,
                "reviewer_note": result.data.reviewer_note,
            },
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("rate-exception resolution")
