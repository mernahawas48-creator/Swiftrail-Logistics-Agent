from __future__ import annotations

from app_instance import app
from mcp.server.fastmcp import Context
from schemas import CreditHoldReleaseDecision, ReleaseCreditHoldInput
from tool_support import (
    authorize_session,
    database_failure,
    fail,
    ok,
    unexpected_failure,
    validate_request,
)

from db import SwiftrailDatabaseError, db_cursor


@app.tool()
async def release_credit_hold(
    request: ReleaseCreditHoldInput,
    ctx: Context,
) -> dict:
    """Release an active credit hold with role and human-risk controls."""

    validated, error = validate_request(ReleaseCreditHoldInput, request)
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
                "SELECT * FROM credit_holds WHERE id = %s",
                (validated.hold_id,),
            )
            hold = cursor.fetchone()

        if hold is None:
            return fail(
                "CREDIT_HOLD_NOT_FOUND",
                f"Credit hold #{validated.hold_id} was not found.",
            )
        if hold["status"] != "active":
            return fail(
                "CREDIT_HOLD_ALREADY_RELEASED",
                f"Credit hold #{validated.hold_id} is already '{hold['status']}'.",
                {"current_status": hold["status"]},
            )
        if hold["severity"] not in {"minor", "severe"}:
            return fail(
                "INVALID_HOLD_STATE",
                "The stored hold severity is outside the supported domain.",
            )

        authorization_note = "Routine minor-hold release within delegated authority."
        if hold["severity"] == "severe":
            if identity.role != "finance_manager":
                return fail(
                    "FINANCE_MANAGER_REQUIRED",
                    "A severe credit hold requires a finance-manager session before elicitation.",
                    {"current_role": identity.role},
                )

            decision = validated.decision
            elicitation_action = "pre_collected"
            if decision is None:
                result = await ctx.elicit(
                    message=(
                        f"Credit hold {validated.hold_id} on customer_id={hold['customer_id']} "
                        f"is SEVERE (reason: {hold['reason']}). Confirm or decline release."
                    ),
                    schema=CreditHoldReleaseDecision,
                )
                elicitation_action = str(result.action)
                decision = result.data if result.action == "accept" else None
            if decision is None or not decision.confirm_release:
                return fail(
                    "HUMAN_CONFIRMATION_REQUIRED",
                    "The severe credit hold was not released because confirmation was absent.",
                    {"elicitation_action": elicitation_action},
                )
            authorization_note = decision.authorization_note

        allowed_roles = (
            {"finance_manager"}
            if hold["severity"] == "severe"
            else {"sales_rep", "finance_manager"}
        )
        with db_cursor(transactional=True) as (_, cursor):
            identity, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles=allowed_roles,
            )
            if auth_error:
                return auth_error

            cursor.execute(
                """
                UPDATE credit_holds
                SET status = 'released',
                    released_by = %s,
                    released_at = NOW()
                WHERE id = %s AND status = 'active'
                """,
                (identity.employee_id, validated.hold_id),
            )
            if cursor.rowcount != 1:
                return fail(
                    "CONCURRENT_OR_IDEMPOTENT_UPDATE",
                    "The credit hold changed during the request; no duplicate write was applied.",
                )

            cursor.execute(
                """
                SELECT COUNT(*) AS active_count
                FROM credit_holds
                WHERE customer_id = %s AND status = 'active'
                """,
                (hold["customer_id"],),
            )
            remaining = cursor.fetchone()["active_count"]
            if remaining == 0:
                cursor.execute(
                    "UPDATE customers SET credit_status = 'good' WHERE id = %s",
                    (hold["customer_id"],),
                )

        return ok(
            "CREDIT_HOLD_RELEASED",
            "The credit hold was released successfully.",
            {
                "hold_id": validated.hold_id,
                "customer_id": hold["customer_id"],
                "severity": hold["severity"],
                "released_by": identity.employee_id,
                "remaining_active_holds": remaining,
                "authorization_note": authorization_note,
            },
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("credit-hold release")
