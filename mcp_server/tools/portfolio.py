from __future__ import annotations

from app_instance import app
from mcp.server.fastmcp import Context
from mcp.types import SamplingMessage
from mcp.types import TextContent as SamplingTextContent
from schemas import PortfolioExposureInput, PortfolioRiskSweepInput
from tool_support import (
    authorize_session,
    database_failure,
    ok,
    unexpected_failure,
    validate_request,
)

from db import SwiftrailDatabaseError, db_cursor


def list_portfolio_credit_exposure(request: PortfolioExposureInput) -> dict:
    """List portfolio-wide credit exposure for a finance manager only.

    Dynamic tool visibility is only a convenience. This handler always checks
    the authenticated session role again before returning cross-customer data.
    """

    validated, error = validate_request(PortfolioExposureInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                """
                SELECT ch.*, c.name AS customer_name
                FROM credit_holds AS ch
                JOIN customers AS c ON c.id = ch.customer_id
                WHERE ch.status = 'active'
                ORDER BY ch.severity DESC, ch.placed_at ASC
                """
            )
            active_holds = cursor.fetchall()

            cursor.execute(
                """
                SELECT re.*, s.customer_id
                FROM rate_exceptions AS re
                JOIN shipments AS s ON s.id = re.shipment_id
                WHERE re.status = 'pending' AND re.discount_pct > 15
                ORDER BY re.created_at ASC
                """
            )
            pending_discounts = cursor.fetchall()

            cursor.execute(
                """
                SELECT id, name, credit_status, credit_limit, balance_due
                FROM customers
                WHERE credit_status = 'hold'
                ORDER BY balance_due DESC
                """
            )
            customers_on_hold = cursor.fetchall()

        return ok(
            "PORTFOLIO_EXPOSURE_RETRIEVED",
            "Portfolio credit exposure retrieved successfully.",
            {
                "active_credit_holds": active_holds,
                "pending_above_authority_discounts": pending_discounts,
                "customers_on_hold": customers_on_hold,
            },
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("portfolio exposure lookup")


@app.tool()
async def run_portfolio_risk_sweep(
    request: PortfolioRiskSweepInput,
    ctx: Context,
) -> dict:
    """Score portfolio credit risk, report progress, and request sampling safely."""

    validated, error = validate_request(PortfolioRiskSweepInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute("SELECT * FROM customers ORDER BY id")
            customers = cursor.fetchall()
            scored = []
            total = len(customers)

            for index, customer in enumerate(customers, start=1):
                cursor.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM credit_holds
                    WHERE customer_id = %s AND status = 'active'
                    """,
                    (customer["id"],),
                )
                active_hold_count = cursor.fetchone()["n"]
                credit_limit = float(customer["credit_limit"])
                balance_ratio = (
                    float(customer["balance_due"]) / credit_limit
                    if credit_limit > 0
                    else 0.0
                )
                if validated.include_good_accounts or active_hold_count > 0:
                    scored.append(
                        {
                            "customer_id": customer["id"],
                            "name": customer["name"],
                            "balance_ratio": round(balance_ratio, 3),
                            "active_holds": active_hold_count,
                            "credit_status": customer["credit_status"],
                        }
                    )
                await ctx.report_progress(
                    progress=index,
                    total=total,
                    message=f"Scored {customer['name']} ({index}/{total})",
                )

        scores_text = "\n".join(
            f"- {item['name']}: balance is {item['balance_ratio'] * 100:.0f}% "
            f"of credit limit, {item['active_holds']} active hold(s)"
            for item in scored
        )

        narrative = (
            f"Scanned {len(scored)} account(s). Review accounts with active holds "
            "or high balance-to-limit ratios first."
        )
        sampling_used = False

        client_params = getattr(ctx.session, "client_params", None)
        client_capabilities = getattr(client_params, "capabilities", None)
        sampling_capability = getattr(client_capabilities, "sampling", None)
        if sampling_capability is not None:
            try:
                sampling_result = await ctx.session.create_message(
                    messages=[
                        SamplingMessage(
                            role="user",
                            content=SamplingTextContent(
                                type="text",
                                text=(
                                    "Write a 2-3 sentence portfolio risk summary for a "
                                    "finance manager using only this data:\n" + scores_text
                                ),
                            ),
                        )
                    ],
                    max_tokens=200,
                )
                narrative = getattr(
                    sampling_result.content,
                    "text",
                    str(sampling_result.content),
                )
                sampling_used = True
            except Exception:
                # Sampling is optional. A failed model callback must not erase the
                # deterministic risk scores already computed.
                narrative = (
                    "Sampling was unavailable; deterministic risk scores were "
                    "returned without a generated narrative."
                )

        return ok(
            "PORTFOLIO_RISK_SWEEP_COMPLETED",
            "Portfolio risk sweep completed successfully.",
            {
                "scanned": len(scored),
                "scores": scored,
                "narrative_summary": narrative,
                "sampling_used": sampling_used,
            },
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("portfolio risk sweep")
