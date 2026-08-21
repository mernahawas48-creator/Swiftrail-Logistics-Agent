from __future__ import annotations

from app_instance import app
from schemas import (
    CustomerCreditHoldsInput,
    CustomerInvoicesInput,
    SearchCustomerInput,
    ShipmentRateExceptionInput,
    ShipmentStatusInput,
)
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
def search_customer(request: SearchCustomerInput) -> dict:
    """Retrieve one customer profile through a scoped, validated read query."""

    validated, error = validate_request(SearchCustomerInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                """
                SELECT id, name, credit_limit, balance_due, credit_status
                FROM customers
                WHERE id = %s
                """,
                (validated.customer_id,),
            )
            customer = cursor.fetchone()

        if customer is None:
            return fail(
                "CUSTOMER_NOT_FOUND",
                f"Customer #{validated.customer_id} was not found.",
            )
        return ok(
            "CUSTOMER_RETRIEVED",
            "Customer profile retrieved successfully.",
            {"customer": customer},
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("customer lookup")


@app.tool()
def get_shipment_status(request: ShipmentStatusInput) -> dict:
    """Retrieve the current status and financial details of one shipment."""

    validated, error = validate_request(ShipmentStatusInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                """
                SELECT
                    s.id,
                    s.customer_id,
                    c.name AS customer_name,
                    s.origin,
                    s.destination,
                    s.railcar_id,
                    s.base_rate,
                    s.final_rate,
                    s.status,
                    s.requested_by,
                    s.created_at
                FROM shipments AS s
                JOIN customers AS c ON c.id = s.customer_id
                WHERE s.id = %s
                """,
                (validated.shipment_id,),
            )
            shipment = cursor.fetchone()

        if shipment is None:
            return fail(
                "SHIPMENT_NOT_FOUND",
                f"Shipment #{validated.shipment_id} was not found.",
            )
        return ok(
            "SHIPMENT_RETRIEVED",
            "Shipment status retrieved successfully.",
            {"shipment": shipment},
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("shipment lookup")


@app.tool()
def list_customer_invoices(request: CustomerInvoicesInput) -> dict:
    """List invoices for one validated customer and authenticated session."""

    validated, error = validate_request(CustomerInvoicesInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                "SELECT id FROM customers WHERE id = %s",
                (validated.customer_id,),
            )
            if cursor.fetchone() is None:
                return fail(
                    "CUSTOMER_NOT_FOUND",
                    f"Customer #{validated.customer_id} was not found.",
                )

            cursor.execute(
                """
                SELECT id, customer_id, shipment_id, amount, due_date,
                       paid_status, days_overdue
                FROM invoices
                WHERE customer_id = %s
                ORDER BY due_date DESC, id DESC
                """,
                (validated.customer_id,),
            )
            invoices = cursor.fetchall()

        return ok(
            "INVOICES_RETRIEVED",
            f"Retrieved {len(invoices)} invoice(s).",
            {"customer_id": validated.customer_id, "invoices": invoices},
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("invoice lookup")


@app.tool()
def list_customer_credit_holds(request: CustomerCreditHoldsInput) -> dict:
    """List one customer's own credit holds (active and released).

    Added for the planning/decomposition agent: resolving a blocked shipment
    needs a scoped, single-customer view of hold severity and status before
    deciding whether release is even in scope for the current role. The
    existing ``list_portfolio_credit_exposure`` tool is finance_manager-only
    and cross-customer, which is the wrong shape for a sales_rep-initiated
    triage of one shipment, so this stays open to both roles like the other
    single-customer read tools above.
    """

    validated, error = validate_request(CustomerCreditHoldsInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                "SELECT id FROM customers WHERE id = %s",
                (validated.customer_id,),
            )
            if cursor.fetchone() is None:
                return fail(
                    "CUSTOMER_NOT_FOUND",
                    f"Customer #{validated.customer_id} was not found.",
                )

            cursor.execute(
                """
                SELECT id, customer_id, reason, severity, status,
                       placed_at, released_by, released_at
                FROM credit_holds
                WHERE customer_id = %s
                ORDER BY status = 'active' DESC, placed_at DESC
                """,
                (validated.customer_id,),
            )
            holds = cursor.fetchall()

        return ok(
            "CREDIT_HOLDS_RETRIEVED",
            f"Retrieved {len(holds)} credit hold record(s).",
            {
                "customer_id": validated.customer_id,
                "holds": holds,
                "active_holds": [h for h in holds if h["status"] == "active"],
            },
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("credit-hold lookup")


@app.tool()
def get_shipment_rate_exception(request: ShipmentRateExceptionInput) -> dict:
    """Return the most recent rate exception tied to one shipment, if any.

    Added for the same reason as ``list_customer_credit_holds`` above: the
    write tool ``approve_rate_exception`` requires an ``exception_id`` the
    caller must already know, but nothing previously let the agent discover
    that id starting only from a shipment id.
    """

    validated, error = validate_request(ShipmentRateExceptionInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            _, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error

            cursor.execute(
                "SELECT id FROM shipments WHERE id = %s",
                (validated.shipment_id,),
            )
            if cursor.fetchone() is None:
                return fail(
                    "SHIPMENT_NOT_FOUND",
                    f"Shipment #{validated.shipment_id} was not found.",
                )

            cursor.execute(
                """
                SELECT id, shipment_id, requested_by, discount_pct,
                       justification, status, approved_by, created_at, resolved_at
                FROM rate_exceptions
                WHERE shipment_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (validated.shipment_id,),
            )
            exception = cursor.fetchone()

        return ok(
            "RATE_EXCEPTION_LOOKUP_COMPLETE",
            "Rate exception retrieved." if exception else "No rate exception exists for this shipment.",
            {"shipment_id": validated.shipment_id, "rate_exception": exception},
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("rate-exception lookup")
