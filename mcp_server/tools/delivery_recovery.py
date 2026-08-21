from __future__ import annotations

from app_instance import app
from schemas import ApplyShipmentRerouteInput, CreateDeliveryRecoveryCaseInput
from tool_support import (
    authorize_session,
    database_failure,
    fail,
    ok,
    unexpected_failure,
    validate_request,
)

from db import SwiftrailDatabaseError, db_cursor

ADMIN_COST_LIMIT = 500.0


@app.tool()
def create_delivery_recovery_case(
    request: CreateDeliveryRecoveryCaseInput,
) -> dict:
    """Persist one real delivery-exception recovery case."""

    validated, error = validate_request(CreateDeliveryRecoveryCaseInput, request)
    if error:
        return error
    try:
        with db_cursor(transactional=True) as (_, cursor):
            identity, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles={"sales_rep", "finance_manager"},
            )
            if auth_error:
                return auth_error
            cursor.execute(
                "SELECT id, customer_id, status FROM shipments WHERE id = %s",
                (validated.shipment_id,),
            )
            shipment = cursor.fetchone()
            if shipment is None:
                return fail("SHIPMENT_NOT_FOUND", "Shipment was not found.")
            if shipment["status"] != "delivery_exception":
                return fail(
                    "DELIVERY_EXCEPTION_REQUIRED",
                    "Shipment is not currently in a delivery-exception state.",
                )
            cursor.execute(
                """
                SELECT * FROM delivery_recovery_cases
                WHERE shipment_id = %s
                  AND case_status IN ('open','waiting_customer','waiting_admin')
                ORDER BY id DESC LIMIT 1
                """,
                (validated.shipment_id,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                recovery_case = existing
            else:
                cursor.execute(
                    """
                    INSERT INTO delivery_recovery_cases(
                        shipment_id, customer_id, failure_reason,
                        case_status, created_by
                    ) VALUES (%s, %s, %s, 'waiting_customer', %s)
                    """,
                    (
                        validated.shipment_id,
                        shipment["customer_id"],
                        validated.failure_reason,
                        identity.employee_id,
                    ),
                )
                case_id = int(cursor.lastrowid)
                cursor.execute(
                    "SELECT * FROM delivery_recovery_cases WHERE id = %s",
                    (case_id,),
                )
                recovery_case = cursor.fetchone()
        return ok(
            "DELIVERY_RECOVERY_CASE_READY",
            "Delivery recovery case is ready for customer input.",
            {"recovery_case": recovery_case},
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("delivery recovery case creation")


@app.tool()
def apply_shipment_reroute(request: ApplyShipmentRerouteInput) -> dict:
    """Apply one policy-checked and idempotent reroute decision."""

    validated, error = validate_request(ApplyShipmentRerouteInput, request)
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
                SELECT drc.*, s.destination AS current_destination
                FROM delivery_recovery_cases AS drc
                JOIN shipments AS s ON s.id = drc.shipment_id
                WHERE drc.id = %s
                """,
                (validated.case_id,),
            )
            recovery_case = cursor.fetchone()
        if recovery_case is None:
            return fail("RECOVERY_CASE_NOT_FOUND", "Recovery case was not found.")
        if recovery_case["case_status"] == "resolved":
            if recovery_case["idempotency_key"] == validated.idempotency_key:
                return ok(
                    "REROUTE_ALREADY_APPLIED",
                    "The same reroute was already applied safely.",
                    {"recovery_case": recovery_case},
                )
            return fail(
                "RECOVERY_CASE_ALREADY_RESOLVED",
                "Recovery case was resolved by a different operation.",
            )

        requires_admin = (
            not validated.destination_verified
            or float(validated.estimated_cost) > ADMIN_COST_LIMIT
            or validated.customs_change
            or validated.high_value
        )
        if requires_admin:
            authorization = validated.authorization
            if authorization is None:
                return fail(
                    "HUMAN_AUTHORIZATION_REQUIRED",
                    "This reroute requires a persisted admin decision.",
                )
            if (
                identity.role != "finance_manager"
                or authorization.admin_employee_id != identity.employee_id
            ):
                return fail(
                    "FINANCE_MANAGER_REQUIRED",
                    "The approving admin must own the authenticated finance session.",
                )

        with db_cursor(transactional=True) as (_, cursor):
            identity, auth_error = authorize_session(
                cursor,
                session_id=validated.session_id,
                allowed_roles=(
                    {"finance_manager"}
                    if requires_admin
                    else {"sales_rep", "finance_manager"}
                ),
            )
            if auth_error:
                return auth_error
            cursor.execute(
                """
                UPDATE delivery_recovery_cases
                SET case_status='resolved', selected_option='reroute',
                    requested_destination=%s, estimated_cost=%s,
                    requires_admin=%s, idempotency_key=%s,
                    applied_by=%s, resolved_at=NOW()
                WHERE id=%s AND case_status != 'resolved'
                """,
                (
                    validated.new_destination,
                    validated.estimated_cost,
                    requires_admin,
                    validated.idempotency_key,
                    identity.employee_id,
                    validated.case_id,
                ),
            )
            if cursor.rowcount != 1:
                return fail(
                    "CONCURRENT_OR_IDEMPOTENT_UPDATE",
                    "Recovery case changed before the reroute was applied.",
                )
            cursor.execute(
                """
                UPDATE shipments SET destination=%s, status='pending'
                WHERE id=%s AND status='delivery_exception'
                """,
                (validated.new_destination, recovery_case["shipment_id"]),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Shipment changed before reroute application.")
        return ok(
            "SHIPMENT_REROUTED",
            "Shipment destination was updated successfully.",
            {
                "case_id": validated.case_id,
                "shipment_id": recovery_case["shipment_id"],
                "destination": validated.new_destination,
                "requires_admin": requires_admin,
                "applied_by": identity.employee_id,
            },
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("shipment reroute")
