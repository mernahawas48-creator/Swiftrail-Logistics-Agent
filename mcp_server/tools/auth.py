from __future__ import annotations

from threading import RLock

import session as session_state
from app_instance import app
from mcp.server.fastmcp import Context
from runtime_sessions import track_session
from schemas import AuthenticateInput
from tool_support import (
    database_failure,
    fail,
    ok,
    unexpected_failure,
    validate_request,
)

from db import SwiftrailDatabaseError, db_cursor
from tools.portfolio import list_portfolio_credit_exposure

_tool_state_lock = RLock()
_portfolio_tool_exposed = False


async def _notify_tool_list_changed(ctx: Context) -> None:
    """Send the real MCP tools/list_changed notification."""

    session = ctx.session
    sender = getattr(session, "send_tool_list_changed", None)
    if sender is None:
        raise RuntimeError("Installed MCP SDK cannot send tools/list_changed.")
    await sender()


@app.tool()
async def authenticate(request: AuthenticateInput, ctx: Context) -> dict:
    """Authenticate one explicit client session as an existing employee.

    The finance portfolio tool is dynamically exposed while at least one
    finance-manager session is active. Every sensitive handler still performs
    independent authorization; visibility alone is never trusted.
    """

    global _portfolio_tool_exposed

    validated, error = validate_request(AuthenticateInput, request)
    if error:
        return error

    try:
        with db_cursor() as (_, cursor):
            cursor.execute(
                "SELECT id, name, role FROM employees WHERE id = %s",
                (validated.employee_id,),
            )
            employee = cursor.fetchone()

        if employee is None:
            return fail(
                "EMPLOYEE_NOT_FOUND",
                f"Employee #{validated.employee_id} was not found.",
            )
        if employee["role"] not in {"sales_rep", "finance_manager"}:
            return fail(
                "UNSUPPORTED_ROLE",
                "The employee role is not supported by this MCP server.",
            )

        previous = session_state.set_session(
            session_id=validated.session_id,
            employee_id=employee["id"],
            role=employee["role"],
            employee_name=employee["name"],
        )
        track_session(ctx.session)

        tool_set_changed = False
        with _tool_state_lock:
            finance_active = session_state.has_role("finance_manager")
            if finance_active and not _portfolio_tool_exposed:
                app.add_tool(
                    list_portfolio_credit_exposure,
                    name="list_portfolio_credit_exposure",
                )
                _portfolio_tool_exposed = True
                tool_set_changed = True
            elif not finance_active and _portfolio_tool_exposed:
                app.remove_tool("list_portfolio_credit_exposure")
                _portfolio_tool_exposed = False
                tool_set_changed = True

        if tool_set_changed:
            await _notify_tool_list_changed(ctx)

        return ok(
            "AUTHENTICATED",
            "Session authentication updated successfully.",
            {
                "session_id": validated.session_id,
                "employee_id": employee["id"],
                "name": employee["name"],
                "role": employee["role"],
                "previous_role": previous.role if previous else None,
                "tool_set_changed": tool_set_changed,
                "portfolio_tool_exposed": _portfolio_tool_exposed,
            },
        )
    except SwiftrailDatabaseError as db_error:
        return database_failure(db_error)
    except Exception:
        return unexpected_failure("session authentication")
