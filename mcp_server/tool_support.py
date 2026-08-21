from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, TypeVar

import session as session_state
from pydantic import BaseModel, ValidationError

from db import SwiftrailDatabaseError

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


def ok(code: str, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": True,
        "code": code,
        "message": message,
        "data": data or {},
    }


def fail(
    code: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "code": code,
        "message": message,
        "data": data or {},
    }


def validate_request(
    model: type[ModelT], request: ModelT | dict[str, Any]
) -> tuple[ModelT | None, dict[str, Any] | None]:
    try:
        validated = request if isinstance(request, model) else model.model_validate(request)
        return validated, None
    except ValidationError as error:
        return None, fail(
            "INVALID_INPUT",
            "The tool request failed server-side validation.",
            {"errors": error.errors(include_url=False)},
        )


def authorize_session(
    cursor,
    *,
    session_id: str,
    allowed_roles: Iterable[str] | None = None,
):
    identity = session_state.get_session(session_id)
    if identity is None:
        return None, fail(
            "AUTHENTICATION_REQUIRED",
            "Authenticate this session before using the requested tool.",
        )

    cursor.execute(
        "SELECT id, name, role FROM employees WHERE id = %s",
        (identity.employee_id,),
    )
    employee = cursor.fetchone()
    if employee is None or employee["role"] != identity.role:
        session_state.clear_session(session_id)
        return None, fail(
            "SESSION_INVALID",
            "The authenticated employee session is no longer valid.",
        )

    allowed = set(allowed_roles or ())
    if allowed and employee["role"] not in allowed:
        return None, fail(
            "FORBIDDEN",
            "The authenticated role is not authorized for this operation.",
            {"required_roles": sorted(allowed), "current_role": employee["role"]},
        )

    return identity, None


def database_failure(error: SwiftrailDatabaseError) -> dict[str, Any]:
    logger.warning("Safe database failure returned by tool: %s", type(error).__name__)
    return fail("DATABASE_UNAVAILABLE", str(error))


def unexpected_failure(operation: str) -> dict[str, Any]:
    logger.exception("Unexpected failure during %s.", operation)
    return fail(
        "INTERNAL_ERROR",
        "The operation could not be completed safely. No unconfirmed write was retained.",
    )
