from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .swiftrail_validator import parse_actions

if TYPE_CHECKING:
    from agent.client import SwiftrailAgent


@dataclass
class ActionExecutionResult:
    action: str
    success: bool
    message: str
    result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None


async def execute_safe_actions(
    *,
    plan_text: str,
    agent: SwiftrailAgent,
    session_id: str,
    shipment_id: int,
    customer_id: int,
) -> list[ActionExecutionResult]:
    results: list[ActionExecutionResult] = []

    for action, args in parse_actions(plan_text):
        if action == "release_credit_hold":
            hold_id = _positive_int(args.get("hold_id"))
            if hold_id is None:
                results.append(
                    ActionExecutionResult(
                        action=action,
                        success=False,
                        message="release_credit_hold requires a valid hold_id.",
                    )
                )
                break

            payload = await _call(
                agent,
                "release_credit_hold",
                {
                    "session_id": session_id,
                    "hold_id": hold_id,
                },
            )

            if payload.get("success") is not True:
                results.append(
                    ActionExecutionResult(
                        action=action,
                        success=False,
                        message=payload.get(
                            "message",
                            "Credit-hold release failed.",
                        ),
                        result=payload,
                    )
                )
                break

            verification = await _call(
                agent,
                "list_customer_credit_holds",
                {
                    "session_id": session_id,
                    "customer_id": customer_id,
                },
            )

            verified = _hold_released(
                verification,
                hold_id,
            )

            results.append(
                ActionExecutionResult(
                    action=action,
                    success=verified,
                    message=(
                        "Credit-hold release verified."
                        if verified
                        else "Credit-hold write succeeded but verification failed."
                    ),
                    result=payload,
                    verification=verification,
                )
            )

        elif action == "approve_rate_exception":
            exception_id = _positive_int(args.get("exception_id"))
            if exception_id is None:
                results.append(
                    ActionExecutionResult(
                        action=action,
                        success=False,
                        message=(
                            "approve_rate_exception requires "
                            "a valid exception_id."
                        ),
                    )
                )
                break

            payload = await _call(
                agent,
                "approve_rate_exception",
                {
                    "session_id": session_id,
                    "exception_id": exception_id,
                },
            )

            if payload.get("success") is not True:
                results.append(
                    ActionExecutionResult(
                        action=action,
                        success=False,
                        message=payload.get(
                            "message",
                            "Rate-exception resolution failed.",
                        ),
                        result=payload,
                    )
                )
                break

            verification = await _call(
                agent,
                "get_shipment_rate_exception",
                {
                    "session_id": session_id,
                    "shipment_id": shipment_id,
                },
            )

            verified = _rate_exception_resolved(
                verification,
                exception_id,
            )

            results.append(
                ActionExecutionResult(
                    action=action,
                    success=verified,
                    message=(
                        "Rate-exception resolution verified."
                        if verified
                        else (
                            "Rate-exception write succeeded "
                            "but verification failed."
                        )
                    ),
                    result=payload,
                    verification=verification,
                )
            )

    return results


async def _call(
    agent: SwiftrailAgent,
    tool_name: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    raw = await agent.call_tool(
        tool_name,
        {"request": request},
    )
    payload = agent.decode_tool_result(raw)

    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004 - malformed external MCP response
            f"{tool_name} returned a malformed response."
        )

    return payload


def _positive_int(value: str | None) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None

    return number if number > 0 else None


def _hold_released(
    payload: dict[str, Any],
    hold_id: int,
) -> bool:
    if payload.get("success") is not True:
        return False

    holds = payload.get("data", {}).get("holds", [])

    return any(
        hold.get("id") == hold_id
        and hold.get("status") == "released"
        for hold in holds
    )


def _rate_exception_resolved(
    payload: dict[str, Any],
    exception_id: int,
) -> bool:
    if payload.get("success") is not True:
        return False

    exception = payload.get("data", {}).get("rate_exception")

    if not isinstance(exception, dict):
        return False

    return (
        exception.get("id") == exception_id
        and exception.get("status") != "pending"
    )
