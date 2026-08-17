import asyncio

from planning.action_executor import execute_safe_actions


class FakeAgent:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_name, args):
        self.calls.append((tool_name, args))

        if tool_name == "release_credit_hold":
            return {
                "success": True,
                "code": "CREDIT_HOLD_RELEASED",
                "message": "released",
                "data": {"hold_id": 7},
            }

        if tool_name == "list_customer_credit_holds":
            return {
                "success": True,
                "code": "CREDIT_HOLDS_RETRIEVED",
                "message": "retrieved",
                "data": {
                    "holds": [
                        {
                            "id": 7,
                            "status": "released",
                        }
                    ]
                },
            }

        if tool_name == "approve_rate_exception":
            return {
                "success": True,
                "code": "RATE_EXCEPTION_AUTO_APPROVED",
                "message": "approved",
                "data": {"exception_id": 4},
            }

        if tool_name == "get_shipment_rate_exception":
            return {
                "success": True,
                "code": "RATE_EXCEPTION_LOOKUP_COMPLETE",
                "message": "retrieved",
                "data": {
                    "rate_exception": {
                        "id": 4,
                        "status": "auto_approved",
                    }
                },
            }

        raise AssertionError(f"Unexpected tool call: {tool_name}")

    def decode_tool_result(self, result):
        return result


def test_release_credit_hold_is_verified():
    agent = FakeAgent()

    results = asyncio.run(
        execute_safe_actions(
            plan_text="ACTION: release_credit_hold hold_id=7",
            agent=agent,
            session_id="session-1",
            shipment_id=3,
            customer_id=3,
        )
    )

    assert len(results) == 1
    assert results[0].success is True

    assert agent.calls == [
        (
            "release_credit_hold",
            {
                "request": {
                    "session_id": "session-1",
                    "hold_id": 7,
                }
            },
        ),
        (
            "list_customer_credit_holds",
            {
                "request": {
                    "session_id": "session-1",
                    "customer_id": 3,
                }
            },
        ),
    ]


def test_rate_exception_is_verified():
    agent = FakeAgent()

    results = asyncio.run(
        execute_safe_actions(
            plan_text="ACTION: approve_rate_exception exception_id=4",
            agent=agent,
            session_id="session-1",
            shipment_id=3,
            customer_id=3,
        )
    )

    assert len(results) == 1
    assert results[0].success is True

    assert agent.calls == [
        (
            "approve_rate_exception",
            {
                "request": {
                    "session_id": "session-1",
                    "exception_id": 4,
                }
            },
        ),
        (
            "get_shipment_rate_exception",
            {
                "request": {
                    "session_id": "session-1",
                    "shipment_id": 3,
                }
            },
        ),
    ]
def test_rejected_write_stops_without_verification():
    class DeniedAgent(FakeAgent):
        async def call_tool(self, tool_name, args):
            self.calls.append((tool_name, args))

            if tool_name == "release_credit_hold":
                return {
                    "success": False,
                    "code": "INSUFFICIENT_AUTHORITY",
                    "message": "Finance manager approval is required.",
                    "data": {},
                }

            raise AssertionError(
                f"Unexpected tool call after rejection: {tool_name}"
            )

    agent = DeniedAgent()

    results = asyncio.run(
        execute_safe_actions(
            plan_text="ACTION: release_credit_hold hold_id=7",
            agent=agent,
            session_id="session-1",
            shipment_id=3,
            customer_id=3,
        )
    )

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].message == "Finance manager approval is required."

    assert agent.calls == [
        (
            "release_credit_hold",
            {
                "request": {
                    "session_id": "session-1",
                    "hold_id": 7,
                }
            },
        )
    ]

def test_rejected_write_stops_remaining_actions():
    class DeniedAgent(FakeAgent):
        async def call_tool(self, tool_name, args):
            self.calls.append((tool_name, args))

            if tool_name == "release_credit_hold":
                return {
                    "success": False,
                    "code": "FINANCE_MANAGER_REQUIRED",
                    "message": "Finance manager approval is required.",
                    "data": {},
                }

            raise AssertionError(
                f"Unexpected tool call after rejection: {tool_name}"
            )

    agent = DeniedAgent()

    results = asyncio.run(
        execute_safe_actions(
            plan_text=(
                "ACTION: release_credit_hold hold_id=7\n"
                "ACTION: approve_rate_exception exception_id=4"
            ),
            agent=agent,
            session_id="session-1",
            shipment_id=3,
            customer_id=3,
        )
    )

    assert len(results) == 1
    assert results[0].success is False

    assert agent.calls == [
        (
            "release_credit_hold",
            {
                "request": {
                    "session_id": "session-1",
                    "hold_id": 7,
                }
            },
        )
    ]