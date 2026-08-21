"""
Full run through every protocol concern, using the seed data described in
db/seed.sql / README.md. Run with:

    python demo.py --transport stdio        # against mcp_server/server.py
    python demo.py --transport http --url http://localhost:8000/mcp   # against server_http.py

Each step below is labeled with the concern it demonstrates so the output
doubles as the demo transcript (see demo/demo_transcript.md for a captured
run).
"""

from __future__ import annotations

import argparse
import asyncio
import json

from client import SwiftrailAgent

SESSION_ID = "demo-session-001"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show(agent: SwiftrailAgent, result) -> dict | str | None:
    decoded = agent.decode_tool_result(result)
    print(json.dumps(decoded, indent=2, default=str) if isinstance(decoded, dict) else decoded)
    return decoded


async def run(transport: str, url: str | None) -> None:
    agent = SwiftrailAgent(transport, url)
    try:
        section("1. Initialize and verify declared capabilities")
        await agent.connect()
        print(f"tools.listChanged = {agent.supports_tool_list_changes()}")

        section("2. Discover initial tools")
        tools = await agent.discover_tools()
        print([tool.name for tool in tools])

        section("3. Authenticate the demo session as sales_rep")
        result = await agent.call_tool(
            "authenticate",
            {"request": {"session_id": SESSION_ID, "employee_id": 1}},
        )
        show(agent, result)

        section("4. Read scoped customer and invoice data")
        result = await agent.call_tool(
            "search_customer",
            {"request": {"session_id": SESSION_ID, "customer_id": 3}},
        )
        show(agent, result)
        result = await agent.call_tool(
            "list_customer_invoices",
            {"request": {"session_id": SESSION_ID, "customer_id": 3}},
        )
        show(agent, result)

        section("5. Read policy resource and render prompt")
        policy = await agent.read_resource("policy://credit-and-discount-authority")
        if policy:
            print(policy.contents[0].text[:450] + "...")
        prompts = await agent.list_prompts()
        if prompts:
            rendered = await agent.get_prompt(
                "draft_rate_exception_justification",
                {
                    "shipment_id": "500",
                    "discount_pct": "25",
                    "reason_summary": "Customer committed to three additional shipments this quarter.",
                },
            )
            if rendered:
                print(rendered.messages[0].content.text)

        section("6. Above-authority discount fails safely for sales_rep")
        result = await agent.call_tool(
            "approve_rate_exception",
            {"request": {"session_id": SESSION_ID, "exception_id": 2}},
        )
        show(agent, result)

        section("7. Authenticate as finance_manager and receive tool-list change")
        result = await agent.call_tool(
            "authenticate",
            {"request": {"session_id": SESSION_ID, "employee_id": 3}},
        )
        show(agent, result)

        # Notification-capable clients use the dirty flag. Explicit re-listing is
        # also the safe fallback if a server/client implementation misses it.
        if agent.tool_list_dirty or not agent.supports_tool_list_changes():
            tools = await agent.discover_tools()
            print("Updated tools:", [tool.name for tool in tools])

        section("8. Resolve above-authority discount through elicitation")
        result = await agent.call_tool(
            "approve_rate_exception",
            {"request": {"session_id": SESSION_ID, "exception_id": 2}},
        )
        show(agent, result)

        section("9. View finance-only portfolio exposure")
        result = await agent.call_tool(
            "list_portfolio_credit_exposure",
            {"request": {"session_id": SESSION_ID}},
        )
        show(agent, result)

        section("10. Release severe hold through elicitation")
        result = await agent.call_tool(
            "release_credit_hold",
            {"request": {"session_id": SESSION_ID, "hold_id": 2}},
        )
        show(agent, result)

        section("11. Run portfolio sweep with progress and sampling fallback")

        async def progress(current, total, message):
            print(f"[{current}/{total}] {message}")

        result = await agent.call_tool(
            "run_portfolio_risk_sweep",
            {
                "request": {
                    "session_id": SESSION_ID,
                    "include_good_accounts": True,
                }
            },
            progress_callback=progress,
        )
        show(agent, result)

        section("Demo completed")
    finally:
        await agent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--url", default=None)
    args = parser.parse_args()
    asyncio.run(run(args.transport, args.url))
