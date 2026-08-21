from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from agent.client import SwiftrailAgent
from planning.orchestrator import (
    DecompositionMethod,
    SwiftrailPlanningOrchestrator,
)

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Swiftrail Planning Agent against the real MCP server."
    )

    parser.add_argument("--shipment-id", type=int, required=True)
    parser.add_argument("--customer-id", type=int, required=True)
    parser.add_argument("--employee-id", type=int, required=True)

    parser.add_argument(
        "--method",
        choices=[
            DecompositionMethod.DECOMPOSITION_FIRST.value,
            DecompositionMethod.DYNAMIC.value,
        ],
        default=DecompositionMethod.DECOMPOSITION_FIRST.value,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
    )

    parser.add_argument("--url", default=None)

    parser.add_argument(
        "--session-id",
        default="planning-session-001",
    )

    parser.add_argument("--model", default=None)

    return parser


async def run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing from the root .env file."
        )

    model_name = (
        args.model
        or os.getenv("MISTRAL_MODEL")
        or "mistral-small-latest"
    )

    llm = ChatMistralAI(
        model=model_name,
        api_key=api_key,
        temperature=0.1,
        max_retries=2,
    )

    agent = SwiftrailAgent(
        args.transport,
        args.url,
    )

    try:
        await agent.connect()

        auth_result = await agent.call_tool(
            "authenticate",
            {
                "request": {
                    "session_id": args.session_id,
                    "employee_id": args.employee_id,
                }
            },
        )

        auth = agent.decode_tool_result(auth_result)

        if not isinstance(auth, dict) or auth.get("success") is not True:
            raise RuntimeError(
                f"Authentication failed: {auth}"
            )

        print(
            "\nAuthenticated as: "
            f"{auth.get('data', {}).get('role', 'unknown')}"
        )

        orchestrator = SwiftrailPlanningOrchestrator(
            agent=agent,
            session_id=args.session_id,
            llm=llm,
            employee_id=args.employee_id,
        )

        outcome = await orchestrator.run(
            shipment_id=args.shipment_id,
            customer_id=args.customer_id,
            method=DecompositionMethod(args.method),
        )

        print("\n" + "=" * 68)
        print("PLANNING RESULT")
        print("=" * 68)
        print(outcome.result)

        if outcome.action_results:
            print("\n" + "=" * 68)
            print("SAFE ACTION EXECUTION")
            print("=" * 68)

            for action_result in outcome.action_results:
                status = (
                    "SUCCESS"
                    if action_result.success
                    else "FAILED"
                )

                print(
                    f"{action_result.action}: "
                    f"{status} - {action_result.message}"
                )

        print(f"\nArtifact: {outcome.artifact_path}")

    finally:
        await agent.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )

    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
