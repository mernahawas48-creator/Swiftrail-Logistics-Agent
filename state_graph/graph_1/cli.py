from __future__ import annotations

import argparse
import json
from typing import Any

from state_graph.core.types import TicketStatus
from state_graph.graph_1.graph import GRAPH_NAME
from state_graph.graph_1.live import build_live_service


def _print(value: Any) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate the live Delivery Exception Recovery graph."
    )
    parser.add_argument(
        "--mcp-url", default="http://127.0.0.1:8000/mcp"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start")
    start.add_argument("--shipment-id", type=int, required=True)
    start.add_argument("--session-id", required=True)
    start.add_argument("--employee-id", type=int, required=True)
    start.add_argument("--failure-reason", required=True)

    status = commands.add_parser("status")
    status.add_argument("--run-id", required=True)

    customer = commands.add_parser("customer")
    customer.add_argument("--run-id", required=True)
    customer.add_argument(
        "--choice-json",
        required=True,
        help="JSON object containing action and the selected option fields.",
    )

    commands.add_parser("hitl-tasks")
    hitl = commands.add_parser("resolve-hitl")
    hitl.add_argument("--task-id", required=True)
    decision = hitl.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    hitl.add_argument("--note", required=True)
    hitl.add_argument("--admin-employee-id", type=int, required=True)

    tickets = commands.add_parser("tickets")
    tickets.add_argument(
        "--status", choices=[status.value for status in TicketStatus]
    )
    investigate = commands.add_parser("investigate-ticket")
    investigate.add_argument("--ticket-id", required=True)
    resolve = commands.add_parser("resolve-ticket")
    resolve.add_argument("--ticket-id", required=True)
    resolve.add_argument("--resolution-note", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = build_live_service(mcp_url=args.mcp_url)

    if args.command == "start":
        result = service.start_run(
            GRAPH_NAME,
            {
                "shipment_id": args.shipment_id,
                "session_id": args.session_id,
                "employee_id": args.employee_id,
                "failure_reason": args.failure_reason,
            },
        )
    elif args.command == "status":
        result = service.get_run(args.run_id)
    elif args.command == "customer":
        payload = json.loads(args.choice_json)
        if not isinstance(payload, dict):
            raise ValueError("--choice-json must contain one JSON object.")
        result = service.submit_external_input(args.run_id, payload)
    elif args.command == "hitl-tasks":
        result = service.pending_hitl_tasks()
    elif args.command == "resolve-hitl":
        result = service.resolve_hitl(
            args.task_id,
            approved=args.approve,
            note=args.note,
            admin_employee_id=args.admin_employee_id,
        )
    elif args.command == "tickets":
        status = TicketStatus(args.status) if args.status else None
        result = service.tickets(status)
    elif args.command == "investigate-ticket":
        result = service.investigate_ticket(args.ticket_id)
    else:
        result = service.resolve_ticket(
            args.ticket_id, resolution_note=args.resolution_note
        )
    _print(result)


if __name__ == "__main__":
    main()
