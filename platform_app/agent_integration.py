from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

MEMORY_RAG_AGENT_ID = "memory_rag_agent"
PLANNING_AGENT_ID = "planning_agent"

_MEMORY_START = re.compile(
    r"^start\s+customer\s+(?P<customer>\d+)"
    r"(?:\s*,?\s*role\s+(?P<role>sales_rep|finance_manager))?$",
    re.IGNORECASE,
)
_PLANNING_START = re.compile(
    r"^plan\s+shipment\s+(?P<shipment>\d+)"
    r"\s*,?\s*customer\s+(?P<customer>\d+)"
    r"\s*,?\s*employee\s+(?P<employee>\d+)"
    r"(?:\s*,?\s*method\s+(?P<method>decomposition_first|dynamic))?$",
    re.IGNORECASE,
)


def _default_memory_agent() -> Any:
    from agent.agent_loop import AgentLoop

    return AgentLoop()


def _default_planning_runner(
    *,
    shipment_id: int,
    customer_id: int,
    employee_id: int,
    method: str,
    run_id: str,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing from the root .env file.")

    async def operation() -> dict[str, Any]:
        from langchain_mistralai import ChatMistralAI

        from agent.client import SwiftrailAgent
        from planning.orchestrator import (
            DecompositionMethod,
            SwiftrailPlanningOrchestrator,
        )

        mcp_url = os.getenv("SWIFTRAIL_MCP_URL", "http://127.0.0.1:8000/mcp")
        agent = SwiftrailAgent("http", mcp_url)
        try:
            await agent.connect()
            raw_auth = await agent.call_tool(
                "authenticate",
                {
                    "request": {
                        "session_id": run_id,
                        "employee_id": employee_id,
                    }
                },
            )
            auth = agent.decode_tool_result(raw_auth)
            if not isinstance(auth, dict) or auth.get("success") is not True:
                raise RuntimeError(f"Planning-agent authentication failed: {auth}")

            llm = ChatMistralAI(
                model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
                api_key=api_key,
                temperature=0.1,
                max_retries=2,
            )
            orchestrator = SwiftrailPlanningOrchestrator(
                agent=agent,
                session_id=run_id,
                llm=llm,
                employee_id=employee_id,
            )
            outcome = await orchestrator.run(
                shipment_id=shipment_id,
                customer_id=customer_id,
                method=DecompositionMethod(method),
            )
            return {
                "method": outcome.method.value,
                "result": outcome.result,
                "artifact_path": outcome.artifact_path,
                "action_results": [asdict(item) for item in outcome.action_results],
            }
        finally:
            await agent.close()

    return asyncio.run(operation())


class PlatformAgentIntegration:
    """Expose the existing memory/RAG and planning agents to the platform."""

    def __init__(
        self,
        memory_agent_factory: Callable[[], Any] = _default_memory_agent,
        planning_runner: Callable[..., dict[str, Any]] = _default_planning_runner,
    ) -> None:
        self._memory_agent_factory = memory_agent_factory
        self._planning_runner = planning_runner
        self._memory_agent: Any | None = None
        self._runs: dict[str, dict[str, Any]] = {}

    @property
    def memory_agent(self) -> Any:
        if self._memory_agent is None:
            self._memory_agent = self._memory_agent_factory()
        return self._memory_agent

    def chat(self, agent_id: str, message: str, run_id: str | None) -> dict[str, Any]:
        if agent_id == MEMORY_RAG_AGENT_ID:
            return self._chat_memory(message, run_id)
        if agent_id == PLANNING_AGENT_ID:
            return self._chat_planning(message, run_id)
        raise KeyError(agent_id)

    def _chat_memory(self, message: str, run_id: str | None) -> dict[str, Any]:
        if run_id is None:
            match = _MEMORY_START.fullmatch(message.strip())
            if match is None:
                return {
                    "reply": (
                        "Start a customer-scoped session with: "
                        "start customer 3, role finance_manager"
                    )
                }
            customer_id = int(match.group("customer"))
            role = (match.group("role") or "sales_rep").lower()
            run_id = self.memory_agent.start(customer_id=customer_id)
            self._runs[run_id] = {
                "run": {
                    "run_id": run_id,
                    "graph_name": MEMORY_RAG_AGENT_ID,
                    "status": "active",
                    "current_node": "await_query",
                },
                "history": [{"sequence": 1, "node_name": "session_started"}],
                "state": {
                    "customer_id": customer_id,
                    "role": role,
                    "last_result": None,
                },
            }
            return {
                "run_id": run_id,
                "reply": (
                    f"Memory/RAG session started for customer {customer_id} as {role}. "
                    "Ask a policy question or ask what happened previously."
                ),
                "status": "active",
                "current_node": "await_query",
            }

        record = self._require_run(run_id, MEMORY_RAG_AGENT_ID)
        try:
            result = self.memory_agent.process(
                run_id,
                [{"role": "employee", "content": message}],
                role=record["state"]["role"],
            )
        except Exception as exc:
            record["run"].update(status="failed", current_node="agent_failed")
            record["history"].append(
                {
                    "sequence": len(record["history"]) + 1,
                    "node_name": "agent_failed",
                }
            )
            record["state"]["error"] = str(exc)
            return {
                "run_id": run_id,
                "reply": f"The Memory/RAG request failed safely: {exc}",
                "status": "failed",
                "current_node": "agent_failed",
            }

        category = str(result["category"])
        record["run"].update(status="active", current_node=category)
        record["history"].append(
            {"sequence": len(record["history"]) + 1, "node_name": category}
        )
        record["state"]["last_result"] = result
        return {
            "run_id": run_id,
            "reply": result["answer"],
            "status": "active",
            "current_node": category,
        }

    def _chat_planning(self, message: str, run_id: str | None) -> dict[str, Any]:
        match = _PLANNING_START.fullmatch(message.strip())
        if match is None:
            if run_id is not None:
                record = self._require_run(run_id, PLANNING_AGENT_ID)
                return {
                    "run_id": run_id,
                    "reply": (
                        "This planning run is already complete. Start another plan with: "
                        "plan shipment 3, customer 3, employee 1"
                    ),
                    "status": record["run"]["status"],
                    "current_node": record["run"]["current_node"],
                }
            return {
                "reply": (
                    "Start the live planning agent with: "
                    "plan shipment 3, customer 3, employee 1, "
                    "method decomposition_first"
                )
            }

        planning_run_id = f"planning-{uuid.uuid4().hex}"
        method = (match.group("method") or "decomposition_first").lower()
        self._runs[planning_run_id] = {
            "run": {
                "run_id": planning_run_id,
                "graph_name": PLANNING_AGENT_ID,
                "status": "running",
                "current_node": "authenticate_mcp",
            },
            "history": [{"sequence": 1, "node_name": "authenticate_mcp"}],
            "state": {
                "shipment_id": int(match.group("shipment")),
                "customer_id": int(match.group("customer")),
                "employee_id": int(match.group("employee")),
                "method": method,
            },
        }
        record = self._runs[planning_run_id]

        try:
            outcome = self._planning_runner(
                shipment_id=record["state"]["shipment_id"],
                customer_id=record["state"]["customer_id"],
                employee_id=record["state"]["employee_id"],
                method=method,
                run_id=planning_run_id,
            )
        except Exception as exc:
            record["run"].update(status="failed", current_node="planning_failed")
            record["history"].append(
                {"sequence": 2, "node_name": "planning_failed"}
            )
            record["state"]["error"] = str(exc)
            return {
                "run_id": planning_run_id,
                "reply": f"The planning run failed safely: {exc}",
                "status": "failed",
                "current_node": "planning_failed",
            }

        record["run"].update(status="completed", current_node="complete")
        record["history"].extend(
            {"sequence": index, "node_name": node}
            for index, node in enumerate(
                ("decompose", "execute_subtasks", "verify_actions", "complete"),
                start=2,
            )
        )
        record["state"]["outcome"] = outcome
        action_summary = "; ".join(
            f"{item['action']}: {'success' if item['success'] else 'failed'}"
            for item in outcome.get("action_results", [])
        )
        suffix = f" Actions: {action_summary}." if action_summary else ""
        return {
            "run_id": planning_run_id,
            "reply": f"{outcome['result']}{suffix}",
            "status": "completed",
            "current_node": "complete",
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.get(run_id)

    def _require_run(self, run_id: str, agent_id: str) -> dict[str, Any]:
        record = self._runs.get(run_id)
        if record is None or record["run"]["graph_name"] != agent_id:
            raise ValueError(f"No {agent_id} run exists for {run_id}.")
        return record
