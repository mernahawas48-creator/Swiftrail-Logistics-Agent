"""Planning Agent Orchestration.

This is the single entry point the rest of the system (agent/agent_loop.py,
planning_eval's comparison harness, or a small CLI) calls to run the new
planning agent against a real "review this blocked shipment" request. It
owns exactly one job: wiring decomposition, execution, routing, and tracing
together correctly -- it re-executes nothing that decomposition.py,
dynamic_decomposition.py, execution_adapter.py, planning_router.py, or
environment.py already do.

This agent is intentionally separate from agent.agent_loop.AgentLoop (the
memory/RAG agent): it does not import or modify that module, and it opens
its own connection to the same mcp_server/ the memory/RAG agent already
uses, the same way agent/demo.py does.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

from .algorithms.decomposition import decompose_blocked_shipment, execute_plan_swiftrail
from .algorithms.dynamic_decomposition import dynamic_decompose_blocked_shipment
from .algorithms.environment import Environment
from .divergence import compute_divergence
from .execution_adapter import SubtaskExecutionAdapter
from .trace_logger import log_decomposition_first_run, log_divergence, log_dynamic_run

if TYPE_CHECKING:  # pragma: no cover
    from agent.client import SwiftrailAgent


class DecompositionMethod(str, Enum):
    DECOMPOSITION_FIRST = "decomposition_first"
    DYNAMIC = "dynamic"


@dataclass
class PlanningRunResult:
    method: DecompositionMethod
    result: str
    artifact_path: str


class SwiftrailPlanningOrchestrator:
    """Owns one authenticated MCP connection and runs either decomposition
    method against it. Grounding for LATS/Reflexion sub-tasks (owned by the
    self-correction concern) is threaded through as a real, non-random
    ``Environment`` -- see algorithms/environment.py -- not the toolkit's randomized evaluator. """
    def __init__(
        self,
        agent: "SwiftrailAgent",
        session_id: str,
        llm: BaseChatModel,
        employee_id: int,
    ):
        self.agent = agent
        self.session_id = session_id
        self.llm = llm
        self.employee_id = employee_id
    async def run(
        self,
        shipment_id: int,
        customer_id: int,
        method: DecompositionMethod = DecompositionMethod.DECOMPOSITION_FIRST,
    ) -> PlanningRunResult:
        if method is DecompositionMethod.DECOMPOSITION_FIRST:
            return await self._run_decomposition_first(shipment_id, customer_id)
        return await self._run_dynamic(shipment_id, customer_id)

    async def _run_decomposition_first(self, shipment_id: int, customer_id: int) -> PlanningRunResult:
        static_plan = decompose_blocked_shipment(
            shipment_id,
            customer_id,
            self.llm,
        )

        environment = Environment(
            shipment_id=shipment_id,
            employee_id=self.employee_id,
        )

        adapter = SubtaskExecutionAdapter(
            agent=self.agent,
            session_id=self.session_id,
            llm=self.llm,
            environment=environment,
        )
        outputs, llm_calls = await execute_plan_swiftrail(static_plan, adapter)
        terminal = static_plan.terminal_tasks()
        if len(terminal) != 1:
            raise ValueError(f"Expected exactly one terminal reasoning task, found {terminal}")
        result = outputs[terminal[0]]

        artifact = log_decomposition_first_run(
            shipment_id=shipment_id,
            customer_id=customer_id,
            plan_dump=static_plan.plan.model_dump(),
            execution_history=adapter.history,
            final_result=result,
            total_llm_calls=llm_calls,
        )
        return PlanningRunResult(DecompositionMethod.DECOMPOSITION_FIRST, result, str(artifact))

    async def _run_dynamic(self, shipment_id: int, customer_id: int) -> PlanningRunResult:
        async def call_tool(tool_name: str, args: dict) -> dict:
            raw = await self.agent.call_tool(tool_name, args)
            return self.agent.decode_tool_result(raw)

        environment = Environment(
            shipment_id=shipment_id,
            employee_id=self.employee_id,
        )

        steps = await dynamic_decompose_blocked_shipment(
            shipment_id=shipment_id,
            customer_id=customer_id,
            session_id=self.session_id,
            llm=self.llm,
            call_tool=call_tool,
            environment=environment,
        )
        result = steps[-1].output if steps else "No steps were executed."
        # Roughly one LLM call per non-forced decision, plus the final
        # synthesis call; forced steps (the safety override) cost zero LLM
        # calls for the *decision* itself, which is part of why dynamic
        # decomposition's real per-run cost is data, not a guess -- see
        # planning_eval's comparison table.
        llm_calls = sum(1 for s in steps if not s.forced) + 1
        artifact = log_dynamic_run(
            shipment_id=shipment_id,
            customer_id=customer_id,
            steps=steps,
            final_result=result,
            total_llm_calls=llm_calls,
        )
        return PlanningRunResult(DecompositionMethod.DYNAMIC, result, str(artifact))

    async def run_both_and_log_divergence(
        self,
        shipment_id: int,
        customer_id: int,
    ):
        """Run both decomposition methods and record their divergence."""
        static_plan = decompose_blocked_shipment(shipment_id, customer_id, self.llm)

        async def call_tool(tool_name: str, args: dict) -> dict:
            raw = await self.agent.call_tool(tool_name, args)
            return self.agent.decode_tool_result(raw)

        environment = Environment(
            shipment_id=shipment_id,
            employee_id=self.employee_id,
        )

        dynamic_steps = await dynamic_decompose_blocked_shipment(
            shipment_id=shipment_id,
            customer_id=customer_id,
            session_id=self.session_id,
            llm=self.llm,
            call_tool=call_tool,
            environment=environment,
        )
        divergence = compute_divergence(static_plan, dynamic_steps)
        artifact = log_divergence(divergence)
        return divergence, str(artifact)
