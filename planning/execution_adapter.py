"""Subtask Execution Adapter.

This is the seam between "we have a validated DAG of sub-tasks" and "sub-tasks
actually happen": for a TOOL_CALL sub-task it calls the real MCP server
through the same ``SwiftrailAgent`` the memory/RAG agent uses (agent/client.py) --
no new client, no direct DB access from planning code. For a REASONING
sub-task it builds a ``PlanningProfile`` and hands off to
``planning_router.solve_subtask`` (Plan-and-Solve / Tree of Thoughts / LATS).

Every sub-task's textual "output" (what downstream, dependent sub-tasks see
in their context) is grounded: for tool calls it's the tool's actual JSON
result, not a model's guess at what the tool would return. This is what
makes the DAG's dependency edges meaningful instead of decorative.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

from .planning_router import PlanningProfile, RoutedPlanningResult, solve_subtask
from .swiftrail_subtask import SubtaskKind, SubtaskMeta

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from agent.client import SwiftrailAgent
    from .algorithms.environment import Environment


class SubtaskExecutionError(RuntimeError):
    """Raised when a deterministic tool call fails; the caller (dynamic
    decomposition / the orchestrator) decides whether that is a hard stop
    or a signal to replan, it is not swallowed here."""

    def __init__(self, task_id: str, tool_name: str, error_code: str, message: str):
        super().__init__(f"[{task_id}] {tool_name} -> {error_code}: {message}")
        self.task_id = task_id
        self.tool_name = tool_name
        self.error_code = error_code


@dataclass
class SubtaskExecution:
    """One executed node's evidence, kept for trace logging and divergence
    comparison. ``raw_tool_result`` is only set for TOOL_CALL sub-tasks so
    the trace can show real MCP payloads, not just the summarized text."""

    task_id: str
    kind: SubtaskKind
    output: str
    duration_s: float
    tool_name: str | None = None
    raw_tool_result: dict | None = None
    routed: RoutedPlanningResult | None = None


@dataclass
class SubtaskExecutionAdapter:
    agent: "SwiftrailAgent"
    session_id: str
    llm: BaseChatModel
    environment: "Environment | None" = None
    # Every executed node, in execution order -- consumed by trace_logger
    # and divergence.py.
    history: list[SubtaskExecution] = field(default_factory=list)

    async def execute(
        self,
        task_id: str,
        instruction: str,
        meta: SubtaskMeta,
        outputs_so_far: dict[str, str],
    ) -> str:
        started = time.monotonic()
        if meta.kind is SubtaskKind.TOOL_CALL:
            output, raw = await self._execute_tool_call(task_id, meta, outputs_so_far)
            self.history.append(
                SubtaskExecution(
                    task_id=task_id,
                    kind=meta.kind,
                    output=output,
                    duration_s=time.monotonic() - started,
                    tool_name=meta.tool_name,
                    raw_tool_result=raw,
                )
            )
            return output

        result = self._execute_reasoning(task_id, instruction, meta, outputs_so_far)
        self.history.append(
            SubtaskExecution(
                task_id=task_id,
                kind=meta.kind,
                output=result.output,
                duration_s=time.monotonic() - started,
                routed=result,
            )
        )
        return result.output

    async def _execute_tool_call(
        self,
        task_id: str,
        meta: SubtaskMeta,
        outputs_so_far: dict[str, str],
    ) -> tuple[str, dict]:
        if meta.tool_name is None or meta.build_args is None:
            raise ValueError(f"{task_id}: TOOL_CALL sub-task is missing tool_name/build_args")

        args = meta.build_args(self.session_id, outputs_so_far)
        result = await self.agent.call_tool(
            meta.tool_name,
            {"request": args},
        )
        payload = self.agent.decode_tool_result(result)
        if not isinstance(payload, dict):
            raise SubtaskExecutionError(task_id, meta.tool_name, "MALFORMED_RESPONSE", str(payload))
        # Every mcp_server/tool_support.ok()/fail() envelope carries an explicit
        # boolean "success" field -- that is the ground truth, not the presence
        # or absence of any particular key.
        if payload.get("success") is not True:
            raise SubtaskExecutionError(
                task_id,
                meta.tool_name,
                payload.get("code", "UNKNOWN_ERROR"),
                payload.get("message", "Tool call failed"),
            )
        # The grounded fact downstream reasoning tasks read. Keeping this as
        # compact JSON (not a prose paraphrase) is deliberate: it is what
        # lets a later reasoning sub-task or the environment validator check
        # an exact field (severity, discount_pct, role) instead of trusting
        # a summary.
        return json.dumps(payload, default=str), payload

    def _execute_reasoning(
        self,
        task_id: str,
        instruction: str,
        meta: SubtaskMeta,
        outputs_so_far: dict[str, str],
    ) -> RoutedPlanningResult:
        context = "\n\n".join(
            f"OUTPUT FROM {dep_id}:\n{dep_output}" for dep_id, dep_output in outputs_so_far.items()
        ) or "No prerequisite outputs."
        profile = PlanningProfile(
            instruction=instruction,
            context=context,
            needs_branching=meta.needs_branching,
            high_stakes=meta.high_stakes,
            grounded_validation_available=meta.grounded_validation_available,
        )
        return solve_subtask(profile, self.llm, environment=self.environment)
