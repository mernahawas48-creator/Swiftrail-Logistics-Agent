from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from ..swiftrail_subtask import SubtaskKind
from .decomposition import _TOOL_BINDINGS, TOOL_CATALOG


class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str


def dynamic_decomposition(goal: str, llm: BaseChatModel, max_steps: int = 4) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    for step in range(max_steps):
        observation = "\n".join(f"{task}: {result}" for task, result in history) or "None"
        decision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke([
            ("system", "You are an adaptive planner. Use prior observations before deciding what comes next."),
            ("human", f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task. Set done to true only when the goal is met.
When done is true, use an empty string for next_task."""),
        ], temperature=0.1)
        if decision.done:
            break
        task = decision.next_task.strip()
        if not task:
            raise ValueError(f"Dynamic planner omitted next_task at step {step + 1}")
        response = llm.invoke([
            ("system", "Execute the next adaptive sub-task using the observations provided."),
            ("human", f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}"),
        ], temperature=0.2)
        result = response.content
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        result = result.strip()
        history.append((task, result))
    return history


# ---------------------------------------------------------------------------
# Swiftrail dynamic decomposition: "review a blocked shipment", interleaved
# ---------------------------------------------------------------------------
#
# Unlike decompose_blocked_shipment() (decomposition-first: the whole DAG is
# committed before any tool has run), this generates one step, actually
# executes it against the real MCP server, and only then decides the next
# step. The reason this is not busywork for this request type specifically:
# a severe, active credit hold changes what the *safe* next step even is
# (escalate, don't attempt release), and decomposition-first has no
# mechanism to notice that mid-plan -- it just executes whatever
# fetch_credit_holds -> ... node ordering the up-front plan happened to
# contain. See divergence.py for where this is made explicit and measured.

from dataclasses import dataclass, field  # noqa: E402  (grouped near use, not at top, to keep the diff to the generic function above minimal)
from typing import Awaitable, Callable  # noqa: E402


class DynamicSwiftrailDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    kind: SubtaskKind | None = None
    tool_name: str | None = None
    instruction: str | None = None
    rationale: str = ""


@dataclass
class DynamicStep:
    step: int
    kind: SubtaskKind
    tool_name: str | None
    instruction: str | None
    output: str
    raw: dict | None = None
    # True when a hard-coded safety rule -- not the model -- chose this step.
    # This is what "grounded" means for dynamic replanning: the divergence
    # is not "the model felt like doing something different", it's a fact
    # the model is not allowed to reason its way around.
    forced: bool = False


def _forced_next_step(observed: dict[str, dict]) -> DynamicSwiftrailDecision | None:
    holds_payload = observed.get("fetch_credit_holds") or {}
    active_holds = holds_payload.get("data", {}).get("active_holds", [])
    if any(h.get("severity") == "severe" for h in active_holds):
        return DynamicSwiftrailDecision(
            done=False,
            kind=SubtaskKind.REASONING,
            instruction=(
                "A severe, active credit hold was found on this customer. Do not "
                "propose releasing it yourself. Produce an escalation to a "
                "finance manager naming the exact hold id and reason, and state "
                "which parts of the request (if any) remain safe to resolve "
                "without release, e.g. auto-approvable rate exceptions."
            ),
            rationale=(
                "hard safety rule: a severe active hold forces an escalation "
                "step next, overriding whatever the model would otherwise pick"
            ),
        )
    return None


async def dynamic_decompose_blocked_shipment(
    shipment_id: int,
    customer_id: int,
    session_id: str,
    llm: BaseChatModel,
    call_tool: Callable[[str, dict], Awaitable[dict]],
    max_steps: int = 6,
    environment=None,
) -> list[DynamicStep]:
    """Interleaved decomposition. ``call_tool`` is injected (rather than
    importing planning.execution_adapter directly) so this module has no
    dependency on planning_router/execution_adapter and cannot form an
    import cycle with the algorithms package `__init__.py` that already
    imports this module."""

    goal = (
        f"Review shipment {shipment_id} and customer {customer_id}. Identify the "
        "financial blockers, determine the safest resolution sequence, perform only "
        "the actions permitted by my authority, and escalate anything that cannot be "
        "resolved safely."
    )
    observed: dict[str, dict] = {}
    steps: list[DynamicStep] = []

    for step_no in range(1, max_steps + 1):
        forced = _forced_next_step(observed)
        if forced is not None:
            decision = forced
        else:
            observation = json.dumps(observed, default=str) if observed else "None yet."
            decision = llm.with_structured_output(
                DynamicSwiftrailDecision,
                method="json_schema",
            ).invoke([
                (
                    "system",
                    "You are an adaptive Swiftrail planner. Decide only the single "
                    "next step from what has actually been observed so far -- do not "
                    "plan ahead.\n\n" + TOOL_CATALOG,
                ),
                (
                    "human",
                    f"Goal: {goal}\nObserved tool results so far:\n{observation}\n\n"
                    "If every needed fact has been gathered, set done=true and "
                    "kind='reasoning' with an instruction to produce the final "
                    "resolution sequence. Otherwise pick exactly one next tool_call "
                    "tool id from the catalog above, or a reasoning step if a "
                    "decision (not another lookup) is what's needed next.",
                ),
            ], temperature=0.1)

        if decision.kind == SubtaskKind.TOOL_CALL:
            if decision.tool_name not in _TOOL_BINDINGS:
                raise ValueError(f"step {step_no}: unknown tool id {decision.tool_name!r}")
            resolved_name, arg_fn = _TOOL_BINDINGS[decision.tool_name]
            args = arg_fn(session_id, {}, shipment_id, customer_id)
            raw = await call_tool(resolved_name, args)
            observed[decision.tool_name] = raw
            steps.append(
                DynamicStep(
                    step=step_no,
                    kind=SubtaskKind.TOOL_CALL,
                    tool_name=resolved_name,
                    instruction=None,
                    output=json.dumps(raw, default=str),
                    raw=raw,
                    forced=forced is not None,
                )
            )
            if decision.done:
                break
            continue

        # Reasoning step: route the sub-task through the shared planning router.
        # Local import avoids an import cycle with the algorithms package.
        from ..planning_router import PlanningProfile, solve_subtask

        context = json.dumps(observed, default=str)

        profile = PlanningProfile(
            instruction=(
                decision.instruction
                or "Produce the final safe Swiftrail resolution sequence."
            ),
            context=f"Goal: {goal}\nObserved tool results:\n{context}",
            needs_branching=True,
            high_stakes=True,
            grounded_validation_available=environment is not None,
        )

        routed = solve_subtask(
            profile,
            llm,
            environment=environment,
        )
        steps.append(
            DynamicStep(
                step=step_no,
                kind=SubtaskKind.REASONING,
                tool_name=None,
                instruction=decision.instruction,
                output=routed.output,
                forced=forced is not None,
            )
        )
        break

    return steps
