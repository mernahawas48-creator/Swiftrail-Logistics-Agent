from __future__ import annotations

import asyncio
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from ..models import Plan
from ..swiftrail_subtask import SubtaskKind, SubtaskMeta, SwiftrailPlan


PLANNER_SYSTEM = """You are a careful task-decomposition planner.
Produce a small executable DAG, not a prose checklist. Every task must make a concrete
contribution to the goal. Independent research or analysis tasks should be parallel.
The plan must end with exactly one synthesis task depending on every necessary branch."""


class PlannedTask(BaseModel):
    """Wire schema; richer semantic constraints are applied by the Task domain model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]


class GeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


def decompose_goal(goal: str, llm: BaseChatModel) -> Plan:
    generated = llm.with_structured_output(
        GeneratedPlan,
        method="json_schema",
    ).invoke([
        ("system", PLANNER_SYSTEM),
        ("human", f"""Decompose this goal into 3-6 tasks: {goal!r}
Use short task ids such as t1. Dependencies may refer only to tasks in the plan.
Preserve the supplied goal exactly in the plan's goal field."""),
    ], temperature=0.1)
    # The caller's goal remains authoritative even if the model paraphrases it.
    payload = generated.model_dump()
    payload["goal"] = goal
    return Plan.model_validate(payload)


def execute_plan(plan: Plan, llm: BaseChatModel, max_workers: int = 4) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for batch in plan.execution_batches():
        prompts: dict[str, str] = {}
        for task_id in batch:
            task = plan.task(task_id)
            context = "\n\n".join(
                f"OUTPUT FROM {dependency}:\n{outputs[dependency]}"
                for dependency in task.depends_on
            ) or "No prerequisite outputs."
            prompts[task_id] = f"""Overall goal: {plan.goal}
                Current task: {task.instruction}
                Prerequisite outputs:
                {context}
                Complete only the current task. Be concrete and concise. Do not invent sources."""
        # unnecessary but nice to have
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
            futures = {
                pool.submit(
                    llm.invoke,
                    [
                        ("system", "You execute one node in a validated task DAG."),
                        ("human", prompt),
                    ],
                    temperature=0.2,
                ): task_id
                for task_id, prompt in prompts.items()
            }
            for future in as_completed(futures):
                content = future.result().content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("The chat model returned an empty or unsupported response")
                outputs[futures[future]] = content.strip()
    return outputs


def final_output(plan: Plan, outputs: dict[str, str]) -> str:
    terminals = plan.terminal_tasks()
    if len(terminals) != 1:
        raise ValueError(f"Expected exactly one terminal synthesis task, found {terminals}")
    return outputs[terminals[0]]


# ---------------------------------------------------------------------------
# Swiftrail decomposition-first: "review a blocked shipment"
# ---------------------------------------------------------------------------
#
# The generic decompose_goal() above hands the model a free-text goal and
# lets it invent both the DAG and each node's content from scratch, with no
# grounding, and that is exactly what the lab says not to ship: "not the
# toolkit's generic demo prompts". For our recurring real request --
#   "Review shipment {id} and customer {id}. Identify the financial
#    blockers, determine the safest resolution sequence, perform only the
#    actions permitted by my authority, and escalate anything that cannot
#    be resolved safely."
# -- the *evidence-gathering* nodes are always the same five deterministic
# MCP lookups (get_shipment_status, search_customer, list_customer_invoices,
# and, once we know which holds/exceptions exist, release_credit_hold /
# approve_rate_exception are NOT called here -- decomposition-first is
# read-only reconnaissance; writes only ever happen through the reasoning
# synthesis task's recommendation, never blindly). What genuinely benefits
# from an LLM call is the DAG *shape decision* itself: whether the
# credit-hold and rate-exception lookups are worth doing depends on what the
# shipment/customer/invoice lookups already returned, and later revisions of
# this lab's request wording may add or drop branches. So we still let the
# model propose the plan, but we ground its choices to the fixed, real tool
# catalog below instead of letting it invent free-text steps.

TOOL_CATALOG = """Available MCP tools (use exactly these tool ids, no others):
- fetch_shipment: get_shipment_status(shipment_id) -> shipment status, rate, customer_id
- fetch_customer: search_customer(customer_id) -> credit_limit, balance_due, credit_status
- fetch_invoices: list_customer_invoices(customer_id) -> overdue invoices for the customer
- fetch_credit_holds: list_customer_credit_holds(customer_id) -> this customer's own holds
- fetch_rate_exception: get_shipment_rate_exception(shipment_id) -> this shipment's rate exception
Any task that is not one of the tool lookups above is a reasoning task: it must
synthesize a resolution sequence from the lookups' outputs, respecting role authority
(sales_rep vs finance_manager) and escalating whatever it cannot safely resolve itself."""


class ReasoningRole(str, Enum):
    LINEAR = "linear"
    BRANCHING = "branching"
    FINAL = "final"
class SwiftrailPlannedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]
    kind: SubtaskKind
    tool_name: str | None = None
    reasoning_role: ReasoningRole | None = None


class SwiftrailGeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[SwiftrailPlannedTask]


# Fixed builders for each deterministic tool id in TOOL_CATALOG. Kept as a
# static registry (not generated by the LLM) because the *arguments* to a
# real, writable-adjacent MCP server must never be model-authored strings --
# only real ids captured in the request or already-fetched tool output.
def _shipment_args(session_id: str, _: dict[str, str], shipment_id: int) -> dict:
    return {"session_id": session_id, "shipment_id": shipment_id}


def _customer_args(session_id: str, _: dict[str, str], customer_id: int) -> dict:
    return {"session_id": session_id, "customer_id": customer_id}


_TOOL_BINDINGS = {
    "fetch_shipment": ("get_shipment_status", lambda sid, outs, shipment_id, customer_id: _shipment_args(sid, outs, shipment_id)),
    "fetch_customer": ("search_customer", lambda sid, outs, shipment_id, customer_id: _customer_args(sid, outs, customer_id)),
    "fetch_invoices": ("list_customer_invoices", lambda sid, outs, shipment_id, customer_id: _customer_args(sid, outs, customer_id)),
    "fetch_credit_holds": ("list_customer_credit_holds", lambda sid, outs, shipment_id, customer_id: _customer_args(sid, outs, customer_id)),
    "fetch_rate_exception": ("get_shipment_rate_exception", lambda sid, outs, shipment_id, customer_id: _shipment_args(sid, outs, shipment_id)),
}


def decompose_blocked_shipment(
    shipment_id: int,
    customer_id: int,
    llm: BaseChatModel,
) -> SwiftrailPlan:
    """Decomposition-first: generate the *entire* DAG in one shot, before any
    tool has been called, then execute it in topological order. This is the
    method that gets blindsided when an early lookup reveals something the
    up-front plan did not anticipate -- see dynamic_decomposition.py and
    divergence.py for the case where that actually happens."""

    goal = (
        f"Review shipment {shipment_id} and customer {customer_id}. Identify the "
        "financial blockers, determine the safest resolution sequence, perform only "
        "the actions permitted by my authority, and escalate anything that cannot be "
        "resolved safely."
    )
    generated = llm.with_structured_output(
        SwiftrailGeneratedPlan,
        method="json_schema",
    ).invoke([
        ("system", PLANNER_SYSTEM + "\n\n" + TOOL_CATALOG),
        (
            "human",
            f"Decompose this goal into a DAG of 6-8 tasks: {goal!r}\n"
        "Every deterministic lookup must set kind='tool_call', tool_name to one of "
        "the tool ids listed above, and reasoning_role=null. Use exactly three "
        "kind='reasoning' tasks, each with tool_name=null:\n"
        "1) reasoning_role='linear': summarize the confirmed blockers and authority/"
        "policy constraints from the gathered evidence; it must depend on the selected "
        "tool_call evidence tasks.\n"
        "2) reasoning_role='branching': compare multiple plausible safe resolution "
        "sequences and their trade-offs; it must depend on the linear reasoning task.\n"
        "3) reasoning_role='final': choose the final safe executable/escalation plan; "
        "it must depend on the branching reasoning task and be the only terminal task.\n"
        "Do not invent tool ids.",
        ),
    ], temperature=0.1)

    plan_payload = {
        "goal": goal,
        "tasks": [
            t.model_dump(
                exclude={"kind", "tool_name", "reasoning_role"}
            )
            for t in generated.tasks
        ],
    }
    plan = Plan.model_validate(
        plan_payload
    )  # enforces acyclicity, unique ids, known deps

    # Validate the three reasoning roles before building routing metadata.
    reasoning_tasks = [
        task
        for task in generated.tasks
        if task.kind is SubtaskKind.REASONING
    ]

    expected_roles = {
        ReasoningRole.LINEAR,
        ReasoningRole.BRANCHING,
        ReasoningRole.FINAL,
    }

    actual_roles = [
        task.reasoning_role
        for task in reasoning_tasks
    ]

    if len(reasoning_tasks) != 3 or set(actual_roles) != expected_roles:
        raise ValueError(
            "The Swiftrail decomposition must contain exactly one linear, "
            "one branching, and one final reasoning task."
        )

    role_tasks = {
        task.reasoning_role: task
        for task in reasoning_tasks
    }

    linear_task = role_tasks[ReasoningRole.LINEAR]
    branching_task = role_tasks[ReasoningRole.BRANCHING]
    final_task = role_tasks[ReasoningRole.FINAL]

    tool_task_ids = {
        task.id
        for task in generated.tasks
        if task.kind is SubtaskKind.TOOL_CALL
    }

    # Linear reasoning must receive all gathered evidence.
    if not tool_task_ids.issubset(set(linear_task.depends_on)):
        raise ValueError(
            "The linear reasoning task must depend on all selected "
            "tool-call evidence tasks."
        )

    # Branching reasoning uses the linear analysis.
    if linear_task.id not in branching_task.depends_on:
        raise ValueError(
            "The branching reasoning task must depend on the linear task."
        )

    # Final high-stakes reasoning uses the alternatives from ToT.
    if branching_task.id not in final_task.depends_on:
        raise ValueError(
            "The final reasoning task must depend on the branching task."
        )

    # The final reasoning node must be the only terminal node.
    if plan.terminal_tasks() != [final_task.id]:
        raise ValueError(
            "The final reasoning task must be the plan's only terminal task."
        )

    meta: dict[str, SubtaskMeta] = {}

    for t in generated.tasks:
        if t.kind is SubtaskKind.TOOL_CALL:
            if t.tool_name not in _TOOL_BINDINGS:
                raise ValueError(
                    f"{t.id}: unknown or missing tool_name {t.tool_name!r}"
                )

            if t.reasoning_role is not None:
                raise ValueError(
                    f"{t.id}: tool-call tasks must not set reasoning_role"
                )

            resolved_name, arg_fn = _TOOL_BINDINGS[t.tool_name]

            meta[t.id] = SubtaskMeta(
                kind=SubtaskKind.TOOL_CALL,
                tool_name=resolved_name,
                build_args=lambda sid, outs, _fn=arg_fn: _fn(
                    sid,
                    outs,
                    shipment_id,
                    customer_id,
                ),
            )

        else:
            if t.tool_name is not None or t.reasoning_role is None:
                raise ValueError(
                    f"{t.id}: reasoning tasks require reasoning_role "
                    "and must not set tool_name"
                )

            if t.reasoning_role is ReasoningRole.LINEAR:
                # Linear evidence analysis -> Plan-and-Solve
                needs_branching = False
                high_stakes = False

            elif t.reasoning_role is ReasoningRole.BRANCHING:
                # Compare alternatives -> Tree of Thoughts
                needs_branching = True
                high_stakes = False

            else:
                # Final grounded high-stakes decision -> LATS
                needs_branching = True
                high_stakes = True

            meta[t.id] = SubtaskMeta(
                kind=SubtaskKind.REASONING,
                needs_branching=needs_branching,
                high_stakes=high_stakes,
                grounded_validation_available=True,
            )

    return SwiftrailPlan(plan=plan, meta=meta)


async def execute_plan_swiftrail(plan: SwiftrailPlan, adapter: Any) -> tuple[dict[str, str], int]:
    """Executes a SwiftrailPlan through a SubtaskExecutionAdapter, batch by
    batch, reusing the same topological_generations-derived batching
    decomposition-first already committed to (SwiftrailPlan.execution_batches,
    itself a thin pass-through to the toolkit's Plan.execution_batches).
    ``adapter`` is typed as Any (not planning.execution_adapter.SubtaskExecutionAdapter)
    purely to avoid a real import cycle: execution_adapter -> planning_router
    -> algorithms/__init__.py -> this module.

    Returns (outputs, approx_llm_call_count). Every tool_call sub-task costs
    zero LLM calls -- its output *is* the grounded tool result -- so this
    count only reflects reasoning sub-tasks, which is exactly the number
    the decomposition-first row of the cost table needs.
    """

    outputs: dict[str, str] = {}
    llm_calls = 0
    for batch in plan.execution_batches():
        deps_per_task = {
            task_id: {dep: outputs[dep] for dep in plan.task(task_id).depends_on}
            for task_id in batch
        }
        results = await asyncio.gather(
            *(
                adapter.execute(
                    task_id,
                    plan.task(task_id).instruction,
                    plan.meta[task_id],
                    deps_per_task[task_id],
                )
                for task_id in batch
            )
        )
        for task_id, output in zip(batch, results):
            outputs[task_id] = output
            if plan.meta[task_id].kind is SubtaskKind.REASONING:
                llm_calls += 1
    return outputs, llm_calls
