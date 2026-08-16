from __future__ import annotations

import asyncio

import pytest

from .divergence import compute_divergence
from .models import Plan, Task
from .swiftrail_subtask import SubtaskKind, SubtaskMeta, SwiftrailPlan
from .algorithms.dynamic_decomposition import DynamicStep


def _tool_meta(tool_name: str) -> SubtaskMeta:
    return SubtaskMeta(kind=SubtaskKind.TOOL_CALL, tool_name=tool_name, build_args=lambda sid, outs: {})


def _reasoning_meta() -> SubtaskMeta:
    return SubtaskMeta(kind=SubtaskKind.REASONING, needs_branching=True, high_stakes=True, grounded_validation_available=True)


def _static_plan() -> SwiftrailPlan:
    """A minimal but realistic version of decompose_blocked_shipment's shape:
    three independent tool_call lookups, then one terminal reasoning node."""
    plan = Plan.model_validate(
        {
            "goal": "Review shipment 500 and customer 3.",
            "tasks": [
                {"id": "t1", "instruction": "fetch shipment", "depends_on": []},
                {"id": "t2", "instruction": "fetch customer", "depends_on": []},
                {"id": "t3", "instruction": "fetch credit holds", "depends_on": []},
                {"id": "t4", "instruction": "propose resolution", "depends_on": ["t1", "t2", "t3"]},
            ],
        }
    )
    meta = {
        "t1": _tool_meta("get_shipment_status"),
        "t2": _tool_meta("search_customer"),
        "t3": _tool_meta("list_customer_credit_holds"),
        "t4": _reasoning_meta(),
    }
    return SwiftrailPlan(plan=plan, meta=meta)


# --- Acyclicity / DAG construction (reused from the toolkit's Plan) --------


def test_cycle_is_rejected_at_construction_time():
    with pytest.raises(Exception):
        Plan.model_validate(
            {
                "goal": "cyclic",
                "tasks": [
                    {"id": "a", "instruction": "does something", "depends_on": ["b"]},
                    {"id": "b", "instruction": "does something else", "depends_on": ["a"]},
                ],
            }
        )


def test_swiftrail_plan_requires_metadata_for_every_task():
    plan = Plan.model_validate(
        {
            "goal": "goal text",
            "tasks": [{"id": "a", "instruction": "does something concrete", "depends_on": []}],
        }
    )
    with pytest.raises(ValueError):
        SwiftrailPlan(plan=plan, meta={})  # missing metadata for task "a"


# --- Topological execution --------------------------------------------------


def test_execution_batches_put_independent_lookups_before_the_synthesis_task():
    swiftrail_plan = _static_plan()
    batches = swiftrail_plan.execution_batches()
    assert batches[0] == ["t1", "t2", "t3"]
    assert batches[-1] == ["t4"]


# --- Static vs dynamic divergence -------------------------------------------


def test_forced_escalation_after_two_lookups_is_detected_as_divergence():
    swiftrail_plan = _static_plan()  # would run t1, t2, t3 lookups then synthesize
    dynamic_steps = [
        DynamicStep(step=1, kind=SubtaskKind.TOOL_CALL, tool_name="get_shipment_status", instruction=None, output="{}", raw={}),
        DynamicStep(step=2, kind=SubtaskKind.TOOL_CALL, tool_name="list_customer_credit_holds", instruction=None, output="{}", raw={}),
        DynamicStep(
            step=3,
            kind=SubtaskKind.REASONING,
            tool_name=None,
            instruction="escalate",
            output="Escalating to finance manager.",
            forced=True,
        ),
    ]
    report = compute_divergence(swiftrail_plan, dynamic_steps)
    assert report.diverged is True
    # Static's topological order is [get_shipment_status, search_customer,
    # list_customer_credit_holds] (t1, t2, t3). Dynamic already made 2 tool
    # calls (in a different order: it skipped straight to the hold lookup)
    # before the forced escalation fired, so the comparison is anchored at
    # index 2 -- the static plan's *third* planned lookup, which dynamic
    # decomposition never reached at all.
    assert report.static_tool_sequence == [
        "get_shipment_status",
        "search_customer",
        "list_customer_credit_holds",
    ]
    assert report.point.static_next == "list_customer_credit_holds"
    assert report.point.dynamic_next == "reasoning:escalate"


def test_no_divergence_when_dynamic_follows_the_same_lookups_and_finishes():
    swiftrail_plan = _static_plan()
    dynamic_steps = [
        DynamicStep(step=1, kind=SubtaskKind.TOOL_CALL, tool_name="get_shipment_status", instruction=None, output="{}", raw={}),
        DynamicStep(step=2, kind=SubtaskKind.TOOL_CALL, tool_name="search_customer", instruction=None, output="{}", raw={}),
        DynamicStep(step=3, kind=SubtaskKind.TOOL_CALL, tool_name="list_customer_credit_holds", instruction=None, output="{}", raw={}),
        DynamicStep(step=4, kind=SubtaskKind.REASONING, tool_name=None, instruction="synthesize", output="Resolution.", forced=False),
    ]
    report = compute_divergence(swiftrail_plan, dynamic_steps)
    assert report.diverged is False


# --- execute_plan_swiftrail respects dependency-only context ---------------


def test_execute_plan_swiftrail_passes_only_declared_dependencies():
    from .algorithms.decomposition import execute_plan_swiftrail

    swiftrail_plan = _static_plan()
    seen_context: dict[str, dict] = {}

    class FakeAdapter:
        async def execute(self, task_id, instruction, meta, outputs_so_far):
            seen_context[task_id] = dict(outputs_so_far)
            return f"output-of-{task_id}"

    outputs, llm_calls = asyncio.run(execute_plan_swiftrail(swiftrail_plan, FakeAdapter()))

    assert outputs["t4"] == "output-of-t4"
    # t4 depends on t1, t2, t3 -- it must see exactly those three outputs,
    # not an ambient dump of everything ever produced.
    assert set(seen_context["t4"].keys()) == {"t1", "t2", "t3"}
    assert seen_context["t1"] == {}  # t1 has no dependencies
    assert llm_calls == 1  # only t4 is a reasoning sub-task
