"""Swiftrail-specific sub-task metadata.

DAG construction, dependency mapping, acyclicity enforcement, and
topological/execution-batch ordering already live in ``planning.models.Plan``
(forked from the reference toolkit) and are reused as-is here -- see
``Plan.validate_dag`` (cycle check at construction time) and
``Plan.execution_batches`` (topological execution). This module does not
re-implement any of that.

What the toolkit's ``Task`` does not know about is *what kind of sub-task*
a node represents in our system: a deterministic, single MCP tool call
(``search_customer``, ``get_shipment_status``, ...) versus a sub-task that
genuinely needs reasoning/search and must be routed to Plan-and-Solve /
Tree of Thoughts / LATS (owned by planning/planning_router.py). That
distinction is Swiftrail-specific, so it's added here as metadata attached
to a toolkit ``Task`` by id, rather than by forking ``Task``/``Plan``
themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .models import Plan


class SubtaskKind(str, Enum):
    # A single, deterministic call into an existing MCP tool. No LLM
    # reasoning is involved in *producing* the result -- the tool's return
    # value *is* the sub-task's output. This is the "Logical/Deterministic
    # Task" branch in the lab's routing chart.
    TOOL_CALL = "tool_call"
    # A sub-task that requires weighing evidence, considering branches, or
    # producing a written decision/justification. Routed to
    # planning_router.solve_subtask (Plan-and-Solve / ToT / LATS).
    REASONING = "reasoning"


# Builds the concrete MCP tool arguments for one sub-task given the session
# id and the grounded outputs already produced by its dependencies. Kept as
# a plain callable (not a template string) so a builder can read structured
# facts out of an earlier tool's JSON result, e.g. "release the hold whose
# customer_id matches the shipment we already fetched".
ArgBuilder = Callable[[str, dict[str, str]], dict]


@dataclass(frozen=True)
class SubtaskMeta:
    kind: SubtaskKind
    # Populated when kind == TOOL_CALL.
    tool_name: str | None = None
    build_args: ArgBuilder | None = None
    # Populated when kind == REASONING; forwarded into a PlanningProfile.
    needs_branching: bool = False
    high_stakes: bool = False
    grounded_validation_available: bool = False


@dataclass
class SwiftrailPlan:
    """A toolkit ``Plan`` plus the Swiftrail routing metadata for each task.

    Composition, not subclassing: pydantic's ``Plan.tasks: list[Task]`` is
    not something we can safely widen in a subclass, and we don't need to --
    every DAG/acyclicity/ordering concern is answered by ``self.plan``.
    """

    plan: Plan
    meta: dict[str, SubtaskMeta] = field(default_factory=dict)

    def __post_init__(self) -> None:
        known = {task.id for task in self.plan.tasks}
        missing = known - set(self.meta)
        if missing:
            raise ValueError(
                f"Tasks missing Swiftrail routing metadata: {sorted(missing)}"
            )

    # Thin pass-throughs so callers mostly interact with one object.
    def execution_batches(self) -> list[list[str]]:
        return self.plan.execution_batches()

    def task(self, task_id: str):
        return self.plan.task(task_id)

    def terminal_tasks(self) -> list[str]:
        return self.plan.terminal_tasks()
