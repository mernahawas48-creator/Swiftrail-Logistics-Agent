from __future__ import annotations

from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from .algorithms import lats, plan_and_solve, tree_of_thoughts
from .algorithms.environment import Environment


class PlanningMethod(str, Enum):
    PLAN_AND_SOLVE = "plan_and_solve"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    LATS = "lats"


class PlanningProfile(BaseModel):
    """Describe the reasoning shape of one already-decomposed sub-task."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=5)
    context: str = ""
    needs_branching: bool = False
    high_stakes: bool = False
    grounded_validation_available: bool = False

    def prompt(self) -> str:
        if self.context.strip():
            return (
                f"Sub-task:\n{self.instruction.strip()}\n\n"
                f"Available context:\n{self.context.strip()}"
            )
        return self.instruction.strip()
        

class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: PlanningMethod
    rationale: str


class RoutedPlanningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: PlanningMethod
    output: str
    routing_rationale: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    success: bool | None = None


def route_subtask(profile: PlanningProfile) -> RoutingDecision:
    """Choose the planning algorithm from the sub-task's reasoning shape.

    Routing policy:
    - LATS: branching + high stakes + a real grounded validator.
    - Tree of Thoughts: branching/lookahead without the full LATS conditions.
    - Plan-and-Solve: linear reasoning with no genuine search requirement.

    The router is deterministic so routing itself does not add another LLM
    call and can be evaluated reproducibly.
    """

    if (
        profile.needs_branching
        and profile.high_stakes
        and profile.grounded_validation_available
    ):
        return RoutingDecision(
            method=PlanningMethod.LATS,
            rationale=(
                "The sub-task has multiple plausible branches, a costly wrong "
                "choice, and grounded external feedback is available, so LATS "
                "can search alternatives and validate them against the real "
                "environment."
            ),
        )

    if profile.needs_branching:
        return RoutingDecision(
            method=PlanningMethod.TREE_OF_THOUGHTS,
            rationale=(
                "The sub-task needs lookahead across multiple plausible "
                "reasoning paths, but it does not require the full grounded "
                "MCTS loop, so Tree of Thoughts is appropriate."
            ),
        )

    return RoutingDecision(
        method=PlanningMethod.PLAN_AND_SOLVE,
        rationale=(
            "The sub-task is primarily linear reasoning with no genuine need "
            "to search competing branches, so Plan-and-Solve is the simplest "
            "suitable method."
        ),
    )


def solve_subtask(
    profile: PlanningProfile,
    llm: BaseChatModel,
    environment: Environment | None = None,
    *,
    tot_depth: int = 2,
    tot_beam_width: int = 2,
    lats_iterations: int = 2,
    lats_n_actions: int = 2,
    lats_exploration_weight: float = 1.414,
) -> RoutedPlanningResult:
    """Route one sub-task and execute the selected planning algorithm."""

    decision = route_subtask(profile)
    problem = profile.prompt()

    if decision.method == PlanningMethod.PLAN_AND_SOLVE:
        output = plan_and_solve(problem, llm)
        return RoutedPlanningResult(
            method=decision.method,
            output=output,
            routing_rationale=decision.rationale,
        )

    if decision.method == PlanningMethod.TREE_OF_THOUGHTS:
        thoughts = tree_of_thoughts(
            problem,
            llm,
            depth=tot_depth,
            beam_width=tot_beam_width,
        )

        if not thoughts:
            raise RuntimeError("Tree of Thoughts returned no candidate paths")

        best = max(thoughts, key=lambda thought: thought.score)

        return RoutedPlanningResult(
            method=decision.method,
            output=best.state,
            routing_rationale=decision.rationale,
            score=best.score,
        )

    if environment is None:
        raise ValueError(
            "LATS routing requires a grounded Environment implementation"
        )

    result = lats(
        problem,
        llm,
        environment,
        iterations=lats_iterations,
        n_actions=lats_n_actions,
        exploration_weight=lats_exploration_weight,
    )

    return RoutedPlanningResult(
        method=decision.method,
        output=result.output,
        routing_rationale=decision.rationale,
        score=result.best_score,
        success=result.success,
    )