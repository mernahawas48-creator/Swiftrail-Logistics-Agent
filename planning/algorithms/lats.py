from __future__ import annotations

import math
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from ..models import EnvironmentFeedback
from .environment import Environment


class LATSAction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: str = Field(min_length=2)
    state: str = Field(min_length=2)


class LATSActionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[LATSAction] = Field(min_length=1, max_length=3)


class ValueEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)


@dataclass
class LATSNode:
    state: str
    action: str = "root"
    parent: LATSNode | None = field(default=None, repr=False)
    children: list[LATSNode] = field(default_factory=list, repr=False)
    visits: int = 0
    value_sum: float = 0.0
    environment_score: float = 0.0
    model_score: float = 0.0
    feedback: EnvironmentFeedback | None = None
    reflections: list[str] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class LATSResult:
    success: bool
    output: str
    best_score: float
    iterations: int
    root: LATSNode


LATS_ACTION_SYSTEM = """You are the action generator inside the LATS search
for the Swiftrail Logistics planning agent.

Generate distinct candidate resolution plans for one already-decomposed
planning sub-task.

Use only the supplied task, current trajectory, evidence, tool outputs,
policy constraints, and previous reflections.

Do not invent customer data, shipment state, invoice information, employee
authority, policy rules, or MCP tool results.

A candidate may recommend an MCP action, but it must never claim that the
action succeeded unless the supplied state already contains a successful
tool result.

Candidates should be safe, concrete, and meaningfully different from each
other."""


LATS_VALUE_SYSTEM = """You are the model-based value estimator inside LATS
for the Swiftrail Logistics planning agent.

The external environment feedback is the source of truth.

Use the environment score and feedback when estimating whether this branch
is useful to continue exploring. Never override or contradict grounded
environment feedback with your own assumptions.

The score represents future usefulness of the branch, not whether you
personally prefer the answer."""


LATS_REFLECTION_SYSTEM = """You create a short branch-level LATS reflection
for the Swiftrail Logistics planning agent.

The reflection must be grounded in the external environment feedback.
Explain what caused the branch to fail and what a later search branch
should change.

Do not invent a new failure reason and do not contradict the environment."""


def _uct(node: LATSNode, exploration_weight: float) -> float:
    if node.visits == 0:
        return float("inf")

    parent_visits = max(
        node.parent.visits if node.parent else 1,
        1,
    )

    return (
        node.mean_value
        + exploration_weight
        * math.sqrt(math.log(parent_visits) / node.visits)
    )


def _select_leaf(
    root: LATSNode,
    exploration_weight: float,
) -> LATSNode:
    node = root

    while node.children:
        node = max(
            node.children,
            key=lambda child: _uct(
                child,
                exploration_weight,
            ),
        )

    return node


def _backpropagate(
    node: LATSNode,
    value: float,
) -> None:
    while node is not None:
        node.visits += 1
        node.value_sum += value
        node = node.parent


def _trajectory_reflections(
    node: LATSNode,
) -> list[str]:
    path: list[str] = []

    while node is not None:
        path.extend(node.reflections)
        node = node.parent

    return list(reversed(path))


def lats(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    iterations: int = 2,
    n_actions: int = 2,
    exploration_weight: float = 1.414,
) -> LATSResult:

    task = task.strip()

    if not task:
        raise ValueError("task cannot be empty")

    if iterations < 1:
        raise ValueError("iterations must be positive")

    if n_actions < 1 or n_actions > 3:
        raise ValueError(
            "n_actions must be between 1 and 3"
        )

    if exploration_weight < 0:
        raise ValueError(
            "exploration_weight cannot be negative"
        )

    root = LATSNode(
        state="No candidate resolution has been attempted yet."
    )

    best = root
    completed_iterations = 0

    for iteration in range(1, iterations + 1):

        completed_iterations = iteration

        # SELECT
        leaf = _select_leaf(
            root,
            exploration_weight,
        )

        lessons = _trajectory_reflections(leaf)

        lesson_text = (
            "\n".join(
                f"- {item}"
                for item in lessons[-4:]
            )
            or "- None yet."
        )

        # EXPAND
        proposed = llm.with_structured_output(
            LATSActionBatch,
            method="json_schema",
        ).invoke(
            [
                (
                    "system",
                    LATS_ACTION_SYSTEM,
                ),
                (
                    "human",
                    f"""Planning sub-task and available context:
{task}

Current trajectory/state:
{leaf.state}

Reflections learned from failed branches:
{lesson_text}

Propose exactly {n_actions} distinct candidate resolution plans.

For every candidate:
- action: briefly name the strategy being attempted.
- state: provide the complete proposed resolution for this sub-task.

Respect all supplied dependencies, authority restrictions, policies,
and observed tool results.

Do not claim that an action has already been successfully executed unless
that success appears in the supplied context.""",
                ),
            ],
            temperature=0.5,
        )

        for item in proposed.actions[:n_actions]:

            child = LATSNode(
                state=item.state.strip(),
                action=item.action.strip(),
                parent=leaf,
            )

            leaf.children.append(child)

            # EVALUATE USING EXTERNAL / GROUNDED FEEDBACK
            feedback = environment.evaluate(
                child.state
            )

            child.feedback = feedback
            child.environment_score = feedback.score

            # MODEL VALUE ESTIMATE
            value_judgment = llm.with_structured_output(
                ValueEstimate,
                method="json_schema",
            ).invoke(
                [
                    (
                        "system",
                        LATS_VALUE_SYSTEM,
                    ),
                    (
                        "human",
                        f"""Planning sub-task:
{task}

Candidate resolution:
{child.state}

Grounded environment score:
{feedback.score}

Grounded environment feedback:
{feedback.details}

Estimate how useful this branch is for future search.
Treat the external feedback as authoritative.""",
                    ),
                ],
                temperature=0.1,
            )

            child.model_score = (
                value_judgment.score
            )

            # Environment feedback remains the dominant signal.
            combined_value = (
                0.75 * child.environment_score
                + 0.25 * child.model_score
            )

            # REFLECT ON FAILED BRANCH
            if not feedback.success:

                response = llm.invoke(
                    [
                        (
                            "system",
                            LATS_REFLECTION_SYSTEM,
                        ),
                        (
                            "human",
                            f"""Planning sub-task:
{task}

Attempted strategy:
{child.action}

Candidate resolution:
{child.state}

Grounded environment feedback:
{feedback.details}

Write one short reflection explaining:
1. why this branch failed according to the environment,
2. what the next branch should change.""",
                        ),
                    ],
                    temperature=0.2,
                )

                reflection = response.content

                if (
                    not isinstance(
                        reflection,
                        str,
                    )
                    or not reflection.strip()
                ):
                    raise RuntimeError(
                        "The chat model returned an empty "
                        "or unsupported reflection"
                    )

                child.reflections.append(
                    reflection.strip()
                )

            # BACKPROPAGATE
            _backpropagate(
                child,
                combined_value,
            )

            # Keep the externally strongest candidate.
            if (
                best is root
                or child.environment_score
                > best.environment_score
            ):
                best = child

            # Stop immediately when the real environment
            # confirms a successful candidate.
            if feedback.success:
                return LATSResult(
                    success=True,
                    output=child.state,
                    best_score=child.environment_score,
                    iterations=completed_iterations,
                    root=root,
                )

    return LATSResult(
        success=False,
        output=best.state,
        best_score=best.environment_score,
        iterations=completed_iterations,
        root=root,
    )


def flatten_lats_tree(
    root: LATSNode,
) -> list[dict]:

    records: list[dict] = []

    queue: list[
        tuple[LATSNode, str | None]
    ] = [
        (root, None)
    ]

    next_id = 0

    while queue:

        node, parent_id = queue.pop(0)

        node_id = f"n{next_id}"
        next_id += 1

        records.append(
            {
                "id": node_id,
                "parent_id": parent_id,
                "action": node.action,
                "state": node.state,
                "visits": node.visits,
                "mean_value": node.mean_value,
                "environment_score": (
                    node.environment_score
                ),
                "model_score": (
                    node.model_score
                ),
                "feedback": (
                    node.feedback.model_dump()
                    if node.feedback
                    else None
                ),
                "reflections": node.reflections,
            }
        )

        queue.extend(
            (child, node_id)
            for child in node.children
        )

    return records