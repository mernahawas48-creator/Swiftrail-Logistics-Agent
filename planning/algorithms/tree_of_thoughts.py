from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from ..models import Thought


class ThoughtCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[str] = Field(min_length=1, max_length=3)


class ThoughtEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


THOUGHT_GENERATION_SYSTEM = """You are the Tree-of-Thoughts candidate generator for the Swiftrail Logistics planning agent.
You receive one already-decomposed planning sub-task and its available context.
Generate genuinely different next reasoning paths; do not produce paraphrases of the same idea.
Use only the supplied facts, policy text, tool outputs, and constraints.
Do not invent shipment state, customer data, employee authority, policy rules, or MCP results.
Do not claim that an MCP action succeeded unless the supplied context contains a successful tool result."""


THOUGHT_EVALUATION_SYSTEM = """You are the model-based evaluator used inside Tree of Thoughts for the Swiftrail Logistics planning agent.
Evaluate the candidate against the supplied problem and evidence.
This is an ungrounded model score, so do not pretend that you executed tools or checked the database.
Reward correctness, feasibility, constraint/authority compliance, and useful progress.
Penalize unsupported assumptions, policy violations, stale paths, and actions outside the current employee's authority.
Do not reward confidence or writing style."""


def tree_of_thoughts(
    problem: str,
    llm: BaseChatModel,
    depth: int = 2,
    beam_width: int = 2,
) -> list[Thought]:
    """Search several reasoning paths with breadth-wise beam pruning.

    The LLM generates and self-evaluates candidate paths. The best
    ``beam_width`` candidates survive each level. These scores are deliberately
    model-based; grounded validation belongs to LATS / EnvironmentFeedback.
    """
    problem = problem.strip()
    if not problem:
        raise ValueError("problem cannot be empty")
    if depth < 1:
        raise ValueError("depth must be positive")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")

    frontier = [Thought(state="Start", score=0.5, rationale="root")]

    for _ in range(depth):
        candidates: list[Thought] = []

        for parent in frontier:
            generated = llm.with_structured_output(
                ThoughtCandidates,
                method="json_schema",
            ).invoke(
                [
                    ("system", THOUGHT_GENERATION_SYSTEM),
                    (
                        "human",
                        f"""Planning sub-task and available context:
{problem}

Current partial reasoning path:
{parent.state}

Propose two distinct promising continuations.
Each candidate must be a complete updated partial path: preserve any relevant
prior decisions and add one concrete next reasoning step.
Prefer candidates that make meaningful progress toward a safe, executable
Swiftrail resolution.""",
                    ),
                ],
                temperature=0.5,
            )

            # Avoid spending evaluation calls on duplicate branches.
            unique_states: list[str] = []
            seen: set[str] = set()

            for raw_state in generated.candidates[:2]:
                state = raw_state.strip()

                if not state or state in seen:
                    continue

                seen.add(state)
                unique_states.append(state)

            for state in unique_states:
                judged = llm.with_structured_output(
                    ThoughtEvaluation,
                    method="json_schema",
                ).invoke(
                    [
                        ("system", THOUGHT_EVALUATION_SYSTEM),
                        (
                            "human",
                            f"""Planning sub-task and available context:
{problem}

Candidate reasoning path:
{state}

Score this candidate from 0.0 to 1.0 based on:
- correctness against the supplied evidence,
- feasibility with the stated tools and current state,
- compliance with policy, dependencies, and employee authority,
- useful progress toward resolving the sub-task,
- absence of invented or unsupported facts.

Return a concise rationale for the score.""",
                        ),
                    ],
                    temperature=0.1,
                )

                candidates.append(
                    Thought(
                        state=state,
                        score=judged.score,
                        rationale=judged.rationale,
                    )
                )

        # Breadth-wise beam search: prune weaker branches after each level.
        frontier = sorted(
            candidates,
            key=lambda item: item.score,
            reverse=True,
        )[:beam_width]

        if not frontier:
            break

    return frontier