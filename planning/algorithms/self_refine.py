import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ..critique_rubric import rubric_text
from ..models import EnvironmentFeedback


def deterministic_checks(goal: str, draft: str) -> list[str]:
    issues: list[str] = []
    if len(draft.split()) < 80:
        issues.append("The deliverable is under 80 words and is probably incomplete.")
    goal_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z]{5,}", goal)
        if word.lower() not in {"create", "design", "write", "build", "about", "using"}
    }
    represented = [term for term in goal_terms if term in draft.lower()]
    if goal_terms and not represented:
        issues.append("The output contains none of the goal's significant terms.")
    if not re.search(r"(^|\n)(#{1,3}\s+|\d+[.)]\s+|[-*]\s+|ACTION:)", draft):
        issues.append(
            "The deliverable has no visible structure "
            "(headings, list items, or ACTION lines)."
        )
    return issues


@dataclass
class ReflectionResult:
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]
    revision_feedback: EnvironmentFeedback | None = None


def reflect_and_refine(
    goal: str,
    draft: str,
    llm: BaseChatModel,
    *,
    critic_llm: BaseChatModel | None = None,
    environment=None,
) -> ReflectionResult:
    grounded = deterministic_checks(goal, draft)

    if environment is not None:
        grounded_feedback = environment.evaluate(draft)
        grounded.extend(grounded_feedback.details)

    grounded_report = "\n".join(
        f"- {issue}" for issue in grounded
    ) or "- Deterministic/grounded checks passed."

    critic = critic_llm or llm

    # This can be done better, how should it be done?
    critique_response = critic.invoke([
        (
            "system",
            "You are a separate critic. Judge against the rubric and external "
            "evidence; do not rewrite the draft.",
        ),
        (
            "human",
            f"""Goal: {goal}
Rubric:
{rubric_text()}

External deterministic/grounded checks:
{grounded_report}

Draft:
{draft}

List concrete issues. If there are none, respond exactly PASS.""",
        ),
    ], temperature=0.2)

    critique = critique_response.content
    if not isinstance(critique, str) or not critique.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    critique = critique.strip()

    if critique.upper() == "PASS" and not grounded:
        revised = draft
    else:
        response = llm.invoke([
            (
                "system",
                "Revise a deliverable using both external checks and an "
                "independent critique.",
            ),
            (
                "human",
                f"""Goal: {goal}

Draft:
{draft}

Grounded checks:
{grounded_report}

Critique:
{critique}

Return only the improved deliverable.""",
            ),
        ], temperature=0.2)

        revised = response.content
        if not isinstance(revised, str) or not revised.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        revised = revised.strip()

    post_feedback = (
        environment.evaluate(revised)
        if environment is not None
        else None
    )

    return ReflectionResult(
        draft,
        critique,
        revised,
        grounded,
        post_feedback,
    )