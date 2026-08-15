from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .critique_rubric import rubric_text
from .models import EnvironmentFeedback


@dataclass(slots=True)
class ReflectionResult:
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]
    revision_feedback: EnvironmentFeedback | None = None


def deterministic_checks(goal: str, draft: str) -> list[str]:
    """Cheap structural checks retained from the reference toolkit and extended for Swiftrail."""
    issues: list[str] = []
    if len(draft.split()) < 30:
        issues.append("The deliverable is under 30 words and is probably incomplete.")
    goal_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z]{5,}", goal)
        if word.lower() not in {"create", "design", "write", "build", "about", "using"}
    }
    represented = [term for term in goal_terms if term in draft.lower()]
    if goal_terms and not represented:
        issues.append("The output contains none of the goal's significant terms.")
    if not re.search(r"(^|\n)(#{1,3}\s+|\d+[.)]\s+|[-*]\s+|ACTION:)", draft):
        issues.append("The deliverable has no visible structure (headings, list items, or ACTION lines).")
    return issues


def reflect_and_refine(
    goal: str,
    draft: str,
    llm: Any,
    *,
    critic_llm: Any | None = None,
    environment: Any | None = None,
) -> ReflectionResult:
    """Reference Self-Refine flow: one draft -> independent critique -> one revision.

    The optional environment is the grounded source of truth. The optional critic_llm
    makes the critic independent from the acting model.
    """
    grounded: list[str] = deterministic_checks(goal, draft)
    grounded_feedback = None
    if environment is not None:
        grounded_feedback = environment.evaluate(draft)
        grounded.extend(grounded_feedback.details)
    grounded_report = "\n".join(f"- {issue}" for issue in grounded) or "- Deterministic/grounded checks passed."

    critic = critic_llm or llm
    critique_response = critic.invoke([
        ("system", "You are an independent critic. Do not rewrite the draft. Evaluate it against the rubric and external evidence."),
        ("human", f"""Goal: {goal}

Rubric:
{rubric_text()}

External checks:
{grounded_report}

Draft:
{draft}

List concrete issues. If there are none, respond exactly PASS."""),
    ], temperature=0.2)
    critique = str(critique_response.content).strip()
    if not critique:
        raise RuntimeError("The critic returned an empty response")

    if critique.upper() == "PASS" and not grounded:
        revised = draft
    else:
        response = llm.invoke([
            ("system", "Revise the Swiftrail deliverable using the external checks and independent critique. Return only the improved deliverable."),
            ("human", f"Goal: {goal}\n\nDraft:\n{draft}\n\nGrounded checks:\n{grounded_report}\n\nIndependent critique:\n{critique}"),
        ], temperature=0.2)
        revised = str(response.content).strip()
        if not revised:
            raise RuntimeError("The revision model returned an empty response")

    post_feedback = environment.evaluate(revised) if environment is not None else None
    return ReflectionResult(draft, critique, revised, grounded, post_feedback)
