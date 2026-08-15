from __future__ import annotations

from typing import Any

from .critique_rubric import rubric_text


def evaluate_with_independent_critic(goal: str, draft: str, critic_llm: Any, grounded_feedback: str) -> str:
    response = critic_llm.invoke([
        ("system", "You are an independent quality critic. Do not rewrite the answer. Identify concrete violations only."),
        ("human", f"Goal:\n{goal}\n\nRubric:\n{rubric_text()}\n\nGrounded evidence:\n{grounded_feedback}\n\nCandidate:\n{draft}\n\nReturn PASS or a concise list of issues."),
    ], temperature=0.0)
    return str(response.content).strip()
