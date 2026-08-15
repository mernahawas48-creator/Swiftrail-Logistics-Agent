from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .episodic_buffer import EpisodicReflectionBuffer
from .models import EnvironmentFeedback


@dataclass(slots=True)
class ReflexionTrial:
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None


@dataclass(slots=True)
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]


def reflexion(
    task: str,
    llm: Any,
    environment: Any,
    max_trials: int = 3,
    memory_size: int = 3,
    *,
    critic_llm: Any | None = None,
) -> ReflexionResult:
    if max_trials < 1 or memory_size < 1:
        raise ValueError("max_trials and memory_size must be positive")
    buffer = EpisodicReflectionBuffer(memory_size)
    trials: list[ReflexionTrial] = []
    best_attempt = ""
    best_score = -1.0
    critic = critic_llm or llm

    for number in range(1, max_trials + 1):
        recalled = buffer.prompt_text()
        response = llm.invoke([
            ("system", "You are the acting agent in a Reflexion loop for Swiftrail Shipment Exception Resolution. Produce the complete structured plan. Apply remembered lessons without discussing them."),
            ("human", f"""Task: {task}

Episodic memory from previous failed trials:
{recalled}

Use explicit ACTION lines, for example:
ACTION: check_shipment
ACTION: check_customer
ACTION: check_invoices
ACTION: check_credit_hold
ACTION: check_rate_exception
ACTION: escalate role=finance_manager
ACTION: release_credit_hold hold_id=2
ACTION: approve_rate_exception exception_id=2
ACTION: release_shipment

Return the complete plan only."""),
        ], temperature=0.2)
        attempt = str(response.content).strip()
        if not attempt:
            raise RuntimeError("The acting model returned an empty response")

        feedback = environment.evaluate(attempt)
        trial = ReflexionTrial(number, attempt, feedback)
        if feedback.score > best_score:
            best_attempt, best_score = attempt, feedback.score
        if feedback.success:
            trials.append(trial)
            return ReflexionResult(True, attempt, trials, buffer.items())

        reflection_response = critic.invoke([
            ("system", "Generate a concise first-person Reflexion memory, not a revised answer. Base it on external environment evidence."),
            ("human", f"""Task: {task}
Failed attempt:
{attempt}

Grounded environment feedback (score {feedback.score}):
{chr(10).join('- ' + item for item in feedback.details)}

State what I did wrong and the specific strategy I should use next trial. Start with 'I'."""),
        ], temperature=0.2)
        reflection = str(reflection_response.content).strip()
        if not reflection:
            raise RuntimeError("The reflection model returned an empty response")
        trial.reflection = reflection
        trials.append(trial)
        buffer.add(reflection)

    return ReflexionResult(False, best_attempt, trials, buffer.items())
