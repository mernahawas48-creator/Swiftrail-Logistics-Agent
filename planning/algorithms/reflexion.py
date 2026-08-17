from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ..episodic_buffer import EpisodicReflectionBuffer
from ..models import EnvironmentFeedback
from .environment import Environment


@dataclass
class ReflexionTrial:
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None


@dataclass
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]


def reflexion(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    max_trials: int = 3,
    memory_size: int = 3,
    *,
    critic_llm: BaseChatModel | None = None,
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
            (
                "system",
                "You are the acting agent in a Reflexion loop. Attempt the entire "
                "task again. For Swiftrail shipment exception tasks, produce a "
                "structured plan that respects observed evidence, employee authority, "
                "financial safety, and required escalations.",
            ),
            (
                "human",
                f"""Task: {task}
Episodic memory from previous failed trials:
{recalled}

For Swiftrail resolution plans, use explicit ACTION lines when actions are needed, for example:
ACTION: check_shipment
ACTION: check_customer
ACTION: check_invoices
ACTION: check_credit_hold
ACTION: check_rate_exception
ACTION: escalate role=finance_manager
ACTION: release_credit_hold hold_id=2
ACTION: approve_rate_exception exception_id=2
ACTION: release_shipment

Produce the complete deliverable. Apply remembered lessons without discussing them.""",
            ),
        ], temperature=0.2)

        attempt = response.content
        if not isinstance(attempt, str) or not attempt.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        attempt = attempt.strip()

        feedback = environment.evaluate(attempt)
        trial = ReflexionTrial(number=number, attempt=attempt, feedback=feedback)

        if feedback.score > best_score:
            best_attempt, best_score = attempt, feedback.score

        if feedback.success:
            trials.append(trial)
            return ReflexionResult(True, attempt, trials, buffer.items())

        response = critic.invoke([
            (
                "system",
                "Generate a concise first-person Reflexion memory, not a revised "
                "answer. Base the lesson on the grounded external environment feedback.",
            ),
            (
                "human",
                f"""Task: {task}
Failed attempt:
{attempt}

External environment feedback (score {feedback.score}):
{chr(10).join('- ' + item for item in feedback.details)}

State what I did wrong and the specific strategy I should use next trial. Start with 'I'.""",
            ),
        ], temperature=0.2)

        reflection = response.content
        if not isinstance(reflection, str) or not reflection.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        reflection = reflection.strip()

        trial.reflection = reflection
        trials.append(trial)
        buffer.add(reflection)

    return ReflexionResult(False, best_attempt, trials, buffer.items())