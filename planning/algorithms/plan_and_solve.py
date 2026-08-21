from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

PLAN_AND_SOLVE_SYSTEM = """You are the Plan-and-Solve reasoner for the Swiftrail Logistics planning agent.
You solve one already-decomposed planning sub-task at a time.
Use only the facts, policy text, tool outputs, and constraints supplied in the question.
Do not invent customer data, shipment state, employee authority, policy rules, or MCP tool results.
Clearly separate PLAN from SOLUTION.
If required evidence is missing, state what is missing instead of assuming it."""


def plan_and_solve(question: str, llm: BaseChatModel) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")

    response = llm.invoke(
        [
            ("system", PLAN_AND_SOLVE_SYSTEM),
            (
                "human",
                f"""Sub-task and available context:
{question}

First make a short explicit plan for this sub-task. Then carry out that plan step by step.
Respect every authority, policy, dependency, and tool-result constraint present in the context.
You may recommend a later MCP action, but never claim that an action succeeded unless a tool result says it did.
Check the final conclusion against the supplied evidence.""",
            ),
        ],
        temperature=0.2,
    )

    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError(
            "The chat model returned an empty or unsupported response"
        )

    return response.content.strip()