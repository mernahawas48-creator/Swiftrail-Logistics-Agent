from typing import Any

from rag.naive_rag.generator import MistralTextGenerator, TextGenerator

from .base import ContextStrategy


class RecursiveSummarization(ContextStrategy):
    """Replace older messages with a model-generated compact summary.

    Needs a real TextGenerator (defaults to MistralTextGenerator, which
    reads MISTRAL_API_KEY from .env). Unlike the previous implementation,
    this actually asks a model to compress old_messages -- it can lose
    detail and it spends real output tokens, which is the whole point
    of the strategy and the tradeoff the comparison table needs to show
    honestly.
    """

    def __init__(
        self,
        keep_last_messages: int = 5,
        generator: TextGenerator | None = None,
    ):
        if keep_last_messages < 1:
            raise ValueError("keep_last_messages must be at least 1.")
        self.keep_last_messages = keep_last_messages
        self.generator = generator or MistralTextGenerator()

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) <= self.keep_last_messages:
            return messages

        old_messages = messages[:-self.keep_last_messages]
        recent_messages = messages[-self.keep_last_messages:]

        transcript = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in old_messages
        )

        prompt = (
            "Summarize the following conversation turns into a short "
            "paragraph. Preserve every operational detail: customer "
            "commitments, disputes, discounts, credit holds, and "
            "shipment/invoice facts. Drop only small talk and routine "
            "status chatter.\n\n"
            f"{transcript}"
        )

        summary_text = self.generator.generate(prompt)

        summary_message = {
            "role": "system",
            "content": f"Summary of older context: {summary_text}",
        }

        return [summary_message] + recent_messages
