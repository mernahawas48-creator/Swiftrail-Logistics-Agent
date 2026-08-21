from types import SimpleNamespace

import pytest

from context_eval.strategies.recursive_summarization import (
    RecursiveSummarization,
)


class StubGenerator:
    def __init__(self):
        self.prompts: list[str] = []
        self.last_usage = SimpleNamespace(output_tokens=7)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Customer has a standing 12% loyalty discount."


def test_recursive_summarization_calls_generator_and_keeps_recent_messages():
    generator = StubGenerator()
    messages = [
        {"role": "employee", "content": "Standing discount is 12%."},
        {"role": "tool", "content": "routine lookup"},
        {"role": "employee", "content": "Check shipment 501."},
    ]

    result = RecursiveSummarization(
        keep_last_messages=1,
        generator=generator,
    ).apply(messages)

    assert len(generator.prompts) == 1
    assert "Standing discount is 12%." in generator.prompts[0]
    assert "standing 12% loyalty discount" in result[0]["content"]
    assert result[1] == messages[-1]


def test_recursive_summarization_skips_model_for_short_context():
    generator = StubGenerator()
    messages = [{"role": "employee", "content": "hello"}]

    result = RecursiveSummarization(
        keep_last_messages=2,
        generator=generator,
    ).apply(messages)

    assert result == messages
    assert generator.prompts == []


def test_recursive_summarization_rejects_invalid_window():
    with pytest.raises(ValueError, match="at least 1"):
        RecursiveSummarization(keep_last_messages=0, generator=StubGenerator())
