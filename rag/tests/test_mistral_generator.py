from types import SimpleNamespace

import pytest

from rag.naive_rag.generator import GenerationUsage, MistralTextGenerator


class StubClient:
    def __init__(self, response):
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.response


def test_mistral_generator_normalizes_text_and_tracks_usage():
    client = StubClient(
        SimpleNamespace(
            content="Grounded answer [1].",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 4,
                "total_tokens": 16,
            },
        )
    )
    generator = MistralTextGenerator(client=client)

    assert generator.generate("Use this context") == "Grounded answer [1]."
    assert client.prompts == ["Use this context"]
    assert generator.last_usage == GenerationUsage(12, 4, 16)
    assert generator.usage_totals == GenerationUsage(12, 4, 16)

    generator.reset_usage()
    assert generator.last_usage == GenerationUsage()


def test_mistral_generator_supports_block_content():
    client = StubClient(
        SimpleNamespace(
            content=[{"type": "text", "text": "Block answer"}],
            usage_metadata=None,
        )
    )
    assert MistralTextGenerator(client=client).generate("prompt") == "Block answer"


def test_mistral_generator_rejects_empty_prompt():
    generator = MistralTextGenerator(client=StubClient(None))
    with pytest.raises(ValueError, match="cannot be empty"):
        generator.generate("   ")
