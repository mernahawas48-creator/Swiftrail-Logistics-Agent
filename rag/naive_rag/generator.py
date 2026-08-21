"""Text-generation adapters shared by the Swiftrail RAG pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")


DEFAULT_MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    """Token usage reported by one model-generation call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TextGenerator(Protocol):
    """Minimal interface required by the RAG and memory pipelines."""

    def generate(self, prompt: str) -> str:
        ...


class MistralTextGenerator:
    """Generate grounded text with Mistral and retain reported token usage.

    The LangChain client is created lazily. Unit tests can inject a small
    object exposing ``invoke(prompt)`` and therefore never need a live API key.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MISTRAL_MODEL,
        client: object | None = None,
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = client
        self._usage_history: list[GenerationUsage] = []

    @property
    def client(self):
        """Create the Mistral client only when generation is requested."""

        if self._client is None:
            try:
                from langchain_mistralai import ChatMistralAI
            except ImportError as exc:
                raise RuntimeError(
                    "Mistral LangChain integration is not installed. Run: "
                    'pip install "langchain-mistralai>=1.1,<2.0"'
                ) from exc

            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise RuntimeError("MISTRAL_API_KEY is not set.")

            self._client = ChatMistralAI(
                model=self.model_name,
                api_key=api_key,
                temperature=self.temperature,
                max_retries=self.max_retries,
            )

        return self._client

    @property
    def last_usage(self) -> GenerationUsage:
        """Return usage for the most recent generation."""

        if not self._usage_history:
            return GenerationUsage()
        return self._usage_history[-1]

    @property
    def usage_totals(self) -> GenerationUsage:
        """Return cumulative usage since the last reset."""

        return GenerationUsage(
            input_tokens=sum(item.input_tokens for item in self._usage_history),
            output_tokens=sum(item.output_tokens for item in self._usage_history),
            total_tokens=sum(item.total_tokens for item in self._usage_history),
        )

    def reset_usage(self) -> None:
        """Clear accumulated usage before an evaluation run."""

        self._usage_history.clear()

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("The generation prompt cannot be empty.")

        response = self.client.invoke(normalized_prompt)
        self._usage_history.append(self._extract_usage(response))

        answer = self._extract_text(response).strip()
        if not answer:
            raise RuntimeError("Mistral returned an empty answer.")
        return answer

    @staticmethod
    def _extract_text(response: object) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

        text = getattr(response, "text", None)
        return text if isinstance(text, str) else ""

    @staticmethod
    def _extract_usage(response: object) -> GenerationUsage:
        usage: Any = getattr(response, "usage_metadata", None)
        if not isinstance(usage, dict):
            metadata = getattr(response, "response_metadata", None)
            if isinstance(metadata, dict):
                usage = metadata.get("token_usage") or metadata.get("usage")

        if not isinstance(usage, dict):
            return GenerationUsage()

        input_tokens = int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        )
        output_tokens = int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
        total_tokens = int(
            usage.get("total_tokens") or (input_tokens + output_tokens)
        )
        return GenerationUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
