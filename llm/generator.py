from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

SYSTEM_PROMPT = (
    "You are a financial document assistant."
    " Use ONLY the provided context."
    " If the context is insufficient, say \"I don't know\"."
)


@dataclass
class OpenAIConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_output_tokens: int = 512


class OpenAIGenerator:
    def __init__(self, config: OpenAIConfig | None = None) -> None:
        self.config = config or OpenAIConfig()
        env_model = os.getenv("OPENAI_MODEL")
        env_temp = os.getenv("OPENAI_TEMPERATURE")
        env_max_tokens = os.getenv("OPENAI_MAX_TOKENS")
        if env_model:
            self.config.model = env_model
        if env_temp:
            try:
                self.config.temperature = float(env_temp)
            except ValueError:
                pass
        if env_max_tokens:
            try:
                self.config.max_output_tokens = int(env_max_tokens)
            except ValueError:
                pass
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide it to use the OpenAI generator."
            )
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required for OpenAI generator.") from exc

        self.client = OpenAI()

    def generate(self, context: str, question: str) -> str:
        if not context.strip():
            return "I don't know."

        prompt = (
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "INSTRUCTION:\n"
            "Answer clearly using ONLY the context. If not enough info, say \"I don't know\"."
        )

        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.config.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
            return (response.output_text or "").strip()

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        return (response.choices[0].message.content or "").strip()


@dataclass
class LocalHFConfig:
    model: str = "google/flan-t5-base"
    max_new_tokens: int = 256


class LocalHFGenerator:
    def __init__(self, config: LocalHFConfig | None = None) -> None:
        self.config = config or LocalHFConfig()
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for LocalHFGenerator. Install it or use OpenAI."
            ) from exc

        self.pipe = pipeline(
            "text2text-generation",
            model=self.config.model,
            tokenizer=self.config.model,
        )

    def generate(self, context: str, question: str) -> str:
        if not context.strip():
            return "I don't know."

        prompt = (
            "Answer the question using only the context. "
            "If the context is insufficient, say \"I don't know\".\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n"
        )

        result = self.pipe(prompt, max_new_tokens=self.config.max_new_tokens)[0]["generated_text"]
        return result.strip()


def build_generator(name: str = "openai"):
    name = (name or "openai").lower()
    if name == "openai":
        return OpenAIGenerator()
    if name in {"local", "hf", "transformers"}:
        return LocalHFGenerator()
    raise ValueError(f"Unknown generator '{name}'. Use 'openai' or 'local'.")
