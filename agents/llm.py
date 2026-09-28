"""Single entry point for all LLM calls.

Agents receive an object with a `structured()` method, never the OpenAI
client directly. Real runs pass OpenAILLM; tests pass FakeLLM.
"""

import os
import time
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Estimated USD price per million tokens. Prices change: override in .env.
INPUT_PRICE_PER_M = float(os.getenv("LLM_INPUT_PRICE_PER_M", "0.20"))
OUTPUT_PRICE_PER_M = float(os.getenv("LLM_OUTPUT_PRICE_PER_M", "1.20"))


class LLMError(Exception):
    """Raised when the LLM cannot produce a valid structured answer."""


@dataclass
class LLMResult(Generic[T]):
    parsed: T
    input_tokens: int
    output_tokens: int
    latency_ms: int
    calls: int  # includes any retry

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_PRICE_PER_M + self.output_tokens * OUTPUT_PRICE_PER_M
        ) / 1_000_000


class LLM(Protocol):
    """Anything with this method can be used as an LLM by the agents."""

    def structured(self, system: str, user: str, schema: type[T]) -> LLMResult[T]: ...


class OpenAILLM:
    def __init__(self, model: str | None = None, timeout: float = 30.0):
        self.model = model or os.getenv("OPENAI_MODEL")
        if not self.model:
            raise LLMError("OPENAI_MODEL is not set in .env")
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY is not set in .env")
        # The SDK itself retries network errors, rate limits and server errors.
        self.client = OpenAI(timeout=timeout, max_retries=2)

    def structured(
        self, system: str, user: str, schema: type[T], attempts: int = 2
    ) -> LLMResult[T]:
        start = time.perf_counter()
        input_tokens = output_tokens = calls = 0
        last_error: Exception | None = None

        # Our own retry is for bad or missing parsed output.
        for _ in range(attempts):
            calls += 1
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    text_format=schema,
                )
            except (OpenAIError, ValidationError) as error:
                last_error = error
                continue

            if response.usage:
                input_tokens += response.usage.input_tokens
                output_tokens += response.usage.output_tokens

            parsed = response.output_parsed
            if parsed is None:
                last_error = LLMError("No parsed output (the model may have refused).")
                continue

            return LLMResult(
                parsed=parsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=int((time.perf_counter() - start) * 1000),
                calls=calls,
            )

        raise LLMError(f"LLM call failed after {attempts} attempts: {last_error}")


class FakeLLM:
    """Test double: returns pre-set answers in order and records every call.

    An item in `responses` can be an Exception, to simulate a failure.
    """

    def __init__(self, responses: list[BaseModel | Exception]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def structured(self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if not self.responses:
            raise LLMError("FakeLLM has no responses left.")
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if not isinstance(answer, schema):
            raise LLMError(f"FakeLLM expected {schema.__name__}, got {type(answer).__name__}.")
        return LLMResult(parsed=answer, input_tokens=0, output_tokens=0, latency_ms=0, calls=1)