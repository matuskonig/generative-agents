import abc
import asyncio
import time
from typing import (
    Any,
    Callable,
    Coroutine,
    Literal,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    overload,
)

import numpy as np
from openai import (
    APITimeoutError,
    AsyncClient,
    LengthFinishReasonError,
    Omit,
    RateLimitError,
    omit,
)
from pydantic import BaseModel

from .async_helpers import Throttler

ResponseFormatType = TypeVar("ResponseFormatType", bound="BaseModel")


class LLMBackendBase(abc.ABC):
    """Base class defining the interface for LLM backend implementations.

    This base class allows for mock implementations in tests without requiring
    the concrete LLMBackend class.
    """

    @abc.abstractmethod
    async def get_text_response(
        self,
        prompt: str,
        params: "CompletionParams" = ...,
    ) -> str:
        pass

    @abc.abstractmethod
    async def get_structured_response(
        self,
        prompt: str,
        response_format: Type[ResponseFormatType],
        params: "CompletionParams" = ...,
    ) -> ResponseFormatType:
        pass

    @overload
    async def embed_text(self, input: str) -> np.ndarray: ...
    @overload
    async def embed_text(self, input: list[str]) -> list[np.ndarray]: ...
    @abc.abstractmethod
    async def embed_text(self, input: str | list[str]) -> np.ndarray | list[np.ndarray]:
        pass


class EmbeddingProvider(abc.ABC):
    """Abstract base class for text embedding implementations.

    Provides unified interface for embedding generation with support for both
    single strings and batch processing. Implementations can use different backends
    (OpenAI API, local models like SentenceTransformer, etc.).

    The embed_text method is overloaded to accept either str or list[str],
    returning single embedding or list respectively.
    """

    @abc.abstractmethod
    async def _embed_impl(self, input: list[str]) -> list[np.ndarray]: ...

    # TODO: add length assertion here
    @overload
    async def embed_text(self, input: str) -> np.ndarray: ...
    @overload
    async def embed_text(self, input: list[str]) -> list[np.ndarray]: ...
    async def embed_text(self, input: str | list[str]) -> np.ndarray | list[np.ndarray]:
        if isinstance(input, str):
            result = await self._embed_impl([input])
            return result[0]
        else:
            return await self._embed_impl(input)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embeddings."""

    def __init__(
        self,
        client: AsyncClient,
        model: str,
    ):
        self.__client = client
        self.__model = model

    async def _embed_impl(self, input: list[str]) -> list[np.ndarray]:
        if len(input) == 0:
            return []

        response = await self.__client.embeddings.create(
            input=input, model=self.__model
        )
        return [
            np.array(embedding_response.embedding)
            for embedding_response in response.data
        ]


class SentenceTransformerProvider(EmbeddingProvider):
    """Local SentenceTransformer embeddings."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        model_kwargs: dict[str, Any] | None = None,
    ):
        try:
            from sentence_transformers import (  # type: ignore [import-not-found]
                SentenceTransformer,
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please install it with: pip install sentence-transformers or as part of optional project dependencies [embedding]"
            )
        self.__model = SentenceTransformer(
            model_name, device=device, model_kwargs=model_kwargs
        )
        self.__batch_size = batch_size

    async def _embed_impl(self, input: list[str]) -> list[np.ndarray]:
        if len(input) == 0:
            return []

        result = self.__model.encode(
            input, batch_size=self.__batch_size, convert_to_numpy=True
        )
        return list(result)


class CompletionParams(TypedDict):
    temperature: float | Omit | None
    max_tokens: int | Omit | None
    max_completion_tokens: int | Omit | None
    top_p: float | Omit | None
    frequency_penalty: float | Omit | None
    presence_penalty: float | Omit | None
    reasoning_effort: (
        Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] | Omit
    )


def create_completion_params(
    temperature: float | Omit | None = omit,
    max_completion_tokens: int | Omit | None = omit,
    max_tokens: int | Omit | None = omit,
    top_p: float | Omit | None = omit,
    frequency_penalty: float | Omit | None = omit,
    presence_penalty: float | Omit | None = omit,
    reasoning_effort: (
        Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] | Omit
    ) = omit,
) -> CompletionParams:
    return CompletionParams(
        temperature=temperature,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        reasoning_effort=reasoning_effort,
    )


def with_resiliency[**P, R](
    delay_sec: float = 1, exp_backoff: float = 1.5, max_length_retries: int = 2
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]:
    """Decorator that retries failed API calls with exponential backoff.

    Specifically handles RateLimitError, APITimeoutError, and LengthFinishReasonError from OpenAI API.
    Uses exponential backoff: delay = delay_sec * (exp_backoff ^ retry_count)
    For RateLimitError and APITimeoutError, retries indefinitely until success.
    For LengthFinishReasonError, retries up to `max_length_retries` times.

    Args:
        delay_sec: Initial delay in seconds before first retry
        exp_backoff: Multiplier for exponential backoff (1.5 = 1.5x increase per retry)
        max_length_retries: Maximum retries allowed for length finish reason errors (default: 2)
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            retry_count = 0
            finish_reason_count = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except (RateLimitError, APITimeoutError):
                    delay = delay_sec * (exp_backoff**retry_count)
                    retry_count += 1
                    await asyncio.sleep(delay)
                except LengthFinishReasonError as e:
                    if finish_reason_count < max_length_retries:
                        finish_reason_count += 1
                        continue
                    raise e

        return inner

    return decorator


class LLMBackend(LLMBackendBase):

    def __init__(
        self,
        client: AsyncClient,
        model: str,
        RPS: int | float | None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:

        from .agent import default_config

        self.__client = client
        self.__chat_model = model
        self.__throttle = Throttler(RPS) if RPS else None
        self.completion_tokens: int = 0
        self.prompt_tokens: int = 0
        self.total_time: float = 0.0
        self.total_requests = 0
        self.system_prompt = default_config().get_system_prompt()
        self.__embedding_provider = embedding_provider

    @with_resiliency(max_length_retries=0)
    async def get_text_response(
        self, prompt: str, params: CompletionParams = create_completion_params()
    ) -> str:
        if self.__throttle:
            await self.__throttle()
        start = time.time()

        response = await self.__client.chat.completions.create(
            model=self.__chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            **params,
        )
        self.total_time += time.time() - start
        self.completion_tokens += (
            response.usage.completion_tokens if response.usage else 0
        )
        self.prompt_tokens += response.usage.prompt_tokens if response.usage else 0
        self.total_requests += 1

        return response.choices[0].message.content or ""

    @with_resiliency(max_length_retries=2)
    async def get_structured_response(
        self,
        prompt: str,
        response_format: Type[ResponseFormatType],
        params: CompletionParams = create_completion_params(),
    ) -> ResponseFormatType:
        if self.__throttle:
            await self.__throttle()

        start = time.time()
        response = await self.__client.chat.completions.parse(
            model=self.__chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            **params,
        )

        self.total_time += time.time() - start
        self.completion_tokens += (
            response.usage.completion_tokens if response.usage else 0
        )
        self.prompt_tokens += response.usage.prompt_tokens if response.usage else 0
        self.total_requests += 1

        message = response.choices[0].message

        if not message.parsed:
            raise ValueError("Model refused to parse the response")
        return message.parsed

    @overload
    async def embed_text(self, input: str) -> np.ndarray: ...
    @overload
    async def embed_text(self, input: list[str]) -> list[np.ndarray]: ...

    @with_resiliency()
    async def embed_text(self, input: str | list[str]) -> np.ndarray | list[np.ndarray]:
        if not self.__embedding_provider:
            raise ValueError(
                "Embedding provider is not set, however embedding action is requested."
            )
        if self.__throttle:
            await self.__throttle()

        return await self.__embedding_provider.embed_text(input)
