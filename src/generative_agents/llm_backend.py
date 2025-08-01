from openai import AsyncClient, RateLimitError, APITimeoutError, NotGiven, NOT_GIVEN
from typing import TypeVar, Type, overload, Callable, Awaitable, TypedDict
from pydantic import BaseModel
import time
from .async_helpers import Throttler
import numpy as np
from dataclasses import dataclass

ResponseFormatType = TypeVar("ResponseFormatType", bound="BaseModel")


class CompletionParams(TypedDict):
    temperature: float | NotGiven | None
    max_completion_tokens: int | NotGiven | None
    top_p: float | NotGiven | None
    frequency_penalty: float | NotGiven | None
    presence_penalty: float | NotGiven | None


def create_completion_params(
    temperature: float | NotGiven | None = NOT_GIVEN,
    max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
    top_p: float | NotGiven | None = NOT_GIVEN,
    frequency_penalty: float | NotGiven | None = NOT_GIVEN,
    presence_penalty: float | NotGiven | None = NOT_GIVEN,
) -> CompletionParams:
    return CompletionParams(
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )


def rate_limit_repeated[**P, R](func: Callable[P, Awaitable[R]]):
    async def inner(*args: P.args, **kwargs: P.kwargs):
        while True:
            try:
                return await func(*args, **kwargs)
            except (RateLimitError, APITimeoutError) as e:
                continue

    return inner


class LLMBackend:

    def __init__(
        self,
        client: AsyncClient,
        model: str,
        RPS: int | float | None,
        embedding_model: str | None = None,
    ) -> None:

        from .agent import default_config

        self.__client = client
        self.__chat_model = model
        self.__embedding_model = embedding_model
        self.__throttle = Throttler(RPS) if RPS else None
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_time = 0
        self.total_requests = 0
        self.system_prompt = default_config().get_system_prompt()

    @rate_limit_repeated
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

    @rate_limit_repeated
    async def get_structued_response(
        self,
        prompt: str,
        response_format: Type[ResponseFormatType],
        params: CompletionParams = create_completion_params(),
    ) -> ResponseFormatType:
        if self.__throttle:
            await self.__throttle()

        start = time.time()

        response = await self.__client.beta.chat.completions.parse(
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
    @rate_limit_repeated
    async def embed_text(self, input: str | list[str]):
        if not self.__embedding_model:
            raise ValueError(
                "Embedding model is not set, however embedding action is requested."
            )
        if self.__throttle:
            await self.__throttle()

        response = await self.__client.embeddings.create(
            input=input,
            model=self.__embedding_model,
        )

        if type(input) == str:
            return np.array(response.data[0].embedding)

        return [
            np.array(embedding_response.embedding)
            for embedding_response in response.data
        ]
