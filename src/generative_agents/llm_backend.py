from openai import AsyncClient, RateLimitError
from typing import TypeVar, Type, overload, Callable, Awaitable
from pydantic import BaseModel
import time
from .async_helpers import Throttler
import numpy as np


ResponseFormatType = TypeVar("ResponseFormatType", bound="BaseModel")


def rate_limit_repeated[**P, R](func: Callable[P, Awaitable[R]]):
    async def inner(*args: P.args, **kwargs: P.kwargs):
        while True:
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                continue

    return inner


class LLMBackend:

    def __init__(
        self,
        client: AsyncClient,
        model: str,
        temperature: float,
        RPS: int | float | None,
        embedding_model: str | None = None,
    ) -> None:

        from .agent import default_builder

        self.__client = client
        self.__chat_model = model
        self.__embedding_model = embedding_model
        self.__temperature = temperature
        self.__throttle = Throttler(RPS) if RPS else None
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_time = 0
        self.total_requests = 0
        self.system_prompt = default_builder().get_system_prompt()

    @rate_limit_repeated
    async def get_text_response(self, prompt: str):
        if self.__throttle:
            await self.__throttle()

        start = time.time()
        response = await self.__client.chat.completions.create(
            model=self.__chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.__temperature,
        )
        self.total_time += time.time() - start
        self.completion_tokens += (
            response.usage.completion_tokens if response.usage else 0
        )
        self.prompt_tokens += response.usage.prompt_tokens if response.usage else 0
        self.total_requests += 1

        return response.choices[0].message.content

    @rate_limit_repeated
    async def get_structued_response(
        self, prompt: str, response_format: Type[ResponseFormatType]
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
            temperature=self.__temperature,
            response_format=response_format,
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
            model=self.__embedding_model,
            input=input,
        )

        if type(input) == str:
            return np.array(response.data[0].embedding)

        return [
            np.array(embedding_response.embedding)
            for embedding_response in response.data
        ]
