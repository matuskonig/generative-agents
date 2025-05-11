from openai import AsyncClient, RateLimitError
from typing import TypeVar, Type
from pydantic import BaseModel
import time
from .async_helpers import Throttler


ResponseFormatType = TypeVar("ResponseFormatType", bound="BaseModel")

SYSTEM_PROMPT = """You are an agent in a society simulation. You will be given a persona you are supposed to act as."""


class LLMBackend:
    def __init__(
        self,
        client: AsyncClient,
        model: str,
        temperature: float,
        RPS: int | float | None,
    ) -> None:
        self.__client = client
        self.__chat_model = model
        self.__temperature = temperature
        self.__throttle = Throttler(RPS) if RPS else None
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_time = 0

    async def get_text_response(self, prompt: str):
        while True:
            try:
                return await self.__get_text_response_impl(prompt)
            except RateLimitError as e:
                continue

    async def __get_text_response_impl(self, prompt: str):
        if self.__throttle:
            await self.__throttle()

        start = time.time()
        response = await self.__client.chat.completions.create(
            model=self.__chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.__temperature,
        )
        self.total_time += time.time() - start
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens

        return response.choices[0].message.content

    async def get_structued_response(
        self, prompt: str, response_format: Type[ResponseFormatType]
    ) -> ResponseFormatType | str | None:
        while True:
            try:
                return await self.__get_structued_response_impl(prompt, response_format)
            except RateLimitError as e:
                continue

    async def __get_structued_response_impl(
        self, prompt: str, response_format: Type[ResponseFormatType]
    ) -> ResponseFormatType | str | None:
        if self.__throttle:
            await self.__throttle()

        start = time.time()
        response = await self.__client.beta.chat.completions.parse(
            model=self.__chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.__temperature,
            response_format=response_format,
        )
        self.total_time += time.time() - start
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens

        message = response.choices[0].message
        return message.parsed or message.refusal
