from openai import AsyncClient
from typing import TypeVar, Type
from pydantic import BaseModel
from .async_helpers import Throtler


ResponseFormatType = TypeVar("ResponseFormatType", bound="BaseModel")

SYSTEM_PROMPT = """You are a helpful assistant used in a society simulation. You will be given a persona you are supposed to act as.  
Please respond to the propmts directly, using your given persona, without adding any text not related to the prompt. 
"""


class LLMBackend:
    def __init__(
        self, client: AsyncClient, model: str, RPS: int | float | None
    ) -> None:
        self.__client = client
        self.__chat_model = model
        self.__throttle = Throtler(RPS) if RPS else None

    async def get_text_response(self, prompt: str):
        if self.__throttle:
            await self.__throttle()

        response = await self.__client.chat.completions.create(
            model=self.__chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

    async def get_structued_response(
        self, prompt: str, response_format: Type[ResponseFormatType]
    ) -> ResponseFormatType | str | None:
        if self.__throttle:
            await self.__throttle()

        response = await self.__client.beta.chat.completions.parse(
            model=self.__chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format=response_format,
            n=1,
        )
        message = response.choices[0].message
        return message.parsed or message.refusal
