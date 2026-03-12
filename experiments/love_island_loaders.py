from typing import Literal, Optional

import pydantic

from generative_agents import AgentModelBase


class LoveIslandPerson(AgentModelBase):
    id: str
    url: str
    name: str
    description: str
    age: Optional[str] = None
    zodiac: Optional[str] = None
    height: Optional[str] = None
    image_url: str
    socials: dict[str, str]
    shows: list[str]
    sex: Literal["M", "F"]
    location: str
    job: Optional[str] = None
    origin: Optional[str] = None
    relationship_count: str | None
    hobbies: str | None
    personality_traits: str | None
    partner_preferences: str | None

    @property
    def full_name(self) -> str:
        return self.name


dataset_adapter = pydantic.TypeAdapter(list[LoveIslandPerson])


class LoveIslandResult(pydantic.BaseModel):
    final_couples: list[tuple[str, str]]


def load_persons(language: Literal["en", "cz"]) -> list[LoveIslandPerson]:
    with open(f"../data/love_island/persons_{language}.json", "r") as f:
        return dataset_adapter.validate_json(f.read())


def load_results():
    with open(f"../data/love_island/results.json", "r") as f:
        return LoveIslandResult.model_validate_json(f.read())
