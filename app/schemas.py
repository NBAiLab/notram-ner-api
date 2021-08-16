from typing import Optional, List, Union

from pydantic import BaseModel, Field


class NerRequest(BaseModel):
    include_entities: Optional[List[str]] = Field(None,
                                                  description="A list of which entity groups to include in the result.")
    group_entities: bool = Field(False, description="Whether or not to group entities by type in the result.")

    wait: bool = True


class NerTextRequest(NerRequest):  # Need this to receive json data
    """
    Used for requests to /entities/text endpoint
    """
    text: str = Field(description="The text to find named entities in.")

    class Config:
        schema_extra = {
            "example": {
                "text": "Per Egil Kummervold og Javier de la Rosa har laget en Colab Notebook mens de begge var "
                        "ansatt ved Nasjonalbiblioteket i Mo i Rana og i Oslo.",
                "group_entities": False,
                "wait": True
            }
        }


class NerUrnRequest(NerRequest):
    """
    Used for requests to /entities/urn endpoint
    """
    urn: str = Field(description="The URN for this document.")

    class Config:
        schema_extra = {
            "example": {
                "urn": "digibok_2015110307521.jsonl",
                "group_entities": False,
                "wait": False
            }
        }


class NerUrlRequest(NerRequest):
    """
    Used for requests to /entities/url endpoint
    """
    url: str = Field(description="The URL for some webpage.")

    class Config:
        schema_extra = {
            "example": {
                "url": "https://www.nb.no/",
                "group_entities": False,
                "wait": True
            }
        }


class NerEntityResult(BaseModel):
    """
    A single entity.
    """
    entity_group: str = Field(description="The predicted entity type for this entity.")
    score: float = Field(ge=0., le=1., description="A confidence score for the prediction.")
    word: str = Field(description="The entity.")
    start: int = Field(ge=0, description="The starting index of this entity in the text.")
    end: int = Field(ge=0, description="The ending index of this entity in the text.")


class NerGroupedEntitiesResult(BaseModel):
    """
    A group of entities
    """
    entity_group: str = Field(description="The entity group.")
    items: List[NerEntityResult] = Field(description="A list of predicted entities with this group.")


NerResult = Union[List[NerEntityResult], List[NerGroupedEntitiesResult]]


class NerResponse:
    status: str = Field(
        description="The status of the task, PENDING for pending/invalid tasks and SUCCESS for completed tasks.")
    uuid: str = Field(description="A unique identifier for the task. Can be used to check status of the task, "
                                  "or to re-retrieve results provided the task is not removed from the queue.")
    result: NerResult = Field(description="The result of the computation.")
