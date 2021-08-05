from typing import Optional, List, Union, Any

from pydantic import BaseModel, Field


class NerTextRequest(BaseModel):  # Need this to receive json data
    """
    Used for requests to /entities/text endpoint
    """
    text: str = Field(description="The text to find named entities in.")
    include_entities: None = Field(None, description="A list of which entity groups to include in the result.")
    group_entities: bool = Field(False, description="Whether or not to group entities by type in the result.")


class NerEntityResponse(BaseModel):
    """
    A single entity, returned by
    """
    entity_group: Optional[str] = Field(description="The predicted entity type for this entity.")
    score: float = Field(ge=0., le=1., description="A confidence score for the prediction.")
    entity: str = Field(description="The entity.")
    start: int = Field(ge=0, description="The starting index of this entity in the text.")
    end: int = Field(ge=0, description="The ending index of this entity in the text.")


class NerGroupedEntitiesResponse(BaseModel):
    entity_group: str = Field(description="The entity group.")
    items: List[NerEntityResponse] = Field(description="A list of predicted entities with this group.")


class NerUrnRequest(BaseModel):
    urn: str = Field(description="The URN for this document.")
    include_entities: None = Field(None, description="A list of which entity groups to include in the result.")
    group_entities: bool = Field(False, description="Whether or not to group entities by type in the result.")


NerResponse = Union[List[NerEntityResponse], List[NerGroupedEntitiesResponse]]

if __name__ == '__main__':
    print(NerTextRequest.schema())
    print(NerEntityResponse.schema())
    print(NerGroupedEntitiesResponse.schema())
    print(NerUrnRequest.schema())
