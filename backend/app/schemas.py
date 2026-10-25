"""Validated API request and response models."""
from typing import Literal
from pydantic import BaseModel, Field

class DescriptorWeight(BaseModel):
    name: str
    weight: float = Field(gt=0)

class SearchRequest(BaseModel):
    dataset: Literal["objects", "mpeg7_1m"]
    keywords: list[str] = []
    keyword_mode: Literal["and", "or"] = "and"
    query_image_id: int | None = None
    descriptors: list[DescriptorWeight] = []
    k: int = Field(default=10, ge=1, le=500)

class SearchResult(BaseModel):
    image_id: int
    url: str
    distance: float
    rank: int
    descriptor_distances: dict[str, float | None] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)

class SearchResponse(BaseModel):
    query: dict
    query_image_url: str | None = None
    results: list[SearchResult]
    took_ms: float
