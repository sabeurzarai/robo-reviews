"""API request and response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewRecord(BaseModel):
    name: str
    reviews_text: str = Field(alias="reviews.text")
    reviews_rating: float = Field(alias="reviews.rating", ge=1, le=5)
    categories: str

    model_config = {"populate_by_name": True}


class UploadResponse(BaseModel):
    rows_loaded: int
    message: str


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1)


class SentimentResponse(BaseModel):
    sentiment: str


class ClusterRequest(BaseModel):
    records: list[ReviewRecord]
    k: int | None = Field(default=None, ge=1, le=6)


class ClusterResponse(BaseModel):
    records_clustered: int
    categories_found: int
    insights: list[dict]


class SummaryRequest(BaseModel):
    category_insight: dict


class SummaryResponse(BaseModel):
    article: str
