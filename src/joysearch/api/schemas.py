from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    catalog_id: Literal["wands_search"] = "wands_search"
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    algorithm: Literal["bm25", "dense", "hybrid"] = "bm25"
    debug: bool = False

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value.strip()


class RecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    catalog_id: Literal["amazon_baby_rec"] = "amazon_baby_rec"
    user_id: str | None = Field(default=None, max_length=150)
    top_k: int = Field(default=10, ge=1, le=100)
    algorithm: Literal["popularity", "itemcf"] = "itemcf"
    exclude_seen: bool = True


class SearchResult(BaseModel):
    catalog_id: Literal["wands_search"]
    item_id: str
    title: str
    score: float
    rank: int
    sources: list[str]
    metadata: dict[str, str]
    evidence: dict | None = None


class SearchResponse(BaseModel):
    catalog_id: Literal["wands_search"]
    algorithm: Literal["bm25", "dense", "hybrid"]
    model_version: str
    results: list[SearchResult]
    degraded: bool
    trace_id: str
    latency_ms: float
