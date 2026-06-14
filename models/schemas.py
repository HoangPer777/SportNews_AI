from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ArticleSchema(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    id: Optional[int] = None
    title: str
    content: str
    source: str = Field(..., max_length=100)
    url: str
    published_at: datetime
    category: str
    created_at: Optional[datetime] = None
    embedding: Optional[list[float]] = None  # cached embedding from DB


class PlanSchema(BaseModel):
    date_range: str
    sub_goals: list[str]
    corpus_summary: str


class HighlightedNewsItem(BaseModel):
    headline: str
    summary: str
    source: str
    url: str


class ReportSchema(BaseModel):
    executive_summary: str
    trending_keywords: list[str]
    highlighted_news: list[HighlightedNewsItem]
    generated_at: datetime


class GenerateReportRequest(BaseModel):
    period_type: Literal["daily", "weekly"] = "daily"
    lookback_days: Optional[int] = None


class ReportMetadata(BaseModel):
    period_type: str
    period_start: date
    period_end: date
    lookback_days: int
    source_count: int = 0
    ranked_count: int = 0
    stage: Optional[str] = None


class ReportResponse(BaseModel):
    status: str
    report: Optional[ReportSchema] = None
    metadata: Optional[ReportMetadata] = None
    error: Optional[str] = None


class ReportState(TypedDict):
    articles: list[ArticleSchema]
    plan: PlanSchema
    retrieved_articles: list[ArticleSchema]
    ranked_articles: list[ArticleSchema]
    report: Optional[ReportSchema]
    metadata: Optional[ReportMetadata]
    review_status: str
    rewrite_count: int
    error: Optional[str]
