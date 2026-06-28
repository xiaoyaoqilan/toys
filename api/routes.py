"""FastAPI 路由。"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .container import get_container


router = APIRouter()


class CrawlRequest(BaseModel):
    pages: int = Field(default=3, ge=1, le=10, description="finurls 爬取页数")
    enrich_content: bool = Field(default=False, description="是否抓取原文正文")


class CrawlResponse(BaseModel):
    ok: bool
    data: Dict[str, Any]


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceItem(BaseModel):
    fact_id: str
    canonical_title: str
    canonical_summary: str
    sources: List[str]
    confidence: str
    score: float


class AnswerResponse(BaseModel):
    ok: bool
    answer: str
    confidence: str
    sources: List[SourceItem]
    trace: List[Dict[str, Any]]
    elapsed_ms: int


class StatsResponse(BaseModel):
    ok: bool
    kb_stats: Dict[str, Any]
    llm_ready: bool


class FactListResponse(BaseModel):
    ok: bool
    items: List[Dict[str, Any]]
    total: int


@router.get("/health", response_model=Dict[str, Any])
def health() -> Dict[str, Any]:
    c = get_container()
    return {
        "status": "ok",
        "embedding_backend": c.embedder.backend,
        "llm_ready": False,  # No LLM mode
        "kb_stats": c.kb.stats(),
    }


@router.post("/crawl", response_model=CrawlResponse)
def crawl(req: CrawlRequest) -> CrawlResponse:
    c = get_container()
    c.conf.crawler.max_pages = req.pages
    try:
        t0 = time.time()
        result = c.run_crawl_pipeline(enrich_content=req.enrich_content)
        result["elapsed_s"] = round(time.time() - t0, 2)
        return CrawlResponse(ok=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa", response_model=AnswerResponse)
def qa(req: QuestionRequest) -> AnswerResponse:
    c = get_container()
    t0 = time.time()
    result = c.rag.answer(req.question, top_k=req.top_k)
    elapsed = int((time.time() - t0) * 1000)
    sources = [SourceItem(**s) for s in result.sources]
    return AnswerResponse(
        ok=True,
        answer=result.answer,
        confidence=result.confidence,
        sources=sources,
        trace=result.trace,
        elapsed_ms=elapsed,
    )


@router.get("/facts", response_model=FactListResponse)
def list_facts(
    confidence: Optional[str] = None,
    verified_only: bool = False,
    limit: int = 100,
) -> FactListResponse:
    c = get_container()
    facts = c.kb.all_trusted() if verified_only else list(c.kb._facts.values())
    if confidence:
        facts = [f for f in facts if f.confidence == confidence]
    items = [f.to_dict() for f in facts[:limit]]
    return FactListResponse(ok=True, items=items, total=len(items))


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    c = get_container()
    return StatsResponse(
        ok=True,
        kb_stats=c.kb.stats(),
        llm_ready=False,  # No LLM mode
    )
