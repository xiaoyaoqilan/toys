"""领域数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RawArticle:
    url: str
    title: str
    source: str
    source_display: str
    published_at: Optional[datetime] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    content: str = ""
    summary: str = ""
    raw_html: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        import hashlib
        return hashlib.sha1(self.url.encode("utf-8")).hexdigest()


@dataclass
class NormalizedFact:
    fact_id: str
    canonical_title: str
    canonical_summary: str
    embedding: List[float] = field(default_factory=list)
    simhash: int = 0
    entities: Dict[str, List[str]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    source_count: int = 0
    confidence: str = "low"
    score: float = 0.0
    verified: bool = False
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    article_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class VerificationResult:
    fact_id: str
    is_trusted: bool
    matched_sources: List[str]
    similarity_scores: Dict[str, float]
    evidence_articles: List[str]
    confidence_level: str
    notes: str = ""


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    trace: List[Dict[str, Any]] = field(default_factory=list)
