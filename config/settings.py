"""项目全局配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
RAW_DIR = DATA_DIR / "raw"
TRUSTED_DIR = DATA_DIR / "trusted"
INBOX_DIR = DATA_DIR / "inbox"

for _p in (DATA_DIR, INDEX_DIR, RAW_DIR, TRUSTED_DIR, INBOX_DIR):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class SourceConf:
    name: str
    display: str
    url: str
    priority: int = 100


SOURCE_REGISTRY: List[SourceConf] = [
    SourceConf("reuters", "Reuters", "https://www.reuters.com/", priority=98),
    SourceConf("wsj", "Wall Street Journal", "https://www.wsj.com/", priority=96),
    SourceConf("bloomberg", "Bloomberg", "https://www.bloomberg.com/", priority=97),
    SourceConf("cnbc", "CNBC", "https://www.cnbc.com/", priority=94),
    SourceConf("nytimes", "New York Times", "https://www.nytimes.com/", priority=93),
    SourceConf("guardian", "The Guardian", "https://www.theguardian.com/", priority=90),
    SourceConf("economist", "The Economist", "https://www.economist.com/", priority=92),
    SourceConf("marketwatch", "Market Watch", "https://www.marketwatch.com/", priority=91),
    SourceConf("yfinance", "Yahoo Finance", "https://finance.yahoo.com/", priority=88),
    SourceConf("forbes", "Forbes", "https://www.forbes.com/", priority=85),
    SourceConf("businessinsider", "Business Insider", "https://www.businessinsider.com/", priority=86),
    SourceConf("wired", "Wired Business", "https://www.wired.com/", priority=78),
    SourceConf("medium", "Medium Business", "https://medium.com/", priority=72),
]


@dataclass(slots=True)
class CrawlerConf:
    finurls_seed: str = "https://finurls.com/"
    max_pages: int = 3
    page_interval_sec: float = 1.5
    per_source_min_articles: int = 0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    request_timeout: int = 25
    connect_timeout: int = 10


@dataclass(slots=True)
class DedupConf:
    simhash_ham_threshold: int = 3
    embedding_dim: int = 384
    semantic_thresh: float = 0.88
    batch_size: int = 64


@dataclass(slots=True)
class VerifyConf:
    min_sources_for_trusted: int = 3
    semantic_match_thresh: float = 0.85
    source_prior: dict = field(default_factory=lambda: {s.name: s.priority for s in SOURCE_REGISTRY})


@dataclass(slots=True)
class RAGConf:
    top_k_vector: int = 6
    top_k_keyword: int = 8
    final_top_k: int = 5
    rerank_enabled: bool = True
    system_prompt: str = (
        "你是一名资深金融投研分析师，需基于给定的多信源财经资讯回答问题。"
        "回答必须：1) 仅使用提供的上下文事实，不要编造；"
        "2) 对每条结论标注来源媒体；"
        "3) 明确给出事实置信度（高/中/低）；"
        "4) 如证据不足请直接说明。"
    )


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class AppConf:
    crawler: CrawlerConf = field(default_factory=CrawlerConf)
    dedup: DedupConf = field(default_factory=DedupConf)
    verify: VerifyConf = field(default_factory=VerifyConf)
    rag: RAGConf = field(default_factory=RAGConf)
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", ""))
    index_dir: Path = INDEX_DIR
    raw_dir: Path = RAW_DIR
    trusted_dir: Path = TRUSTED_DIR
    inbox_dir: Path = INBOX_DIR


def load() -> AppConf:
    return AppConf()
