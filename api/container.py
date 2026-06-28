"""应用级容器：装配 crawler / verifier / kb / rag。"""
from __future__ import annotations

from threading import Lock
from typing import Optional

from config import AppConf, load
from crawler import FinurlsCrawler, RawArticle, crawl_and_fetch_content
from verifier import CrossSourceVerifier
from kb import Embedder, FaissKnowledgeBase
from rag import FinanceRAG


class AppContainer:
    def __init__(self, conf: Optional[AppConf] = None):
        self.conf = conf or load()
        self.embedder = Embedder()  # TF-IDF 不需要指定 dim
        self.kb = FaissKnowledgeBase(self.conf.index_dir, self.embedder)
        self.verifier = CrossSourceVerifier(
            self.embedder,
            self.kb,
            min_sources=self.conf.verify.min_sources_for_trusted,
            semantic_thresh=self.conf.verify.semantic_match_thresh,
        )
        self.crawler = FinurlsCrawler(self.conf)
        # 无 LLM 模式，直接初始化 FinanceRAG
        self.rag = FinanceRAG(self.kb, self.embedder, llm=None)
        self._crawl_lock = Lock()

    def run_crawl_pipeline(self, enrich_content: bool = False) -> dict:
        with self._crawl_lock:
            raw = self.crawler.crawl()
            if enrich_content:
                raw = crawl_and_fetch_content(raw, self.conf)
            facts, stats = self.verifier.ingest(raw)
            return {
                "raw_articles": len(raw),
                "facts": len(facts),
                "kb_stats": self.kb.stats(),
                "ingest_stats": stats,
            }


_container: Optional[AppContainer] = None


def get_container() -> AppContainer:
    global _container
    if _container is None:
        _container = AppContainer()
    return _container


def reset_container() -> AppContainer:
    global _container
    _container = AppContainer()
    return _container
