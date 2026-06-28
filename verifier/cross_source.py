"""跨媒体事实交叉校验。"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from loguru import logger

from crawler.models import NormalizedFact, RawArticle, VerificationResult
from dedup import SimHash, find_clusters, canonical_title, EntityExtractor
from kb import Embedder, FaissKnowledgeBase


class CrossSourceVerifier:
    def __init__(self, embedder: Embedder, kb: FaissKnowledgeBase, min_sources: int = 3, semantic_thresh: float = 0.85):
        self.embedder = embedder
        self.kb = kb
        self.min_sources = min_sources
        self.semantic_thresh = semantic_thresh
        self.simhash = SimHash()
        self.entity_extractor = EntityExtractor()

    # ---------- 去重 & 归一 ----------
    def normalize_articles(self, articles: Sequence[RawArticle]) -> List[NormalizedFact]:
        if not articles:
            return []

        texts = [self._text_of(a) for a in articles]
        
        # 关键修复：先拟合 TF-IDF 模型到这批文章的语料上
        self.embedder.fit(texts)
        
        embeddings = self.embedder.encode(texts)
        simhashes = [self.simhash.compute(t) for t in texts]

        clusters = find_clusters(
            embeddings,
            simhashes,
            simhash_thresh=3,
            semantic_thresh=0.88,
        )
        logger.info(f"clustered {len(articles)} articles -> {len(clusters)} facts")

        facts: List[NormalizedFact] = []
        entity_cache: Dict[int, Dict] = {}

        for cluster in clusters:
            idxs = list(cluster)
            src_articles = [articles[i] for i in idxs]
            texts_in_cluster = [texts[i] for i in idxs]

            title = canonical_title([a.title for a in src_articles]) or (
                src_articles[0].title if src_articles else ""
            )
            summary = self._build_summary(src_articles)
            all_text = title + " " + summary

            emb = self.embedder.encode_one(all_text).tolist()

            entities = self.entity_extractor.extract(all_text).to_dict()
            sources = sorted({a.source for a in src_articles})
            source_count = len(sources)

            if source_count >= self.min_sources:
                confidence = "high"
                verified = True
            elif source_count == 2:
                confidence = "medium"
                verified = False
            else:
                confidence = "low"
                verified = False

            score = self._score(sources, confidence)

            fact_id = self._make_fact_id(title, summary)

            facts.append(
                NormalizedFact(
                    fact_id=fact_id,
                    canonical_title=title,
                    canonical_summary=summary,
                    embedding=emb,
                    simhash=simhashes[idxs[0]],
                    entities=entities,
                    tags=entities.get("tags", []),
                    sources=sources,
                    source_count=source_count,
                    confidence=confidence,
                    score=score,
                    verified=verified,
                    first_seen=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    article_ids=[a.uid for a in src_articles],
                )
            )
        return facts

    @staticmethod
    def _text_of(a: RawArticle) -> str:
        text = a.content or a.title or ""
        return text

    @staticmethod
    def _build_summary(articles: Sequence[RawArticle]) -> str:
        texts = []
        for a in articles:
            if a.content:
                texts.append(a.content[:300])
            elif a.title:
                texts.append(a.title)
        if not texts:
            return ""
        joined = " ".join(texts)
        joined = re.sub(r"\s+", " ", joined).strip()
        return joined[:600]

    @staticmethod
    def _make_fact_id(title: str, summary: str) -> str:
        h = hashlib.sha1()
        h.update(title.lower().strip().encode("utf-8"))
        h.update(b"\x00")
        h.update(summary[:200].lower().strip().encode("utf-8"))
        return h.hexdigest()[:16]

    @staticmethod
    def _score(sources: List[str], confidence: str) -> float:
        base = {"high": 0.9, "medium": 0.65, "low": 0.35}.get(confidence, 0.2)
        bonus = min(len(sources), 6) * 0.02
        return min(1.0, base + bonus)

    # ---------- 交叉校验已有事实 ----------
    def verify_fact(self, fact: NormalizedFact, candidates: Sequence[RawArticle]) -> VerificationResult:
        q = fact.canonical_title + " " + fact.canonical_summary
        q_emb = self.embedder.encode_one(q)

        matched: List[Tuple[RawArticle, float]] = []
        for c in candidates:
            c_text = self._text_of(c)
            if not c_text:
                continue
            c_emb = self.embedder.encode_one(c_text)
            sim = float(q_emb.dot(c_emb))
            if sim >= self.semantic_thresh:
                matched.append((c, sim))

        matched.sort(key=lambda x: -x[1])
        sources_seen = {c.source for c, _ in matched}
        is_trusted = len(sources_seen) >= self.min_sources

        confidence = "high" if is_trusted else ("medium" if len(sources_seen) >= 2 else "low")

        return VerificationResult(
            fact_id=fact.fact_id,
            is_trusted=is_trusted,
            matched_sources=sorted(sources_seen),
            similarity_scores={c.url: round(s, 4) for c, s in matched},
            evidence_articles=[c.uid for c, _ in matched],
            confidence_level=confidence,
            notes=f"matched {len(matched)} articles across {len(sources_seen)} sources",
        )

    # ---------- 完整管道 ----------
    def ingest(self, articles: Sequence[RawArticle]) -> Tuple[List[NormalizedFact], Dict]:
        facts = self.normalize_articles(articles)
        new_facts = []
        stats = {
            "total_articles": len(articles),
            "unique_facts": len(facts),
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
        }
        for fact in facts:
            existing = self.kb.get(fact.fact_id)
            if existing and existing.sources:
                merged_sources = sorted(set(existing.sources) | set(fact.sources))
                if len(merged_sources) >= self.min_sources and fact.confidence != "high":
                    fact.confidence = "high"
                    fact.verified = True
            self.kb.upsert_fact(fact)
            new_facts.append(fact)
            if fact.confidence == "high":
                stats["high_confidence"] += 1
            elif fact.confidence == "medium":
                stats["medium_confidence"] += 1
            else:
                stats["low_confidence"] += 1
        return new_facts, stats
