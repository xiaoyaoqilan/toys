"""Faiss 向量索引封装 + 金融知识库管理（强化版）。"""
from __future__ import annotations

import json
import os
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from .embeddings import Embedder
from crawler.models import NormalizedFact


class FaissKnowledgeBase:
    INDEX_FILE = "finance_facts.index"
    META_FILE = "finance_facts_meta.pkl"

    def __init__(self, index_dir: str | Path, embedder: Embedder):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._index: Optional[faiss.Index] = None
        self._facts: Dict[str, NormalizedFact] = {}
        self._id_order: List[str] = []
        self._lock = threading.RLock()
        self._load_or_init()

    # ---------- 持久化 ----------
    def _index_path(self) -> Path:
        return self.index_dir / self.INDEX_FILE

    def _meta_path(self) -> Path:
        return self.index_dir / self.META_FILE

    def _load_or_init(self) -> None:
        dim = self.embedder.dim
        if self._index_path().exists() and self._meta_path().exists():
            try:
                self._index = faiss.read_index(str(self._index_path()))
                with self._meta_path().open("rb") as f:
                    payload = pickle.load(f)
                self._facts = payload.get("facts", {})
                self._id_order = payload.get("order", [])
                
                if self._facts:
                    corpus = [f.canonical_title + " " + f.canonical_summary for f in self._facts.values()]
                    self.embedder.fit(corpus)
                    logger.info(f"Embedder fitted on {len(corpus)} facts")
                
                if self._index.d != dim:
                    logger.warning(f"index dim {self._index.d} != embedder dim {dim}, rebuilding")
                    self._init_new(dim)
                    self._rebuild()
                
                logger.info(f"loaded index with {len(self._facts)} facts")
                return
            except Exception as e:
                logger.error(f"load index failed, will reset: {e}")
        self._init_new(dim)

    def _init_new(self, dim: int) -> None:
        self._index = faiss.IndexFlatIP(dim)
        logger.info(f"Initialized new Faiss index (dim={dim})")

    # ---------- 核心写入逻辑 ----------
    def _compute_embeddings(self, facts: List[NormalizedFact]) -> np.ndarray:
        """批量计算事实的嵌入向量。"""
        texts = [f.canonical_title + " " + f.canonical_summary for f in facts]
        embeddings = self.embedder.encode(texts)
        return np.asarray(embeddings, dtype=np.float32)

    def _ensure_embedder_fitted(self) -> None:
        """确保 Embedder 已被训练。"""
        if self._facts and not self.embedder._is_fitted:
            corpus = [f.canonical_title + " " + f.canonical_summary for f in self._facts.values()]
            self.embedder.fit(corpus)
            logger.info(f"Fitted embedder on {len(corpus)} facts")

    def _rebuild(self) -> None:
        """重建索引（强化版：自动计算嵌入、处理维度不匹配）。"""
        if not self._id_order:
            self._init_new(self.embedder.dim)
            return
        
        try:
            self._ensure_embedder_fitted()
            
            facts_list = [self._facts[fid] for fid in self._id_order]
            vecs = self._compute_embeddings(facts_list)
            
            if vecs.shape[1] != self.embedder.dim:
                logger.warning(f"Vector dim {vecs.shape[1]} != expected {self.embedder.dim}, fixing")
                if vecs.shape[1] > self.embedder.dim:
                    vecs = vecs[:, :self.embedder.dim]
                else:
                    padded = np.zeros((vecs.shape[0], self.embedder.dim), dtype=np.float32)
                    padded[:, :vecs.shape[1]] = vecs
                    vecs = padded
            
            faiss.normalize_L2(vecs)
            
            self._init_new(self.embedder.dim)
            self._index.add(vecs)
            
            for i, fid in enumerate(self._id_order):
                self._facts[fid].embedding = vecs[i].tolist()
            
            logger.info(f"Rebuilt index with {len(self._id_order)} vectors")
            
        except Exception as e:
            logger.error(f"Rebuild failed: {e}")
            self._init_new(self.embedder.dim)

    # ---------- 写入接口 ----------
    def upsert_fact(self, fact: NormalizedFact) -> None:
        with self._lock:
            if fact.fact_id in self._facts:
                existing = self._facts[fact.fact_id]
                if fact.article_ids:
                    existing.article_ids = sorted(set(existing.article_ids) | set(fact.article_ids))
                if fact.sources:
                    existing.sources = sorted(set(existing.sources) | set(fact.sources))
                existing.source_count = len(existing.sources)
                existing.last_updated = datetime.utcnow()
                existing.verified = fact.verified or existing.verified
                existing.confidence = fact.confidence or existing.confidence
                existing.score = max(existing.score, fact.score)
                if fact.entities:
                    merged = dict(existing.entities or {})
                    for k, v in fact.entities.items():
                        merged.setdefault(k, [])
                        merged[k] = sorted(set(merged[k]) | set(v))
                    existing.entities = merged
                self._facts[fact.fact_id] = existing
            else:
                self._facts[fact.fact_id] = fact
                self._id_order.append(fact.fact_id)
            self._save()
            self._rebuild()

    def upsert_batch(self, facts: List[NormalizedFact]) -> None:
        """批量插入（优化版：一次 rebuild）。"""
        with self._lock:
            for fact in facts:
                if fact.fact_id in self._facts:
                    existing = self._facts[fact.fact_id]
                    if fact.sources:
                        existing.sources = sorted(set(existing.sources) | set(fact.sources))
                    existing.source_count = len(existing.sources)
                    existing.last_updated = datetime.utcnow()
                    existing.score = max(existing.score, fact.score)
                    self._facts[fact.fact_id] = existing
                else:
                    self._facts[fact.fact_id] = fact
                    self._id_order.append(fact.fact_id)
            logger.info(f"Batch inserted {len(facts)} facts, order: {len(self._id_order)}")
            self._save()
            self._rebuild()
            logger.info(f"Batch rebuild done, total: {len(self._facts)} facts")

    def delete(self, fact_id: str) -> bool:
        """删除指定事实。"""
        with self._lock:
            if fact_id in self._facts:
                del self._facts[fact_id]
                self._id_order.remove(fact_id)
                self._save()
                self._rebuild()
                logger.info(f"Deleted fact: {fact_id}")
                return True
            return False

    def clear(self) -> None:
        """清空所有数据。"""
        with self._lock:
            self._facts = {}
            self._id_order = []
            self._init_new(self.embedder.dim)
            self._save()
            logger.info("Cleared all facts")

    def _save(self) -> None:
        if self._index is not None and self._id_order:
            faiss.write_index(self._index, str(self._index_path()))
        with self._meta_path().open("wb") as f:
            pickle.dump({"facts": self._facts, "order": self._id_order}, f)

    # ---------- 检索接口 ----------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_confidence: Optional[str] = None,
    ) -> List[Tuple[NormalizedFact, float]]:
        """向量检索（Dense Retrieval）。"""
        with self._lock:
            if not self._id_order:
                return []
            
            self._ensure_embedder_fitted()
            
            try:
                q = self.embedder.encode_one(query).reshape(1, -1)
                faiss.normalize_L2(q)
                
                if q.shape[1] != self.embedder.dim:
                    q = q[:, :self.embedder.dim] if q.shape[1] > self.embedder.dim else np.pad(q, ((0,0),(0,self.embedder.dim-q.shape[1])))
                
                k = min(top_k, len(self._id_order))
                scores, idxs = self._index.search(q, k)
                
                results: List[Tuple[NormalizedFact, float]] = []
                for score, idx in zip(scores[0], idxs[0]):
                    if idx < 0 or idx >= len(self._id_order):
                        continue
                    fact_id = self._id_order[idx]
                    fact = self._facts.get(fact_id)
                    if fact is None:
                        continue
                    if filter_confidence and fact.confidence != filter_confidence:
                        continue
                    results.append((fact, float(score)))
                return results
                
            except Exception as e:
                logger.error(f"Vector search failed: {e}, falling back to keyword")
                return self.keyword_search(query, top_k)

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[NormalizedFact, float]]:
        """关键词检索（Sparse Retrieval）。"""
        tokens = [t.lower() for t in query.split() if t]
        if not tokens:
            return []
        
        scored: List[Tuple[str, float]] = []
        for fid, fact in self._facts.items():
            haystack = (fact.canonical_title + " " + fact.canonical_summary).lower()
            score = 0.0
            for t in tokens:
                if t in haystack:
                    score += 1.0
                    if t in fact.canonical_title.lower():
                        score += 0.5
            if score > 0:
                scored.append((fid, score / len(tokens)))
        
        scored.sort(key=lambda x: -x[1])
        picked = scored[:top_k]
        return [(self._facts[fid], s) for fid, s in picked if fid in self._facts]

    def hybrid_search(
        self,
        query: str,
        vector_k: int = 6,
        keyword_k: int = 8,
        final_k: int = 5,
        use_rrf: bool = True,
    ) -> List[Tuple[NormalizedFact, float]]:
        """多路混合检索：向量 + 关键词 + RRF 融合。"""
        # 1. 向量检索
        vec_results = self.search(query, top_k=vector_k)
        vec_rank_map = {fact.fact_id: i + 1 for i, (fact, _) in enumerate(vec_results)}

        # 2. 关键词检索
        kw_results = self.keyword_search(query, top_k=keyword_k)
        kw_rank_map = {fact.fact_id: i + 1 for i, (fact, _) in enumerate(kw_results)}

        if use_rrf:
            # 3. RRF 融合
            fused_scores: Dict[str, float] = {}
            rrf_k = 60
            all_fact_ids = set(vec_rank_map.keys()) | set(kw_rank_map.keys())

            for fid in all_fact_ids:
                score = 0.0
                if fid in vec_rank_map:
                    score += 1.0 / (rrf_k + vec_rank_map[fid])
                if fid in kw_rank_map:
                    score += 1.0 / (rrf_k + kw_rank_map[fid])
                fused_scores[fid] = score

            sorted_fids = sorted(fused_scores.keys(), key=lambda x: -fused_scores[x])
            results = []
            for fid in sorted_fids[:final_k]:
                fact = self._facts.get(fid)
                if fact:
                    results.append((fact, fused_scores[fid]))
            return results
        else:
            fused: Dict[str, Tuple[NormalizedFact, float]] = {}
            for fact, s in vec_results:
                fused[fact.fact_id] = (fact, s)
            for fact, s in kw_results:
                if fact.fact_id in fused:
                    f2, s2 = fused[fact.fact_id]
                    fused[fact.fact_id] = (f2, s2 + s * 0.4)
                else:
                    fused[fact.fact_id] = (fact, s * 0.4)
            merged = sorted(fused.values(), key=lambda x: -x[1])
            return merged[:final_k]

    # ---------- 查询接口 ----------
    def get(self, fact_id: str) -> Optional[NormalizedFact]:
        with self._lock:
            return self._facts.get(fact_id)

    def all_trusted(self) -> List[NormalizedFact]:
        with self._lock:
            return [f for f in self._facts.values() if f.verified]

    def stats(self) -> Dict:
        with self._lock:
            total = len(self._facts)
            trusted = sum(1 for f in self._facts.values() if f.verified)
            high = sum(1 for f in self._facts.values() if f.confidence == "high")
            medium = sum(1 for f in self._facts.values() if f.confidence == "medium")
            low = sum(1 for f in self._facts.values() if f.confidence == "low")
            sources = set()
            for f in self._facts.values():
                sources.update(f.sources)
            return {
                "total": total,
                "trusted": trusted,
                "high_conf": high,
                "medium_conf": medium,
                "low_conf": low,
                "sources": len(sources),
                "index_trained": self._index.is_trained if self._index else False,
                "embedder_fitted": self.embedder._is_fitted,
            }

    def size(self) -> int:
        """返回事实数量。"""
        return len(self._facts)
