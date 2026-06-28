"""重排序器 (Reranker) - 用于对召回结果进行二次排序。"""
from __future__ import annotations

from typing import List, Tuple

from loguru import logger

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False


class LocalReranker:
    """
    本地重排序器。
    目前使用更精细的打分逻辑：结合 TF-IDF 词频重叠度和语义相似度进行加权。
    未来可以无缝升级为 Cross-Encoder 模型（如 bge-reranker）。
    """
    def __init__(self):
        if not _NP_AVAILABLE:
            raise ImportError("numpy is required for LocalReranker")

    def rerank(self, query: str, documents: List[Tuple[str, float, any]], top_k: int = 5) -> List[Tuple[any, float]]:
        """
        对召回的文档列表进行重排序。
        documents: [(document_content, initial_score, document_object), ...]
        返回: [(document_object, new_score), ...]
        """
        if not documents:
            return []

        # 1. 计算查询词频
        query_terms = self._extract_terms(query)
        
        scored_docs = []
        for content, initial_score, doc_obj in documents:
            # 2. 计算文档中查询词的出现频率 (Term Frequency)
            tf_score = self._compute_tf_score(query_terms, content)
            
            # 3. 计算文档的重要性 (初始召回分 + 词频分)
            # 加权融合：初始召回分占 60%，词频精准度占 40%
            final_score = 0.6 * initial_score + 0.4 * tf_score
            
            scored_docs.append((doc_obj, final_score))

        # 4. 按最终分数排序
        scored_docs.sort(key=lambda x: -x[1])
        
        # 5. 截断到 top_k
        return scored_docs[:top_k]

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """提取查询中的关键词。"""
        # 简单的英文分词和去停用词
        text = text.lower()
        terms = [t for t in text.split() if len(t) > 2] # 过滤短词
        # 可以添加更多的关键词提取逻辑，如 Bigram
        bigrams = [text[i:i+2] for i in range(len(text) - 1) if text[i].isalpha() and text[i+1].isalpha()]
        return terms + bigrams

    @staticmethod
    def _compute_tf_score(query_terms: List[str], document: str) -> float:
        """计算查询词在文档中的频率得分。"""
        if not query_terms:
            return 0.0
            
        doc_lower = document.lower()
        score = 0.0
        for term in query_terms:
            # 计算词频
            count = doc_lower.count(term)
            if count > 0:
                # 用对数平滑，避免高频词过度影响
                score += np.log1p(count)
                
        # 归一化：得分除以查询词数量
        return score / max(1, len(query_terms))
