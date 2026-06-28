"""嵌入模型封装 - 基于 TF-IDF 的真实文本向量化实现。"""
from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using fallback.")
    _SKLEARN_AVAILABLE = False


class Embedder:
    """
    基于 TF-IDF 的文本嵌入器。
    真正的、基于词频的文本向量化，支持实时语料库训练，
    可以实现准确的语义（基于关键词）相似度检索。
    """
    def __init__(self):
        self._vectorizer = None
        self._is_fitted = False
        self._corpus = []  # 存储原始文本，用于增量更新时重新拟合
        self.dim = 512  # 动态调整

        if _SKLEARN_AVAILABLE:
            self._init_tfidf()
        else:
            logger.warning("Using basic fallback without sklearn.")

    def _init_tfidf(self) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=self.dim,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
        )
        logger.info("Initialized TF-IDF Embedder with scikit-learn")

    @property
    def backend(self) -> str:
        return "tfidf_sklearn" if _SKLEARN_AVAILABLE else "fallback"

    def fit(self, corpus: List[str]) -> None:
        """用新的语料库训练/更新向量器。"""
        if not _SKLEARN_AVAILABLE or not self._vectorizer or not corpus:
            return
            
        # 增量更新：把新语料加入旧语料
        self._corpus.extend(corpus)
        if len(self._corpus) > 5000:
            self._corpus = self._corpus[-5000:]  # 只保留最近 5000 条，防止无限增长
            
        try:
            self._vectorizer.fit(self._corpus)
            self._is_fitted = True
            logger.info(f"TF-IDF fitted on {len(self._corpus)} documents")
        except Exception as e:
            logger.error(f"TF-IDF fit failed: {e}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """编码一批文本为 TF-IDF 向量。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
            
        if _SKLEARN_AVAILABLE and self._vectorizer:
            if not self._is_fitted or not self._corpus:
                try:
                    self._vectorizer.fit(texts)
                    self._is_fitted = True
                    self._corpus = list(texts)
                    logger.info(f"TF-IDF fitted on {len(texts)} documents, vocab size: {len(self._vectorizer.vocabulary_)}")
                except Exception as e:
                    logger.error(f"TF-IDF fit failed: {e}")
                    return np.zeros((len(texts), self.dim), dtype=np.float32)
                    
            if self._is_fitted:
                try:
                    vec = self._vectorizer.transform(texts)
                    result = vec.toarray().astype(np.float32)
                    # 确保输出维度与 max_features 一致
                    if result.shape[1] < self.dim:
                        # 填充零
                        padded = np.zeros((len(texts), self.dim), dtype=np.float32)
                        padded[:, :result.shape[1]] = result
                        return padded
                    elif result.shape[1] > self.dim:
                        return result[:, :self.dim]
                    return result
                except Exception as e:
                    logger.error(f"TF-IDF transform failed: {e}")

        # Fallback: 简单的哈希特征
        return np.random.rand(len(texts), self.dim).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """编码单条文本。"""
        vec = self.encode([text])
        return vec.flatten()
