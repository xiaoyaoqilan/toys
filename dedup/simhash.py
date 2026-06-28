"""SimHash 文本指纹 + 嵌入向量语义去重。"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class SimHash:
    def __init__(self, hash_bits: int = 64):
        self.hash_bits = hash_bits
        self.mask = (1 << hash_bits) - 1

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        shingles: List[str] = []
        for i in range(len(tokens) - 3):
            shingles.append(" ".join(tokens[i : i + 4]))
        return shingles or tokens

    def _hash(self, token: str) -> int:
        return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest(), "big") & self.mask

    def compute(self, text: str) -> int:
        tokens = self._tokenize(text)
        if not tokens:
            return 0
        v = [0] * self.hash_bits
        for tok in tokens:
            h = self._hash(tok)
            for i in range(self.hash_bits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        fp = 0
        for i, x in enumerate(v):
            if x >= 0:
                fp |= 1 << i
        return fp

    @staticmethod
    def hamming(a: int, b: int) -> int:
        return int(bin(a ^ b).count("1"))


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def find_clusters(
    embeddings: np.ndarray,
    simhash_fps: Sequence[int],
    simhash_thresh: int = 3,
    semantic_thresh: float = 0.88,
) -> List[List[int]]:
    """基于 simhash 近邻 + 语义相似度的两阶段聚类。"""
    n = len(embeddings)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    embeddings_norm = np.vstack([l2_normalize(e) for e in embeddings])

    for i in range(n):
        for j in range(i + 1, n):
            if SimHash.hamming(simhash_fps[i], simhash_fps[j]) <= simhash_thresh:
                union(i, j)
            elif float(np.dot(embeddings_norm[i], embeddings_norm[j])) >= semantic_thresh:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def canonical_title(titles: Sequence[str]) -> str:
    """从同簇多个标题中选出代表性标题（出现次数最多的分词序列）。"""
    scored = Counter()
    for t in titles:
        scored[t.strip()] += 1
    if not scored:
        return ""
    return scored.most_common(1)[0][0]
