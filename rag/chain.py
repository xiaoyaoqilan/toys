"""金融 RAG 链路 - 集成前沿技术：RRF融合、Rerank、Query Rewriting。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import PrivateAttr
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("langchain_core not fully available. Using fallback.")
    _LANGCHAIN_AVAILABLE = False

from crawler.models import NormalizedFact, RagAnswer
from kb import FaissKnowledgeBase, Embedder
from .llm_client import LLMClient
from .reranker import LocalReranker


class FinanceRetriever(BaseRetriever):
    """自定义 LangChain Retriever，集成 RRF 融合和 Rerank。"""
    
    _kb: Any = PrivateAttr()
    _embedder: Any = PrivateAttr()
    _reranker: Any = PrivateAttr()
    _top_k: int = 5
    
    def __init__(self, kb: FaissKnowledgeBase, embedder: Embedder, top_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._kb = kb
        self._embedder = embedder
        self._top_k = top_k
        self._reranker = LocalReranker()

    def _get_relevant_documents(self, query: str) -> List["Document"]:
        """
        执行检索：1. 多路召回 2. RRF融合 3. Rerank重排序
        """
        # 1. 多路召回 + RRF 融合
        # 扩大召回范围 (Pump)
        pump_factor = 3
        fused_results = self._kb.hybrid_search(
            query,
            vector_k=self._top_k * pump_factor,
            keyword_k=self._top_k * pump_factor,
            final_k=self._top_k * pump_factor,
            use_rrf=True # 使用前沿的 RRF 算法
        )
        
        if not fused_results:
            return []
            
        # 2. Rerank 重排序 (精排)
        reranked = self._reranker.rerank(
            query, 
            [(f.canonical_title + " " + f.canonical_summary, score, f) for f, score in fused_results],
            top_k=self._top_k
        )
        
        # 3. 转换回 LangChain Document
        docs = []
        for fact, new_score in reranked:
            metadata = {
                "fact_id": fact.fact_id,
                "sources": fact.sources,
                "confidence": fact.confidence,
                "score": new_score, # 使用 Rerank 后的分数
                "title": fact.canonical_title,
            }
            page_content = f"{fact.canonical_title}\n{fact.canonical_summary}"
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs


class FinanceRAG:
    """金融 RAG 封装 - 集成前沿检索技术和 LLM 增强。"""
    
    def __init__(self, kb: FaissKnowledgeBase, embedder: Embedder, llm: Optional[LLMClient] = None):
        self.kb = kb
        self.embedder = embedder
        self.llm = llm or LLMClient()  # 默认初始化 LLM 客户端
        self.reranker = LocalReranker()
        
        # 初始化 LangChain retriever (集成了 RRF 和 Rerank)
        if _LANGCHAIN_AVAILABLE:
            self.retriever = FinanceRetriever(kb=kb, embedder=embedder)
        else:
            self.retriever = None
            logger.warning("LangChain not available, using native retrieval with RRF+Rerank.")

    def _rewrite_query(self, query: str) -> str:
        """查询改写 (Query Rewriting) - 使用 LLM 或规则"""
        # 优先使用 LLM 进行智能改写
        if self.llm and self.llm.is_available:
            return self.llm.rewrite_query(query)
        # Fallback: 简单的规则改写
        query = query.strip()
        if len(query.split()) < 3:
            return f"{query} latest news analysis"
        return query

    def retrieve(self, question: str, top_k: int = 5) -> List[NormalizedFact]:
        """完整的检索流程：Query Rewriting -> Hybrid Search -> Rerank"""
        # 1. 查询改写
        rewritten_query = self._rewrite_query(question)
        
        # 2. 多路召回 + RRF 融合
        pump_factor = 3
        fused_results = self.kb.hybrid_search(
            rewritten_query,
            vector_k=top_k * pump_factor,
            keyword_k=top_k * pump_factor,
            final_k=top_k * pump_factor,
            use_rrf=True
        )
        
        # 3. Rerank 重排序
        if fused_results:
            reranked = self.reranker.rerank(
                rewritten_query,
                [(f.canonical_title + " " + f.canonical_summary, s, f) for f, s in fused_results],
                top_k=top_k
            )
            return [f for f, _ in reranked]
        return []

    def _build_context(self, facts: List[NormalizedFact]) -> str:
        blocks = []
        for f in facts:
            sources = ", ".join(f.sources) or "unknown"
            blocks.append(
                f"- [fact_id={f.fact_id}|sources={sources}|conf={f.confidence}|score={f.score:.2f}] "
                f"{f.canonical_title} — {f.canonical_summary}"
            )
        return "\n".join(blocks)

    def answer(self, question: str, top_k: int = 5) -> RagAnswer:
        """执行问答：查询改写 -> 检索 -> 响应合成"""
        
        # 使用 LangChain retriever (如果可用)
        if self.retriever and _LANGCHAIN_AVAILABLE:
            try:
                docs = self.retriever.invoke(question)
                facts = []
                for doc in docs:
                    fact = NormalizedFact(
                        fact_id=doc.metadata.get("fact_id", ""),
                        canonical_title=doc.metadata.get("title", ""),
                        canonical_summary=doc.page_content,
                        sources=doc.metadata.get("sources", []),
                        confidence=doc.metadata.get("confidence", "low"),
                        score=doc.metadata.get("score", 0.0),
                        entities={},
                        tags=[],
                        article_ids=[],
                        source_count=len(doc.metadata.get("sources", [])),
                    )
                    facts.append(fact)
            except Exception as e:
                logger.error(f"LangChain retriever failed: {e}, falling back to native.")
                facts = self.retrieve(question, top_k)
        else:
            # 本地实现的完整流程
            facts = self.retrieve(question, top_k)
            
        context = self._build_context(facts)
        
        # 4. 响应合成 (Response Synthesis)
        answer_text = self._synthesize_answer(question, facts)

        return RagAnswer(
            question=question,
            answer=answer_text,
            sources=[f.to_dict() for f in facts],
            confidence=self._overall_confidence(facts),
            trace=[
                {"step": "query_rewriting", "input": question, "output": self._rewrite_query(question)},
                {"step": "hybrid_search_with_rrf", "method": "Dense+Sparse+RRF"},
                {"step": "reranking", "method": "LocalReranker"},
                {"context_count": len(facts)},
                {"mode": "advanced_rag_no_llm"},
            ],
        )

    def _synthesize_answer(self, question: str, facts: List[NormalizedFact]) -> str:
        """响应合成 - 优先使用 LLM 生成分析报告，降级为模板"""
        if not facts:
            return "## 未找到相关信息\n\n抱歉，经过多轮检索与重排序，知识库中暂时没有与您的问题高度相关的事实。"

        # 构建上下文
        context = self._build_context(facts)
        
        # 优先使用 LLM 生成分析报告
        if self.llm and self.llm.is_available:
            logger.info("Using LLM for response synthesis...")
            llm_answer = self.llm.synthesize_answer(question, context)
            if llm_answer:
                # 在 LLM 答案后附加事实来源
                sources_section = "\n\n---\n\n### 事实来源\n"
                for i, f in enumerate(facts, 1):
                    sources_str = "、".join(f.sources) if f.sources else "未知信源"
                    sources_section += f"{i}. [{f.confidence.upper()}] {f.canonical_title} ({sources_str})\n"
                return llm_answer + sources_section
        
        # Fallback: 模板化生成
        logger.info("Using template for response synthesis (LLM not available)...")
        lines = [
            f"## 智能分析报告",
            f"**问题**：{question}",
            "",
            f"### 核心发现 (Top {len(facts)} 高相关度事实)",
        ]
        
        for i, f in enumerate(facts, 1):
            sources_str = "、".join(f.sources) if f.sources else "未知信源"
            lines.append(f"{i}. **[{f.confidence.upper()} | Score: {f.score:.2f}] {f.canonical_title}**")
            lines.append(f"   - **摘要**: {f.canonical_summary[:200]}...")
            lines.append(f"   - **信源**: {sources_str}")
            lines.append("")

        lines.extend([
            "### 分析摘要",
        ])
        
        # 汇总置信度
        high_conf_count = sum(1 for f in facts if f.confidence == "high")
        medium_conf_count = sum(1 for f in facts if f.confidence == "medium")
        
        if high_conf_count >= 2:
            lines.append("✅ 本次检索结果中包含多条高置信度事实（经 ≥3 家独立媒体交叉验证），信息可靠度高。")
        elif medium_conf_count >= 2:
            lines.append("⚠️ 本次检索结果主要由中等置信度事实构成，建议关注后续报道以确认。")
        else:
            lines.append("⚠️ 本次检索结果事实置信度普遍较低，可能为单一来源报道，仅供参考。")

        return "\n".join(lines)

    @staticmethod
    def _overall_confidence(facts: List[NormalizedFact]) -> str:
        if not facts:
            return "low"
        high = sum(1 for f in facts if f.confidence == "high")
        if high >= 2:
            return "high"
        if high >= 1 or any(f.confidence == "medium" for f in facts):
            return "medium"
        return "low"
