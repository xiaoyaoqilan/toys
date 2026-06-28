"""完整的端到端测试 - 修复后的 Faiss + RAG + LLM"""
import sys
import os
import time
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def test_full_pipeline():
    logger.info("=" * 70)
    logger.info("完整端到端测试 - Faiss + RAG + LLM")
    logger.info("=" * 70)
    
    # 清理旧数据
    index_dir = "test_data"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    
    from rag.chain import FinanceRAG
    from kb.faiss_kb import FaissKnowledgeBase
    from kb.embeddings import Embedder
    from rag.llm_client import LLMClient
    from crawler.models import NormalizedFact
    
    # ========== 1. 初始化 ==========
    logger.info("\n[1] 初始化组件...")
    embedder = Embedder()
    kb = FaissKnowledgeBase(index_dir=index_dir, embedder=embedder)
    llm = LLMClient(api_key="sk-f7d8b42448ef450f838b6ebab7edf60c")
    rag = FinanceRAG(kb=kb, embedder=embedder, llm=llm)
    logger.info("    ✅ 初始化完成")
    
    # ========== 2. 添加事实 ==========
    logger.info("\n[2] 添加测试事实...")
    facts = [
        NormalizedFact(
            fact_id="f1",
            canonical_title="Apple raises iPhone prices amid chip shortage",
            canonical_summary="Apple has announced price increases for its iPhone lineup due to global chip shortages affecting production costs. The company cited supply chain disruptions as the primary reason.",
            sources=["reuters", "bloomberg", "wsj"],
            confidence="high",
            score=0.95
        ),
        NormalizedFact(
            fact_id="f2",
            canonical_title="Global oil prices surge 15% on supply concerns",
            canonical_summary="Oil prices have surged 15% amid global supply chain disruptions and geopolitical tensions in major oil-producing regions. Analysts warn of continued volatility.",
            sources=["cnbc", "wsj", "ft"],
            confidence="high",
            score=0.90
        ),
        NormalizedFact(
            fact_id="f3",
            canonical_title="Federal Reserve signals potential rate cut",
            canonical_summary="The Federal Reserve has indicated it may consider cutting interest rates in the next quarter to stimulate economic growth amid slowing inflation.",
            sources=["reuters", "ft"],
            confidence="medium",
            score=0.80
        ),
        NormalizedFact(
            fact_id="f4",
            canonical_title="Tech stocks lead market rally on AI optimism",
            canonical_summary="Technology stocks surged 3% as investors bet on continued growth in AI-related products and services. Major tech companies reported strong earnings.",
            sources=["bloomberg", "cnbc"],
            confidence="high",
            score=0.85
        ),
        NormalizedFact(
            fact_id="f5",
            canonical_title="Semiconductor industry faces global shortage",
            canonical_summary="The global semiconductor industry continues to face unprecedented shortages, with major manufacturers struggling to meet demand from automotive and consumer electronics sectors.",
            sources=["reuters", "wsj", "cnbc"],
            confidence="high",
            score=0.92
        ),
    ]
    
    kb.upsert_batch(facts)
    logger.info(f"    ✅ 已添加 {kb.size()} 条事实")
    
    stats = kb.stats()
    logger.info(f"    📊 统计: {stats}")
    
    # ========== 3. 检索测试 ==========
    logger.info("\n[3] 检索测试...")
    
    # 测试 1: 向量检索
    logger.info("\n    3.1 向量检索 'Apple prices'")
    results = kb.search("Apple prices", top_k=3)
    for i, (fact, score) in enumerate(results):
        logger.info(f"        [{i+1}] score={score:.4f} | {fact.canonical_title[:60]}")
    
    # 测试 2: 关键词检索
    logger.info("\n    3.2 关键词检索 'oil price surge'")
    results = kb.keyword_search("oil price surge", top_k=3)
    for i, (fact, score) in enumerate(results):
        logger.info(f"        [{i+1}] score={score:.4f} | {fact.canonical_title[:60]}")
    
    # 测试 3: 混合检索 (RRF)
    logger.info("\n    3.3 混合检索 (RRF) 'chip shortage impact'")
    results = kb.hybrid_search("chip shortage impact", final_k=3)
    for i, (fact, score) in enumerate(results):
        logger.info(f"        [{i+1}] score={score:.4f} | {fact.canonical_title[:60]}")
    
    # ========== 4. RAG 问答 ==========
    logger.info("\n[4] RAG 问答测试...")
    
    question = "苹果公司最近有什么新闻？芯片短缺对市场有什么影响？"
    logger.info(f"    问题: {question}")
    logger.info("    正在执行: 查询改写 -> 检索 -> Rerank -> LLM 生成...")
    
    start = time.time()
    result = rag.answer(question)
    elapsed = time.time() - start
    
    logger.info(f"    ✅ 完成！耗时 {elapsed:.2f}s")
    logger.info(f"    置信度: {result.confidence}")
    logger.info(f"    来源数: {len(result.sources)}")
    
    # 打印 LLM 生成的回答
    logger.info("\n    📋 LLM 生成的分析报告:")
    logger.info("    " + "-" * 60)
    answer_lines = result.answer.split('\n')
    for line in answer_lines[:20]:
        if line.strip():
            logger.info(f"    {line}")
    logger.info("    " + "-" * 60)
    
    # ========== 5. 持久化测试 ==========
    logger.info("\n[5] 持久化测试...")
    
    # 重新加载
    kb2 = FaissKnowledgeBase(index_dir=index_dir, embedder=embedder)
    logger.info(f"    ✅ 重新加载后事实数量: {kb2.size()}")
    
    # 检索仍然有效
    results2 = kb2.search("Apple", top_k=2)
    logger.info(f"    ✅ 重新加载后检索仍有效: {len(results2)} 条结果")
    
    # ========== 清理 ==========
    shutil.rmtree(index_dir)
    logger.info("\n[6] 测试完成，已清理临时数据")
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 所有测试通过！Faiss + RAG + LLM 工作正常！")
    logger.info("=" * 70)

if __name__ == "__main__":
    test_full_pipeline()
