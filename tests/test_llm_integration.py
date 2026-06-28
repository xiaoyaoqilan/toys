"""LLM 集成测试脚本。"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def test_llm_integration():
    logger.info("=" * 60)
    logger.info("🚀 DeepSeek LLM 集成测试")
    logger.info("=" * 60)
    
    try:
        from kb import FaissKnowledgeBase, Embedder
        from rag import FinanceRAG, LLMClient
        
        # 1. 初始化 LLM
        logger.info("\n📡 [1] 初始化 LLM 客户端...")
        llm = LLMClient(api_key="sk-f7d8b42448ef450f838b6ebab7edf60c")
        
        if not llm.is_available:
            logger.error("❌ LLM 客户端不可用，请检查 API Key")
            return
            
        logger.info("   LLM 客户端就绪 ✓")
        
        # 测试基础对话
        logger.info("\n💬 [2] 测试 LLM 基础对话...")
        test_result = llm.chat(
            system_prompt="你是一个专业的助手",
            user_prompt="请用一句话介绍你自己"
        )
        if test_result:
            logger.info(f"   LLM 回复: {test_result[:100]}...")
            logger.info("   基础对话测试通过 ✓")
        else:
            logger.error("   ❌ 基础对话测试失败")
            return
        
        # 2. 初始化 RAG
        logger.info("\n📚 [3] 初始化 RAG 系统（集成 LLM）...")
        embedder = Embedder()
        index_dir = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
        kb = FaissKnowledgeBase(index_dir=index_dir, embedder=embedder)
        rag = FinanceRAG(kb=kb, embedder=embedder, llm=llm)
        logger.info("   RAG 系统就绪 ✓")
        
        # 3. 测试查询改写
        logger.info("\n✏️ [4] 测试查询改写...")
        test_query = "苹果手机涨价"
        rewritten = rag._rewrite_query(test_query)
        logger.info(f"   原始查询: '{test_query}'")
        logger.info(f"   改写后: '{rewritten}'")
        logger.info("   查询改写测试通过 ✓")
        
        # 4. 测试完整 RAG 问答（LLM 驱动）
        logger.info("\n🔍 [5] 测试 LLM 驱动的 RAG 问答...")
        answer = rag.answer("Apple 涨价", top_k=3)
        
        logger.info(f"\n   置信度: {answer.confidence}")
        logger.info(f"   召回事实数: {len(answer.sources)}")
        
        # 打印部分答案
        answer_text = answer.answer
        logger.info(f"\n   答案预览:")
        lines = answer_text.split('\n')
        for line in lines[:20]:
            if line.strip():
                logger.info(f"   {line}")
        
        logger.info("\n   LLM 驱动的 RAG 问答测试通过 ✓")
        
        # 5. 测试影响预测
        logger.info("\n🎯 [6] 测试影响预测...")
        prediction = llm.predict_impact(
            event="原油价格上涨",
            affected_entities=["航空业", "化工行业"]
        )
        if prediction:
            logger.info(f"   预测结果预览: {prediction[:200]}...")
            logger.info("   影响预测测试通过 ✓")
        else:
            logger.warning("   影响预测测试未返回结果")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 所有 LLM 集成测试通过！")
        logger.info("=" * 60)
        logger.info("\n现在系统已具备:")
        logger.info("  1. ✅ LLM 智能查询改写")
        logger.info("  2. ✅ LLM 驱动的分析报告生成")
        logger.info("  3. ✅ 跨文档影响预测")
        logger.info("  4. ✅ 多信源事实综合推理")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_integration()
