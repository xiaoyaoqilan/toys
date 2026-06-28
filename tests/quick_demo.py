"""快速演示脚本 - 展示 RAG 系统效果。"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def demo():
    logger.info("=" * 60)
    logger.info("🚀 金融 RAG 系统 - 效果演示")
    logger.info("=" * 60)
    
    try:
        from kb import FaissKnowledgeBase, Embedder
        from rag import FinanceRAG
        
        # 加载知识库
        logger.info("\n📚 [1] 加载知识库...")
        embedder = Embedder()
        index_dir = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
        
        if not os.path.exists(index_dir):
            logger.error("❌ 索引不存在，请先运行爬虫构建知识库！")
            return
            
        kb = FaissKnowledgeBase(index_dir=index_dir, embedder=embedder)
        stats = kb.stats()
        logger.info(f"   知识库已加载: {stats.get('total', 0)} 条事实")
        
        # 初始化 RAG
        logger.info("\n⚙️ [2] 初始化 RAG 系统...")
        rag = FinanceRAG(kb=kb, embedder=embedder)
        logger.info("   RAG 系统就绪 ✓")
        
        # 测试查询
        test_queries = [
            "Bitcoin",
            "Apple",
            "Oil price",
            "Stock market",
        ]
        
        for query in test_queries:
            logger.info(f"\n{'─' * 50}")
            logger.info(f"🔍 查询: '{query}'")
            logger.info(f"{'─' * 50}")
            
            answer = rag.answer(query, top_k=3)
            
            logger.info(f"\n📊 答案摘要:")
            # 只打印答案的前几行
            lines = answer.answer.split('\n')
            for line in lines[:15]:
                if line.strip():
                    logger.info(f"   {line}")
                    
            logger.info(f"\n   置信度: {answer.confidence}")
            logger.info(f"   召回事实数: {len(answer.sources)}")
            
            if answer.trace:
                steps = [t.get('step', '') for t in answer.trace if 'step' in t]
                logger.info(f"   技术链路: {' → '.join(steps)}")
                
    except Exception as e:
        logger.error(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo()
