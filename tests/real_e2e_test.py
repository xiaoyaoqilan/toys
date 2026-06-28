"""真实端到端测试 - 简化版"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# ========== 测试 1：直接调用 LLM API ==========
def test_llm_direct():
    logger.info("=" * 60)
    logger.info("测试 1：直接调用 LLM API")
    logger.info("=" * 60)
    
    import requests
    
    api_key = "sk-f7d8b42448ef450f838b6ebab7edf60c"
    url = "https://api.deepseek.com/v1/chat/completions"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "用一句话解释什么是RAG"}
        ],
        "max_tokens": 100
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    logger.info("正在调用 LLM API...")
    start = time.time()
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    result = response.json()
    elapsed = time.time() - start
    
    answer = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    
    logger.info(f"✅ 成功！耗时 {elapsed:.2f}s")
    logger.info(f"   Token: total={usage.get('total_tokens', 0)}")
    logger.info(f"   回答: {answer}")
    return True

# ========== 测试 2：LLM 问答 (模拟 RAG) ==========
def test_llm_with_context():
    logger.info("")
    logger.info("=" * 60)
    logger.info("测试 2：LLM 基于上下文生成回答 (模拟 RAG)")
    logger.info("=" * 60)
    
    import requests
    
    api_key = "sk-f7d8b42448ef450f838b6ebab7edf60c"
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # 模拟检索到的事实
    context = """
    - [Apple raises iPhone prices] Apple has announced price increases due to global chip shortages.
    - [Oil prices surge] Oil prices have surged 15% amid supply chain disruptions.
    - [Fed signals rate cut] Federal Reserve may consider cutting interest rates.
    """
    
    question = "苹果公司最近有什么新闻？对市场有什么影响？"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的金融分析师。请基于提供的事实，分析问题并给出专业的回答。"},
            {"role": "user", "content": f"## 问题\n{question}\n\n## 相关事实\n{context}\n\n请分析这些事实，并回答问题。"}
        ],
        "max_tokens": 500
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    logger.info("正在调用 LLM (带上下文)...")
    start = time.time()
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    result = response.json()
    elapsed = time.time() - start
    
    answer = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    
    logger.info(f"✅ 成功！耗时 {elapsed:.2f}s")
    logger.info(f"   Token: total={usage.get('total_tokens', 0)}")
    logger.info("")
    logger.info("📋 LLM 生成的回答:")
    logger.info("-" * 40)
    logger.info(answer)
    logger.info("-" * 40)
    return True

# ========== 主程序 ==========
if __name__ == "__main__":
    logger.info("🔍 真实端到端测试 (简化版)")
    
    ok1 = test_llm_direct()
    
    logger.info("")
    logger.info("-" * 60)
    
    ok2 = test_llm_with_context()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 测试总结")
    logger.info("=" * 60)
    logger.info(f"  测试 1 (LLM 直接调用): {'✅ 正常' if ok1 else '❌ 异常'}")
    logger.info(f"  测试 2 (LLM + 上下文): {'✅ 正常' if ok2 else '❌ 异常'}")
    
    if ok1 and ok2:
        logger.info("")
        logger.info("🎉 恭喜！LLM 工作完全正常！")
        logger.info("   - 可以正常调用 API")
        logger.info("   - 可以基于上下文生成回答 (这就是 RAG 的核心)")
