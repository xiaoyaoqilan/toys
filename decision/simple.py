"""简单决策系统 - finurls + RAG + LLM。"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

from crawler.finurls import FinurlsCrawler
from crawler.models import NormalizedFact
from kb.embeddings import Embedder
from kb.faiss_kb import FaissKnowledgeBase
from rag.llm_client import LLMClient
from config.settings import AppConf

try:
    from market.binance_analyzer import BinanceAnalyzer
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False


# 停用词
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "can", "could", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "than", "too", "very",
    "s", "t", "just", "about", "also", "said", "new", "its", "says", "like",
    "get", "one", "two", "first", "last", "even", "back", "way", "take",
    "still", "since", "while", "now", "year", "years", "day", "time", "after",
    "over", "into", "could", "made", "may", "yet", "some", "into", "would",
    "been", "here", "there", "their", "them", "then", "than", "well", "down",
    "out", "up", "off", "very", "really", "much", "many", "any", "give",
    "going", "want", "thing", "things", "make", "made", "big", "high", "low",
    "old", "right", "good", "great", "best", "top", "main", "key", "real",
    "late", "early", "long", "short", "few", "more", "less", "next", "previous",
    "company", "companies", "people", "person", "world", "life", "work",
}


def extract_keywords(articles, top_n=8):
    """从文章标题中提取高频关键词。"""
    word_counter = Counter()
    
    for art in articles:
        title = art.title.lower()
        # 提取有意义的单词
        words = re.findall(r'\b[a-z]{3,}\b', title)
        for word in words:
            if word not in STOPWORDS and len(word) >= 3:
                word_counter[word] += 1
    
    # 返回高频词
    return [word for word, count in word_counter.most_common(top_n)]


def main():
    logger.info("=" * 60)
    logger.info("决策系统: finurls → RAG → LLM")
    logger.info("=" * 60)

    conf = AppConf()
    crawler = FinurlsCrawler(conf)
    embedder = Embedder()
    kb = FaissKnowledgeBase(index_dir="data/decision_index", embedder=embedder)
    llm = LLMClient(api_key=os.getenv("DEEPSEEK_API_KEY", ""))

    # 1. 抓取
    logger.info("\n[1] 抓取 finurls...")
    kb.clear()
    articles = crawler.crawl()
    logger.info(f"    {len(articles)} 篇新闻")
    
    # 提取关键词
    keywords = extract_keywords(articles, top_n=10)
    logger.info(f"    提取到的关键词: {keywords}")

    # 2. 建库
    logger.info("\n[2] 构建知识库...")
    seen = set()
    facts = []
    for i, art in enumerate(articles):
        key = art.title.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        facts.append(NormalizedFact(
            fact_id=f"f_{i}",
            canonical_title=art.title,
            canonical_summary=(art.content or art.title)[:200],
            sources=[art.source],
            source_count=1,
            confidence="medium",
            score=0.5,
        ))
    kb.upsert_batch(facts)
    logger.info(f"    {len(facts)} 条事实")

    # 3. 获取 Binance 实时行情 + 布林带
    logger.info("\n[3] Binance 实时行情分析...")
    binance_analyses = []
    
    if BINANCE_AVAILABLE:
        try:
            bn = BinanceAnalyzer()
            crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
            
            for sym in crypto_symbols:
                logger.info(f"    分析 {sym}...")
                result = bn.analyze_symbol(sym)
                binance_analyses.append(result)
                
                # 输出技术指标
                if "ticker" in result and "bollinger" in result:
                    t = result["ticker"]
                    bb = result["bollinger"]
                    logger.info(f"      价格: ${t['price']:,.2f} ({t['change_24h']:+.2f}%)")
                    logger.info(f"      布林带: 上${bb['upper']:,.2f} 中${bb['middle']:,.2f} 下${bb['lower']:,.2f}")
                    logger.info(f"      信号: {bb['signal']}")
                    
        except Exception as e:
            logger.error(f"Binance 分析失败: {e}")
    else:
        logger.warning("Binance 模块不可用")
    
    # 4. 用高频词检索 + LLM 决策（结合新闻+技术）
    logger.info("\n[4] 综合决策分析...")
    results = []

    for keyword in keywords[:5]:
        query = keyword
        logger.info(f"\n    🔍 {query}")
        retrieved = kb.hybrid_search(query, final_k=5)

        if not retrieved:
            logger.info("       无结果")
            continue

        # 新闻上下文
        news_context = ""
        for i, item in enumerate(retrieved, 1):
            f = item[0] if isinstance(item, tuple) else item
            src = "/".join(f.sources[:2])
            news_context += f"{i}. [{src}] {f.canonical_title}\n"
        
        # 技术分析上下文（如果关键词涉及加密）
        tech_context = ""
        crypto_keywords = ["stock", "stocks", "market", "billion"]
        if any(kw in keyword.lower() for kw in crypto_keywords) or "coin" in keyword.lower():
            for ba in binance_analyses:
                if "bollinger" in ba and "ticker" in ba:
                    t = ba["ticker"]
                    bb = ba["bollinger"]
                    tech_context += f"- {ba['symbol']}: ${t['price']:,.2f} ({t['change_24h']:+.2f}%) 布林带{bb['signal']}\n"
        
        # 综合给 LLM
        context = f"新闻：\n{news_context}\n"
        if tech_context:
            context += f"\n技术面（Binance 4H布林带）：\n{tech_context}\n"

        system = "你是交易员。只给结论，不要废话。"
        user = f"""基于以下信息，直接给出结论：

{context}

格式要求：
1. 事实：[1句话总结]
2. 方向：[看多/看空/中性]
3. 操作：[具体建议]
4. 理由：[一句话]

禁止出现套话。"""

        try:
            resp = llm.chat(system, user, temperature=0.1, max_tokens=200)
            results.append({"query": query, "analysis": resp or "无分析"})
            logger.info("       ✅ 完成")
        except Exception as e:
            results.append({"query": query, "analysis": f"错误: {e}"})
            logger.error(f"       ❌ {e}")

    # 5. 输出报告
    logger.info("\n[5] 生成报告...")
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("\n" + "=" * 60)
    print(f"📊 财经决策日报 - {today}")
    print("=" * 60)
    
    # Binance 技术分析部分
    if binance_analyses:
        print("\n## 🔴 Binance 技术分析（4H 布林带）\n")
        for ba in binance_analyses:
            if "ticker" in ba and "bollinger" in ba:
                t = ba["ticker"]
                bb = ba["bollinger"]
                print(f"### {ba['symbol']}")
                print(f"- 价格: ${t['price']:,.2f} ({'🟢' if t['change_24h'] > 0 else '🔴'} {t['change_24h']:+.2f}%)")
                print(f"- 布林带(4H): 上${bb['upper']:,.2f} 中${bb['middle']:,.2f} 下${bb['lower']:,.2f}")
                print(f"- 带宽: {bb['bandwidth']:.2f}%")
                print(f"- 信号: {bb['signal']}")
                print("")

    if not results:
        print("\n⚠️ 没有相关分析结果")
    else:
        for r in results:
            print(f"\n## 🔍 {r['query']}")
            print(r["analysis"])
            print("-" * 40)

    return results


if __name__ == "__main__":
    main()
