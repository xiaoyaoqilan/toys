"""信号提醒系统 - finurls RAG + Binance/yfinance。"""
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
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

from crawler.finurls import FinurlsCrawler
from crawler.models import NormalizedFact
from kb.embeddings import Embedder
from kb.faiss_kb import FaissKnowledgeBase
from rag.llm_client import LLMClient
from config.settings import AppConf
from market.analyzer import MarketAnalyzer


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "can", "could", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "all", "each", "every", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "than", "too", "very",
    "s", "t", "just", "about", "also", "said", "new", "its", "says", "like",
    "get", "one", "two", "first", "last", "even", "back", "way", "take",
    "still", "since", "while", "now", "year", "years", "day", "time",
    "over", "into", "been", "here", "there", "their", "them", "then", "than",
    "well", "down", "out", "up", "off", "very", "really", "much", "many", "any",
    "give", "going", "want", "thing", "things", "make", "made", "big", "high", "low",
    "old", "right", "good", "great", "best", "top", "main", "key", "real",
    "late", "early", "long", "short", "few", "more", "less", "next", "previous",
    "company", "companies", "people", "person", "world", "life", "work",
}


# 关注的标的映射（中文名称 → 英文检索词 + 技术分析符号）
WATCHLIST = {
    "BTCUSDT": {"name": "比特币", "query": "bitcoin BTC crypto", "type": "crypto"},
    "ETHUSDT": {"name": "以太坊", "query": "ethereum ETH crypto", "type": "crypto"},
    "SOLUSDT": {"name": "SOL", "query": "solana SOL crypto", "type": "crypto"},
    "BNBUSDT": {"name": "BNB", "query": "binance BNB crypto", "type": "crypto"},
    "TSLA": {"name": "特斯拉", "query": "tesla TSLA elon musk", "type": "stock"},
    "AAPL": {"name": "苹果", "query": "apple AAPL iphone", "type": "stock"},
    "NVDA": {"name": "英伟达", "query": "nvidia NVDA chip AI", "type": "stock"},
    "GOOGL": {"name": "谷歌", "query": "google alphabet GOOGL", "type": "stock"},
    "QQQ": {"name": "纳指ETF", "query": "nasdaq QQQ tech", "type": "stock"},
    "SPY": {"name": "标普500", "query": "S&P 500 SPY market", "type": "stock"},
}


def extract_keywords(articles, top_n=10):
    """从文章标题中提取高频关键词。"""
    word_counter = Counter()
    for art in articles:
        title = art.title.lower()
        words = re.findall(r'\b[a-z]{3,}\b', title)
        for word in words:
            if word not in STOPWORDS and len(word) >= 3:
                word_counter[word] += 1
    return [word for word, count in word_counter.most_common(top_n)]


def get_signal_emoji(action: str) -> str:
    """获取信号图标。"""
    if "做多" in action or "买入" in action:
        return "🟢"
    elif "做空" in action or "卖出" in action:
        return "🔴"
    else:
        return "⚪"


def main():
    logger.info("=" * 50)
    logger.info("📨 信号提醒系统")
    logger.info("=" * 50)
    
    conf = AppConf()
    crawler = FinurlsCrawler(conf)
    embedder = Embedder()
    kb = FaissKnowledgeBase(index_dir="data/signal_index", embedder=embedder)
    llm = LLMClient(api_key=os.getenv("DEEPSEEK_API_KEY", ""))
    market = MarketAnalyzer()
    
    # 1. 获取新闻
    logger.info("\n[1] finurls 抓取...")
    kb.clear()
    articles = crawler.crawl()
    keywords = extract_keywords(articles, top_n=8)
    logger.info(f"    关键词: {keywords}")
    
    # 构建知识库
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
    
    # 2. 获取技术信号
    logger.info("\n[2] 技术信号分析...")
    market_data = market.get_all_analysis()
    tech_signals = {}
    
    for item in market_data.get("crypto", []) + market_data.get("stocks", []):
        if "bollinger" in item and "error" not in item.get("bollinger", {}):
            symbol = item["symbol"]
            bb = item["bollinger"]
            tech_signals[symbol] = {
                "signal": bb.get("signal", ""),
                "action": bb.get("action", "观望"),
            }
    
    # 3. 用 RAG 检索新闻事实，然后 LLM 分析
    logger.info("\n[3] RAG 检索 + LLM 分析...")
    analysis_results = {}
    
    for symbol, info in WATCHLIST.items():
        # 使用英文检索词
        query = info["query"]
        logger.info(f"    🔍 检索: {query}")
        
        # RAG 检索相关新闻
        retrieved = kb.hybrid_search(query, final_k=3)
        
        if not retrieved:
            analysis_results[symbol] = "无相关新闻"
            continue
        
        # 构建新闻上下文
        news_context = ""
        for item in retrieved:
            f = item[0] if isinstance(item, tuple) else item
            news_context += f"- {f.canonical_title}\n"
        
        # 获取技术信号
        tech_action = tech_signals.get(symbol, {}).get("action", "观望")
        
        # LLM 分析：结合新闻 + 技术信号
        system = "你是交易员。基于事实给出明确判断。"
        user = f"""关于"{info['name']}"：

相关新闻：
{news_context}

技术信号：布林带{tech_action}

请给出：
1. 新闻判断：正面/负面/中性
2. 综合方向：看多/看空/观望
3. 操作建议

只返回结论，不要废话。"""
        
        try:
            resp = llm.chat(system, user, temperature=0.1, max_tokens=150)
            analysis_results[symbol] = resp.strip()
            logger.info(f"       ✅ {resp.strip()[:50]}")
        except Exception as e:
            analysis_results[symbol] = f"分析失败: {e}"
            logger.error(f"       ❌ {e}")
    
    # 4. 生成信号提醒
    logger.info("\n[4] 生成信号提醒...")
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    print("\n" + "=" * 50)
    print(f"📨 信号提醒 - {today}")
    print("=" * 50)
    
    print("\n📊 **标的信号**:\n")
    
    for symbol, info in WATCHLIST.items():
        analysis = analysis_results.get(symbol, "")
        
        # 根据分析结果提取方向
        if "看多" in analysis or "买入" in analysis or "做多" in analysis:
            emoji = "🟢"
            direction = "看多"
        elif "看空" in analysis or "卖出" in analysis or "做空" in analysis:
            emoji = "🔴"
            direction = "看空"
        else:
            emoji = "⚪"
            direction = "观望"
        
        print(f"{emoji} {info['name']} → {direction}")
    
    print("\n💡 **操作建议**:")
    
    # 统计信号
    buy_count = sum(1 for s in analysis_results.values() if "看多" in s or "买入" in s)
    sell_count = sum(1 for s in analysis_results.values() if "看空" in s or "卖出" in s)
    
    if buy_count > sell_count:
        print("🟢 整体偏多，可考虑买入")
    elif sell_count > buy_count:
        print("🔴 整体偏空，可考虑卖出")
    else:
        print("⚪ 信号均衡，保持观望")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
