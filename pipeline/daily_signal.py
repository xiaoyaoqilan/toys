"""金融信号推送系统 - 每天9点执行，抓取finurls，分析，推送到微信。"""
from __future__ import annotations

import os
import re
import sys
import requests
from collections import Counter
from datetime import datetime
from pathlib import Path

# 加载环境变量
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
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "may", "might", "can", "could", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "and", "but", "or", "not",
    "this", "that", "these", "those", "what", "which", "who",
    "all", "each", "every", "more", "most", "other", "some",
    "s", "t", "just", "about", "also", "said", "new", "its",
    "says", "like", "get", "one", "two", "first", "last",
    "even", "back", "way", "take", "still", "since", "while",
    "now", "year", "years", "day", "time", "over", "into",
    "well", "down", "out", "up", "off", "very", "much", "many",
    "company", "companies", "people", "person", "world", "life",
}

WATCHLIST = [
    {"symbol": "BTCUSDT", "name": "比特币", "query": "bitcoin BTC crypto"},
    {"symbol": "ETHUSDT", "name": "以太坊", "query": "ethereum ETH crypto"},
    {"symbol": "SOLUSDT", "name": "SOL", "query": "solana SOL crypto"},
    {"symbol": "BNBUSDT", "name": "BNB", "query": "binance BNB crypto"},
    {"symbol": "TSLA", "name": "特斯拉", "query": "tesla TSLA elon musk"},
    {"symbol": "AAPL", "name": "苹果", "query": "apple AAPL iphone"},
    {"symbol": "NVDA", "name": "英伟达", "query": "nvidia NVDA chip AI"},
    {"symbol": "GOOGL", "name": "谷歌", "query": "google alphabet GOOGL"},
    {"symbol": "QQQ", "name": "纳指ETF", "query": "nasdaq QQQ tech"},
    {"symbol": "SPY", "name": "标普500", "query": "S&P 500 SPY market"},
]


def extract_keywords(articles, top_n=10):
    word_counter = Counter()
    for art in articles:
        title = art.title.lower()
        words = re.findall(r'\b[a-z]{3,}\b', title)
        for word in words:
            if word not in STOPWORDS and len(word) >= 3:
                word_counter[word] += 1
    return [w for w, c in word_counter.most_common(top_n)]


def push_to_wechat(title: str, content: str) -> bool:
    sendkey = os.getenv("SERVERCHAN_KEY", "")
    if not sendkey:
        logger.warning("SERVERCHAN_KEY 未配置，跳过微信推送")
        return False
    
    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    data = {"title": title[:32], "desp": content[:65535]}
    
    try:
        resp = requests.post(url, data=data, timeout=10)
        result = resp.json()
        if result.get("code") == 0:
            logger.info("✅ 微信推送成功")
            return True
        else:
            logger.error(f"❌ 微信推送失败: {result}")
            return False
    except Exception as e:
        logger.error(f"❌ 微信推送异常: {e}")
        return False


def run_pipeline():
    logger.info("=" * 50)
    logger.info("🚀 金融信号系统启动")
    logger.info("=" * 50)
    
    conf = AppConf()
    crawler = FinurlsCrawler(conf)
    embedder = Embedder()
    kb = FaissKnowledgeBase(index_dir="data/finance_index", embedder=embedder)
    llm = LLMClient(api_key=os.getenv("DEEPSEEK_API_KEY", ""))
    market = MarketAnalyzer()
    
    # 1. 抓取新闻
    logger.info("\n[1] 抓取 finurls...")
    kb.clear()
    articles = crawler.crawl()
    keywords = extract_keywords(articles, top_n=8)
    logger.info(f"    获取 {len(articles)} 篇，关键词: {keywords}")
    
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
    
    # 2. 技术信号
    logger.info("\n[2] 技术分析...")
    market_data = market.get_all_analysis()
    tech_signals = {}
    
    for item in market_data.get("crypto", []) + market_data.get("stocks", []):
        if "bollinger" in item and "error" not in item.get("bollinger", {}):
            sym = item["symbol"]
            bb = item["bollinger"]
            tech_signals[sym] = bb.get("action", "观望")
    
    # 3. RAG + LLM 分析
    logger.info("\n[3] RAG + LLM 分析...")
    analysis_results = []
    
    for w in WATCHLIST:
        symbol = w["symbol"]
        name = w["name"]
        query = w["query"]
        
        logger.info(f"    分析 {name}...")
        
        retrieved = kb.hybrid_search(query, final_k=3)
        
        if not retrieved:
            analysis_results.append({"name": name, "direction": "观望", "reason": "无相关新闻"})
            continue
        
        news_text = ""
        for item in retrieved:
            f = item[0] if isinstance(item, tuple) else item
            news_text += f"- {f.canonical_title}\n"
        
        tech_action = tech_signals.get(symbol, "观望")
        
        system = "你是交易员。基于事实给出结论。"
        user = f"""关于"{name}"：

相关新闻：
{news_text}

技术信号：布林带{tech_action}

请给出：
1. 综合方向：看多/看空/观望
2. 理由：一句话

格式：
方向：[看多/看空/观望]
理由：[一句话]"""
        
        try:
            resp = llm.chat(system, user, temperature=0.1, max_tokens=100)
            lines = resp.strip().split('\n')
            direction = "观望"
            reason = ""
            for line in lines:
                if "方向" in line:
                    if "看多" in line: direction = "看多"
                    elif "看空" in line: direction = "看空"
                elif "理由" in line:
                    reason = line.split("：")[-1].strip()
                    if len(reason) > 50: reason = reason[:50]
            
            analysis_results.append({"name": name, "direction": direction, "reason": reason})
            logger.info(f"      ✅ {direction}")
        except Exception as e:
            analysis_results.append({"name": name, "direction": "观望", "reason": f"分析失败"})
            logger.error(f"      ❌ {e}")
    
    # 4. 生成报告
    logger.info("\n[4] 生成报告...")
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    buy_list = [r for r in analysis_results if r["direction"] == "看多"]
    sell_list = [r for r in analysis_results if r["direction"] == "看空"]
    wait_list = [r for r in analysis_results if r["direction"] == "观望"]
    
    report = f"""# 📨 每日信号 - {today}

## 🎯 看多信号 ({len(buy_list)} 个)
"""
    for r in buy_list:
        report += f"- **{r['name']}**：{r['reason']}\n"
    
    report += f"""
## ⚠️ 看空信号 ({len(sell_list)} 个)
"""
    for r in sell_list:
        report += f"- **{r['name']}**：{r['reason']}\n"
    
    report += f"""
## 📊 观望 ({len(wait_list)} 个)
"""
    for r in wait_list[:5]:
        report += f"- {r['name']}\n"
    
    report += """
## 💡 操作建议
"""
    if buy_list and not sell_list:
        report += "整体偏多，可考虑买入。\n"
    elif sell_list and not buy_list:
        report += "整体偏空，可考虑卖出。\n"
    elif buy_list and sell_list:
        report += "信号分歧，保持均衡仓位。\n"
    else:
        report += "无明确信号，保持观望。\n"
    
    report += "\n⚠️ 本报告仅供参考，不构成投资建议。"
    
    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)
    
    # 5. 推送到微信
    logger.info("\n[5] 推送到微信...")
    title = f"📨 {today} 每日信号"
    push_to_wechat(title, report)
    
    logger.info("\n✅ 完成！")
    return report


if __name__ == "__main__":
    run_pipeline()
