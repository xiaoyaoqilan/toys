"""完整演示测试 - 爬取+分析+推送全流程。"""
from __future__ import annotations

import os
import sys
import time
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

from config.settings import AppConf
from crawler.finurls import FinurlsCrawler, crawl_and_fetch_content
from crawler.models import RawArticle, NormalizedFact
from kb.embeddings import Embedder
from kb.faiss_kb import FaissKnowledgeBase
from rag.llm_client import LLMClient
from notify.notifier import Notifier
from collections import defaultdict


def test_full_pipeline():
    """测试完整管道。"""
    logger.info("=" * 70)
    logger.info("🎯 完整管道演示测试")
    logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # 1. 初始化组件
    logger.info("\n[1] 初始化组件...")
    conf = AppConf()
    crawler = FinurlsCrawler(conf)
    embedder = Embedder()
    kb = FaissKnowledgeBase(index_dir="data/demo_index", embedder=embedder)
    llm = LLMClient(api_key=os.getenv("DEEPSEEK_API_KEY", ""))
    notifier = Notifier(
        serverchan_key=os.getenv("SERVERCHAN_KEY", ""),
        smtp_server=os.getenv("SMTP_SERVER", "smtp.qq.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "465")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        email_to=os.getenv("EMAIL_TO", ""),
    )
    logger.info("    ✅ 所有组件初始化完成")
    
    # 2. 尝试爬取真实数据
    logger.info("\n[2] 爬取 finurls 财经新闻...")
    try:
        raw_articles = crawler.crawl()
        logger.info(f"    爬取到 {len(raw_articles)} 篇文章")
    except Exception as e:
        logger.warning(f"    真实爬取失败: {e}")
        raw_articles = []
    
    # 如果没爬到数据，使用模拟数据演示
    if not raw_articles:
        logger.info("\n    ℹ️  无真实数据，使用模拟演示数据\n")
        raw_articles = generate_demo_articles()
    
    logger.info(f"    ✅ 准备 {len(raw_articles)} 篇文章进行处理")
    
    # 3. 归一化和交叉验证
    logger.info("\n[3] 归一化和交叉验证...")
    normalized_facts = normalize_articles(raw_articles)
    
    high_conf = [f for f in normalized_facts if f.confidence == "high"]
    medium_conf = [f for f in normalized_facts if f.confidence == "medium"]
    low_conf = [f for f in normalized_facts if f.confidence == "low"]
    
    logger.info(f"    总事实数: {len(normalized_facts)}")
    logger.info(f"    高置信度 (≥3源): {len(high_conf)}")
    logger.info(f"    中置信度 (2源): {len(medium_conf)}")
    logger.info(f"    低置信度 (1源): {len(low_conf)}")
    
    # 4. 存储到知识库
    logger.info("\n[4] 存储到 Faiss 知识库...")
    kb.upsert_batch(normalized_facts)
    stats = kb.stats()
    logger.info(f"    ✅ 存储完成")
    logger.info(f"    📊 统计: {stats}")
    
    # 5. 分析高置信度事实
    logger.info("\n[5] 分析高置信度事实...")
    high_conf = [f for f in normalized_facts if f.confidence == "high"]
    medium_conf = [f for f in normalized_facts if f.confidence == "medium"]
    low_conf = [f for f in normalized_facts if f.confidence == "low"]
    
    logger.info(f"    总事实数: {len(normalized_facts)}")
    logger.info(f"    高置信度 (≥3源): {len(high_conf)}")
    logger.info(f"    中置信度 (2源): {len(medium_conf)}")
    logger.info(f"    低置信度 (1源): {len(low_conf)}")
    
    # 选择分析的事实：优先高置信度，然后中置信度，最后低置信度Top10
    if high_conf:
        analysis_facts = high_conf
        confidence_label = "高置信度"
    elif medium_conf:
        analysis_facts = medium_conf
        confidence_label = "中置信度"
    else:
        # 从低置信度中选 Top 10 用于参考
        low_conf_sorted = sorted(low_conf, key=lambda x: x.score, reverse=True)[:10]
        analysis_facts = low_conf_sorted
        confidence_label = "低置信度"
    
    logger.info(f"\n    将分析 {len(analysis_facts)} 条 {confidence_label} 事实")
    for i, fact in enumerate(analysis_facts[:5], 1):
        sources_str = "、".join(fact.sources)
        logger.info(f"    [{i}] {fact.canonical_title[:60]}...")
        logger.info(f"        来源: {sources_str}")
        logger.info(f"        置信度: {fact.confidence}, 分数: {fact.score:.2f}")
    
    # 6. 用 LLM 生成严谨的分析报告
    logger.info("\n[6] LLM 生成严谨分析报告...")
    today = datetime.now().strftime("%Y年%m月%d日")
    
    if llm.is_available and analysis_facts:
        try:
            fact_lines = []
            for i, fact in enumerate(analysis_facts[:15], 1):
                sources_str = "/".join(fact.sources[:3])
                fact_lines.append(f"事实{i}: {fact.canonical_title}")
                fact_lines.append(f"  信源: {sources_str}")
                fact_lines.append(f"  置信度: {fact.confidence}")
                fact_lines.append(f"  摘要: {fact.canonical_summary[:200]}")
                fact_lines.append("")
            facts_text = "\n".join(fact_lines)
            
            system_prompt = """你是一位严谨的金融市场分析师，必须遵守以下原则：
1. **区分事实与传闻**：明确哪些是已发生的事实，哪些只是表态/威胁/传闻
2. **区分事件性质**：政策表态≠政策实施，口头威胁≠实际行动
3. **逻辑自洽**：鹰派（偏紧货币政策）→ 看空债券、看空成长股；鸽派→看多债券
4. **保守原则**：信息不足时保持观望，不要过度解读
5. **不做预测**：只分析已知事实的潜在影响，不预测未来"""
            
            user_prompt = f"""请基于以下**已验证的事实**，生成一份**严谨的市场分析**。

# 日期：{today}
# 可用事实：
{facts_text}

# 请按以下结构输出：

## 一、事实梳理（先明确已知条件）
> 【已确认事实】：...
> 【表态/威胁/传闻】：...
> 【信息缺口】：还缺少什么信息才能下确定结论？

## 二、事件定性（判断事件性质）
- 事件类型：[政策表态/实际政策/地缘事件/公司事件]
- 影响级别：[短期情绪/中期趋势/长期格局]
- 确定性：[高/中/低]

## 三、逻辑分析（保持严谨）
### 1. 对利率/债市的影响
- 如果是鹰派表态（偏紧）→ 理论上看空债券
- 但注意：如果只是口头表态，市场可能已经price in

### 2. 对权益市场的影响
- 高利率环境 → 看空成长股（估值压缩）
- 贸易威胁 → 对相关板块形成不确定性

### 3. 具体影响范围
（严格基于事实，不要过度延伸）

## 四、操作建议（保持保守）
> ⚠️ 由于事实有限，以下建议仅供参考：

| 方向 | 标的 | 理由 | 确定性 |
|------|------|------|--------|
| 观望为主 | 大盘 | 信息不足以判断趋势 | ⚪低 |

## 五、风险提示
1. 过度解读单一事件的风险
2. 事实与市场预期可能存在差异
3. 建议等待更多信号确认

---
**重要**：
- 如果事实不足，请明确说"信息不足，建议观望"
- 不要基于单条新闻给出激进的交易建议
- 保持逻辑一致性"""
            
            logger.info("    正在调用 LLM 生成严谨分析...")
            analysis = llm.chat(system_prompt, user_prompt, temperature=0.2, max_tokens=2000)
            if analysis and len(analysis) > 100:
                logger.info(f"    ✅ LLM 生成完成，共 {len(analysis)} 字符")
            else:
                raise Exception("LLM 返回内容过短")
        except Exception as e:
            logger.warning(f"    LLM 调用失败: {e}")
            analysis = generate_analysis_from_facts(analysis_facts, confidence_label)
    else:
        logger.info("    ℹ️  LLM 不可用，使用规则分析")
        analysis = generate_analysis_from_facts(analysis_facts, confidence_label)
    
    # 7. 组装完整报告
    logger.info("\n[7] 组装完整报告...")
    report = assemble_report(today, analysis, analysis_facts, confidence_label)
    logger.info(f"    ✅ 报告组装完成，共 {len(report)} 字符")
    
    # 8. 推送通知
    logger.info("\n[8] 推送通知...")
    title = f"📊 {today} 财经日报 | {len(high_conf)}条高置信度事实"
    push_results = notifier.send(title, report)
    
    if not push_results or all(v == "未配置" for v in push_results.values()):
        logger.warning("    ⚠️  未配置推送渠道，保存到本地")
        os.makedirs("reports", exist_ok=True)
        report_file = f"reports/report_{datetime.now().strftime('%Y-%m-%d')}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        push_results["local"] = f"已保存到 {report_file}"
    
    for channel, result in push_results.items():
        status = "✅" if result == "成功" else "📄"
        logger.info(f"    {status} {channel}: {result}")
    
    elapsed = time.time() - start_time
    
    # 9. 打印最终报告
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ 管道执行完成！耗时 {elapsed:.2f}s")
    logger.info("=" * 70)
    
    logger.info("\n" + "=" * 70)
    logger.info("📋 最终分析报告预览")
    logger.info("=" * 70)
    print("\n" + report)
    logger.info("=" * 70)
    
    return {
        "status": "success",
        "articles": len(raw_articles),
        "facts": len(normalized_facts),
        "high_conf": len(high_conf),
        "report": report,
        "push_results": push_results,
        "elapsed": elapsed,
    }


def generate_demo_articles():
    """生成演示文章（模拟多信源报道）。"""
    demo_data = [
        # 事件1: Apple 涨价（多源交叉验证）
        {
            "title": "Apple raises iPhone prices amid global chip shortage",
            "sources": ["reuters", "bloomberg", "wsj", "cnbc"],
            "content": "Apple Inc. announced today that it will increase iPhone prices by 10-15% due to the ongoing global semiconductor shortage. The company cited rising component costs and supply chain disruptions as the main reasons for the price adjustment.",
        },
        {
            "title": "Apple increases iPhone prices by 15% on chip shortage",
            "sources": ["reuters", "bloomberg", "wsj"],
            "content": "Apple has confirmed a significant price hike for its iPhone lineup, blaming the global chip shortage that has plagued the tech industry for over a year.",
        },
        # 事件2: 油价上涨（多源交叉验证）
        {
            "title": "Global oil prices surge 15% on supply concerns",
            "sources": ["cnbc", "wsj", "ft", "reuters"],
            "content": "Oil prices have surged by 15% in the past week amid growing concerns about global supply constraints. Brent crude futures hit $95 per barrel, the highest level in 2024.",
        },
        {
            "title": "Oil prices jump 12% on Middle East tensions",
            "sources": ["cnbc", "wsj", "ft"],
            "content": "Crude oil prices spiked sharply following escalating tensions in the Middle East, raising fears of supply disruptions in the region.",
        },
        # 事件3: 芯片行业困境（多源交叉验证）
        {
            "title": "Semiconductor industry faces global shortage",
            "sources": ["reuters", "bloomberg", "cnbc"],
            "content": "The global semiconductor industry is grappling with an unprecedented shortage that has affected everything from smartphones to automobiles. Major chipmakers are investing billions in new capacity.",
        },
        # 事件4: AI 科技股上涨（多源交叉验证）
        {
            "title": "Tech stocks lead market rally on AI optimism",
            "sources": ["bloomberg", "cnbc", "marketwatch"],
            "content": "Technology stocks have led a broad market rally as investors bet on the transformative potential of artificial intelligence. The Nasdaq Composite hit a record high.",
        },
        # 事件5: 美联储政策（2源验证）
        {
            "title": "Federal Reserve signals potential rate cut",
            "sources": ["reuters", "ft"],
            "content": "Federal Reserve Chair Jerome Powell hinted at a possible interest rate cut in the coming months, signaling a shift in monetary policy to support economic growth.",
        },
        # 事件6: 单独来源事件
        {
            "title": "Tesla announces new Gigafactory in Mexico",
            "sources": ["electrek"],
            "content": "Tesla has officially announced plans to build a new Gigafactory in Mexico, expected to begin production in 2025.",
        },
        {
            "title": "Bitcoin surges past $70,000",
            "sources": ["coindesk"],
            "content": "Bitcoin has surged past the $70,000 mark for the first time in months, driven by renewed institutional interest and ETF inflows.",
        },
    ]
    
    articles = []
    for i, item in enumerate(demo_data):
        for j, source in enumerate(item["sources"]):
            articles.append(RawArticle(
                url=f"https://example.com/{i}-{j}",
                title=item["title"],
                source=source,
                source_display=source.upper(),
                content=item["content"],
                published_at=datetime.now(),
            ))
    
    return articles


def normalize_articles(articles: list) -> list:
    """归一化文章 - 使用 TF-IDF 相似度聚合相似报道。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    if not articles:
        return []
    
    # 1. 计算文章标题的 TF-IDF 向量
    titles = [a.title.lower() for a in articles]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(titles)
    
    # 2. 计算相似度矩阵
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # 3. 使用简单的聚类算法（单链接聚类）
    n = len(articles)
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 4. 聚合相似度 > 0.4 的文章
    threshold = 0.4
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] > threshold:
                union(i, j)
    
    # 5. 按聚类分组
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    
    # 6. 为每个聚类创建一个事实
    facts = {}
    
    for cluster_id, indices in clusters.items():
        group = [articles[i] for i in indices]
        
        # 选择最长的标题作为规范标题
        main_article = max(group, key=lambda a: len(a.title))
        fact_id = main_article.uid
        
        # 收集所有不同的来源
        sources = list(set(a.source for a in group))
        source_count = len(sources)
        
        # 置信度评估
        if source_count >= 3:
            confidence = "high"
            score = 0.9
        elif source_count >= 2:
            confidence = "medium"
            score = 0.7
        else:
            confidence = "low"
            score = 0.4
        
        # 构建摘要（使用最长的内容）
        contents = [a.content for a in group if a.content]
        if contents:
            summary = max(contents, key=len)[:200]
        else:
            summary = main_article.title
        
        facts[fact_id] = NormalizedFact(
            fact_id=fact_id,
            canonical_title=main_article.title,
            canonical_summary=summary,
            sources=sources,
            source_count=source_count,
            confidence=confidence,
            score=score,
        )
    
    return list(facts.values())


def generate_analysis_from_facts(facts: list, confidence_label: str) -> str:
    """基于事实生成分析（规则版，无 LLM 时使用）。"""
    if not facts:
        return "今日无重大财经新闻。"
    
    # 提取关键词和主题
    themes = {}
    for fact in facts:
        title_lower = fact.canonical_title.lower()
        
        # 简单的主题分类
        if any(w in title_lower for w in ['inflation', 'fed', 'rate', 'monetary', '政策', '通胀']):
            theme = '宏观政策'
        elif any(w in title_lower for w in ['oil', 'energy', 'commodity', '原油', '能源']):
            theme = '大宗商品'
        elif any(w in title_lower for w in ['tech', 'ai', 'chip', 'semiconductor', '科技', '芯片']):
            theme = '科技行业'
        elif any(w in title_lower for w in ['tesla', 'spacex', 'musk', '特斯拉']):
            theme = '新能源汽车'
        elif any(w in title_lower for w in ['stock', 'market', '指数', '股市']):
            theme = '市场行情'
        elif any(w in title_lower for w in ['crypto', 'bitcoin', '加密', '比特币']):
            theme = '加密货币'
        else:
            theme = '其他'
        
        if theme not in themes:
            themes[theme] = []
        themes[theme].append(fact)
    
    # 生成分析
    analysis_parts = []
    analysis_parts.append("## 一、市场主线\n")
    
    main_points = []
    for theme, theme_facts in sorted(themes.items(), key=lambda x: -len(x[1])):
        for fact in theme_facts[:1]:
            main_points.append(f"- **{theme}**: {fact.canonical_title[:80]}")
    
    analysis_parts.append("\n".join(main_points[:3]))
    analysis_parts.append("\n")
    
    analysis_parts.append("## 二、深度分析\n")
    
    for theme, theme_facts in themes.items():
        analysis_parts.append(f"### {theme}")
        for fact in theme_facts[:2]:
            sources = "/".join(fact.sources[:3])
            analysis_parts.append(f"- {fact.canonical_title[:100]} ({sources})")
        analysis_parts.append("")
    
    analysis_parts.append("## 三、投资建议\n")
    
    # 基于主题生成建议
    if '宏观政策' in themes:
        analysis_parts.append("- **关注**: 货币政策走向影响利率敏感板块")
    if '科技行业' in themes:
        analysis_parts.append("- **关注**: 科技板块短期波动机会")
    if '大宗商品' in themes:
        analysis_parts.append("- **关注**: 能源板块受益于价格上涨")
    
    analysis_parts.append("\n## 四、风险提示\n")
    analysis_parts.append("- 市场波动可能加剧，注意仓位管理")
    analysis_parts.append("- 信息置信度有限，需进一步验证")
    
    return "\n".join(analysis_parts)


def assemble_report(today: str, analysis: str, facts: list, confidence_label: str) -> str:
    """组装完整报告。"""
    report = f"""# 📊 {today} 财经日报

## 🎯 今日要点（{confidence_label}）

{analysis}

---

## 📈 高置信度事实详情

"""
    
    for i, fact in enumerate(facts, 1):
        sources_str = "、".join(fact.sources)
        report += f"""### {i}. {fact.canonical_title}

- **置信度**: {fact.confidence} (分数: {fact.score:.2f})
- **信源**: {sources_str}
- **摘要**: {fact.canonical_summary[:150]}

"""
    
    report += f"""---

📅 **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔍 **分析模式**: 仅分析{confidence_label}事实
🤖 **技术**: TF-IDF + Faiss + LLM
"""
    
    return report


if __name__ == "__main__":
    test_full_pipeline()
