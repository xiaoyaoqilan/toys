"""高质量决策引擎 - 整合多源信息生成决策。"""
from __future__ import annotations

import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, ".")

from loguru import logger

from crawler.finurls import FinurlsCrawler
from crawler.models import RawArticle, NormalizedFact
from config.settings import AppConf
from collections import defaultdict


class DecisionEngine:
    """高质量决策引擎。"""
    
    def __init__(self):
        self.conf = AppConf()
        self.crawler = FinurlsCrawler(self.conf)
        
    def run(self) -> Dict:
        """运行决策引擎。"""
        logger.info("=" * 70)
        logger.info("🎯 高质量决策引擎启动")
        logger.info("=" * 70)
        
        # Step 1: 获取大量真实新闻
        logger.info("\n[1] 获取多源财经新闻...")
        articles = self.crawler.crawl()
        logger.info(f"    获取到 {len(articles)} 篇新闻")
        
        # Step 2: 多源聚合与交叉验证
        logger.info("\n[2] 多源聚合与交叉验证...")
        topics = self._aggregate_by_topic(articles)
        logger.info(f"    识别出 {len(topics)} 个话题聚类")
        
        # Step 3: 按主题生成决策信息
        logger.info("\n[3] 生成主题决策信息...")
        topic_decisions = []
        
        for topic_id, topic_data in sorted(topics.items(), key=lambda x: -len(x[1]["sources"]))[:10]:
            decision = self._analyze_topic(topic_data)
            topic_decisions.append(decision)
        
        # Step 4: 综合决策
        logger.info("\n[4] 综合决策分析...")
        final_decision = self._synthesize_decision(topic_decisions)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "articles_count": len(articles),
            "topics_count": len(topics),
            "topic_decisions": topic_decisions,
            "final_decision": final_decision,
        }
    
    def _aggregate_by_topic(self, articles: List[RawArticle]) -> Dict:
        """按主题聚合文章（使用关键词+相似度）。"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not articles:
            return {}
        
        # 构建文本
        texts = [a.title.lower() for a in articles]
        
        # TF-IDF 向量化
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # 计算相似度
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # 聚类（单链接）
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
        
        # 高阈值聚类（0.5以上才聚合）
        threshold = 0.5
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > threshold:
                    union(i, j)
        
        # 分组
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)
        
        # 为每个聚类创建主题
        topics = {}
        for cluster_id, indices in clusters.items():
            if len(indices) < 1:
                continue
            
            group = [articles[i] for i in indices]
            sources = list(set(a.source for a in group))
            
            # 只保留有多个来源的主题
            if len(sources) >= 2:
                topics[f"topic_{cluster_id}"] = {
                    "articles": group,
                    "sources": sources,
                    "source_count": len(sources),
                    "titles": [a.title for a in group],
                    "main_title": max(group, key=lambda a: len(a.title)).title,
                }
        
        return topics
    
    def _analyze_topic(self, topic_data: Dict) -> Dict:
        """分析单个主题。"""
        sources = topic_data["sources"]
        source_count = len(sources)
        main_title = topic_data["main_title"]
        
        # 置信度评估
        if source_count >= 5:
            confidence = "极高"
            score = 0.95
        elif source_count >= 3:
            confidence = "高"
            score = 0.85
        elif source_count >= 2:
            confidence = "中"
            score = 0.70
        else:
            confidence = "低"
            score = 0.40
        
        # 主题分类
        category = self._categorize_topic(main_title)
        
        # 影响评估
        impact = self._assess_impact(main_title, category)
        
        return {
            "title": main_title,
            "category": category,
            "confidence": confidence,
            "score": score,
            "sources": sources,
            "source_count": source_count,
            "impact": impact,
        }
    
    def _categorize_topic(self, title: str) -> str:
        """主题分类。"""
        title_lower = title.lower()
        
        categories = {
            "货币政策/利率": ["fed", "rate", "inflation", "monetary", "fomc", "interest"],
            "贸易/关税": ["tariff", "trade", "import", "export", "sanction"],
            "地缘政治": ["war", "conflict", "sanction", "geopolitical", "military"],
            "科技/AI": ["ai", "chip", "semiconductor", "tech", "nvidia", "apple", "microsoft"],
            "能源/大宗商品": ["oil", "energy", "commodity", "gas", "coal", "metal"],
            "加密货币": ["bitcoin", "crypto", "ethereum", "blockchain"],
            "公司财报": ["earnings", "revenue", "profit", "quarterly", "annual"],
            "市场行情": ["stock", "market", "index", "s&p", "nasdaq", "dow"],
        }
        
        for cat, keywords in categories.items():
            if any(kw in title_lower for kw in keywords):
                return cat
        
        return "其他"
    
    def _assess_impact(self, title: str, category: str) -> Dict:
        """评估潜在影响。"""
        title_lower = title.lower()
        
        # 默认影响
        impact = {
            "rate_impact": "neutral",  # 对利率影响
            "equity_impact": "neutral",  # 对股市影响
            "sector_impact": {},  # 对具体板块影响
        }
        
        # 货币政策类
        if category == "货币政策/利率":
            if any(w in title_lower for w in ["hawk", "tight", "increase", "high"]):
                impact["rate_impact"] = "bearish"  # 利率上升
                impact["equity_impact"] = "bearish"  # 股市下跌
                impact["sector_impact"] = {
                    "positive": ["银行", "保险"],
                    "negative": ["科技", "成长股", "房地产"],
                }
            elif any(w in title_lower for w in ["dove", "ease", "cut", "low"]):
                impact["rate_impact"] = "bullish"  # 利率下降
                impact["equity_impact"] = "bullish"  # 股市上涨
                impact["sector_impact"] = {
                    "positive": ["科技", "成长股", "房地产"],
                    "negative": ["银行"],
                }
        
        # 贸易/关税类
        elif category == "贸易/关税":
            if any(w in title_lower for w in ["tariff", "sanction", "ban"]):
                impact["rate_impact"] = "uncertain"
                impact["equity_impact"] = "bearish"
                impact["sector_impact"] = {
                    "positive": ["本土替代品"],
                    "negative": ["出口企业", "相关供应链"],
                }
        
        # 能源类
        elif category == "能源/大宗商品":
            if any(w in title_lower for w in ["surge", "rise", "high", "increase"]):
                impact["sector_impact"] = {
                    "positive": ["能源", "矿业"],
                    "negative": ["航空", "运输", "制造业"],
                }
        
        return impact
    
    def _synthesize_decision(self, topic_decisions: List[Dict]) -> Dict:
        """综合决策。"""
        # 统计各类主题
        category_counts = defaultdict(int)
        category_impacts = defaultdict(list)
        
        for td in topic_decisions:
            cat = td["category"]
            category_counts[cat] += 1
            category_impacts[cat].append(td["impact"])
        
        # 计算整体倾向
        rate_impacts = []
        equity_impacts = []
        
        for td in topic_decisions:
            if td["confidence"] in ["极高", "高"]:
                rate_impacts.append(td["impact"]["rate_impact"])
                equity_impacts.append(td["impact"]["equity_impact"])
        
        # 决策逻辑
        decision = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topics_analyzed": len(topic_decisions),
            "high_confidence_topics": sum(1 for t in topic_decisions if t["confidence"] in ["极高", "高"]),
        }
        
        # 利率决策
        if rate_impacts:
            bearish_count = sum(1 for r in rate_impacts if r == "bearish")
            bullish_count = sum(1 for r in rate_impacts if r == "bullish")
            
            if bearish_count > bullish_count * 1.5:
                decision["rate_decision"] = {
                    "direction": "偏紧/看空债券",
                    "confidence": "中",
                    "reason": f"有 {bearish_count} 个高置信度事件指向利率上升"
                }
            elif bullish_count > bearish_count * 1.5:
                decision["rate_decision"] = {
                    "direction": "偏松/看多债券",
                    "confidence": "中",
                    "reason": f"有 {bullish_count} 个高置信度事件指向利率下降"
                }
            else:
                decision["rate_decision"] = {
                    "direction": "中性/观望",
                    "confidence": "低",
                    "reason": "多空因素均衡"
                }
        
        # 股市决策
        if equity_impacts:
            bearish_count = sum(1 for r in equity_impacts if r == "bearish")
            bullish_count = sum(1 for r in equity_impacts if r == "bullish")
            
            if bearish_count > bullish_count:
                decision["equity_decision"] = {
                    "direction": "谨慎/看空",
                    "confidence": "中",
                    "action": "降低仓位，规避高风险板块",
                    "reason": f"有 {bearish_count} 个高置信度事件指向股市下跌"
                }
            elif bullish_count > bearish_count:
                decision["equity_decision"] = {
                    "direction": "积极/看多",
                    "confidence": "中",
                    "action": "可以加仓，关注受益板块",
                    "reason": f"有 {bullish_count} 个高置信度事件指向股市上涨"
                }
            else:
                decision["equity_decision"] = {
                    "direction": "中性/观望",
                    "confidence": "低",
                    "action": "保持现有仓位",
                    "reason": "多空因素均衡，无明确方向"
                }
        
        # 板块建议
        all_positive = []
        all_negative = []
        
        for td in topic_decisions:
            if td["confidence"] in ["极高", "高"]:
                imp = td["impact"]["sector_impact"]
                if isinstance(imp, dict):
                    all_positive.extend(imp.get("positive", []))
                    all_negative.extend(imp.get("negative", []))
        
        decision["sector_recommendations"] = {
            "positive": list(set(all_positive))[:5],  # 去重取前5
            "negative": list(set(all_negative))[:5],
        }
        
        return decision


if __name__ == "__main__":
    engine = DecisionEngine()
    result = engine.run()
    
    print("\n" + "=" * 70)
    print("📊 决策引擎输出")
    print("=" * 70)
    
    print(f"\n📈 分析的话题数: {result['topics_count']}")
    print(f"🎯 高置信度话题: {result['final_decision']['high_confidence_topics']}")
    
    print("\n📋 决策详情:")
    decision = result["final_decision"]
    
    if "rate_decision" in decision:
        print(f"\n  利率决策: {decision['rate_decision']['direction']}")
        print(f"  原因: {decision['rate_decision']['reason']}")
    
    if "equity_decision" in decision:
        print(f"\n  股市决策: {decision['equity_decision']['direction']}")
        print(f"  操作建议: {decision['equity_decision']['action']}")
        print(f"  原因: {decision['equity_decision']['reason']}")
    
    if "sector_recommendations" in decision:
        print(f"\n  推荐板块: {decision['sector_recommendations']['positive']}")
        print(f"  规避板块: {decision['sector_recommendations']['negative']}")
