"""每日财经分析管道 - 爬取 → 去重 → 分析 → 推送。"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

from config.settings import AppConf
from crawler.finurls import FinurlsCrawler
from kb.faiss_kb import FaissKnowledgeBase
from kb.embeddings import Embedder
from rag.chain import FinanceRAG
from rag.llm_client import LLMClient
from crawler.models import NormalizedFact
from verifier.cross_source import CrossSourceVerifier
from notify.notifier import Notifier


class DailyPipeline:
    """每日财经分析管道。"""
    
    def __init__(self):
        self.conf = AppConf()
        self.crawler = FinurlsCrawler(self.conf)
        self.embedder = Embedder()
        self.kb = FaissKnowledgeBase(index_dir="data/index", embedder=self.embedder)
        self.llm = LLMClient(api_key=os.getenv("DEEPSEEK_API_KEY", ""))
        self.rag = FinanceRAG(kb=self.kb, embedder=self.embedder, llm=self.llm)
        self.verifier = CrossSourceVerifier()
        self.notifier = Notifier(
            serverchan_key=os.getenv("SERVERCHAN_KEY", ""),
            smtp_server=os.getenv("SMTP_SERVER", "smtp.qq.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "465")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            email_to=os.getenv("EMAIL_TO", ""),
        )
    
    def run(self) -> dict:
        """执行完整的每日管道。"""
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("📰 每日财经分析管道开始执行")
        logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        try:
            # 1. 清理旧数据
            logger.info("\n[1] 清理旧数据...")
            self.kb.clear()
            logger.info("    ✅ 已清空历史数据")
            
            # 2. 爬取新数据
            logger.info("\n[2] 爬取 finurls 财经新闻...")
            raw_articles = self.crawler.crawl()
            logger.info(f"    ✅ 爬取到 {len(raw_articles)} 篇文章")
            
            if not raw_articles:
                logger.warning("    ⚠️  未爬取到任何文章，检查网络或目标网站")
                return {"status": "empty", "articles": 0}
            
            # 3. 归一化和交叉验证
            logger.info("\n[3] 归一化和交叉验证...")
            normalized_facts = self._normalize_articles(raw_articles)
            logger.info(f"    ✅ 归一化为 {len(normalized_facts)} 条事实")
            
            # 4. 存储到知识库
            logger.info("\n[4] 存储到知识库...")
            self.kb.upsert_batch(normalized_facts)
            stats = self.kb.stats()
            logger.info(f"    ✅ 存储完成，统计: {stats}")
            
            # 5. 只分析高置信度事实
            logger.info("\n[5] 分析高置信度事实...")
            high_conf_facts = [f for f in normalized_facts if f.confidence == "high"]
            medium_conf_facts = [f for f in normalized_facts if f.confidence == "medium"]
            logger.info(f"    高置信度: {len(high_conf_facts)} 条")
            logger.info(f"    中置信度: {len(medium_conf_facts)} 条")
            
            # 6. 生成分析报告
            logger.info("\n[6] 生成分析报告...")
            report = self._generate_report(high_conf_facts, medium_conf_facts)
            logger.info(f"    ✅ 报告生成完成")
            
            # 7. 推送通知
            logger.info("\n[7] 推送通知...")
            push_result = self._push_report(report)
            logger.info(f"    推送结果: {push_result}")
            
            elapsed = time.time() - start_time
            logger.info("\n" + "=" * 70)
            logger.info(f"✅ 管道执行完成！耗时 {elapsed:.2f}s")
            logger.info("=" * 70)
            
            return {
                "status": "success",
                "articles": len(raw_articles),
                "facts": len(normalized_facts),
                "high_conf": len(high_conf_facts),
                "report": report,
                "push_result": push_result,
                "elapsed": elapsed,
            }
            
        except Exception as e:
            logger.error(f"❌ 管道执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    def _normalize_articles(self, articles: list) -> list:
        """将原始文章归一化为事实。"""
        from collections import defaultdict
        
        # 按标题聚合相似文章
        title_groups = defaultdict(list)
        
        for article in articles:
            # 使用简单的标题归一化（去除空格、标点）作为聚类键
            key = article.title.lower().strip()
            title_groups[key].append(article)
        
        facts = {}
        
        for key, group in title_groups.items():
            # 使用第一篇文章的标题作为标准标题
            main_article = group[0]
            fact_id = main_article.uid
            
            # 收集所有来源
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
            
            # 构建摘要
            content = main_article.content or main_article.title
            summary = content[:200] if len(content) > 200 else content
            
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
    
    def _generate_report(self, high_conf_facts: list, medium_conf_facts: list) -> str:
        """生成分析报告。"""
        today = datetime.now().strftime("%Y年%m月%d日")
        
        if not high_conf_facts and not medium_conf_facts:
            return f"""## 📊 {today} 财经日报

**今日无重大财经新闻。**

系统爬取了今日所有财经源，但未检测到高置信度（≥3家媒体报道）的重大事件。

建议关注以下低置信度新闻：
"""
        
        # 优先分析高置信度事实
        analysis_facts = high_conf_facts if high_conf_facts else medium_conf_facts
        confidence_label = "高置信度" if high_conf_facts else "中置信度"
        
        # 用 LLM 生成分析
        context = self.kb._build_context(analysis_facts)
        
        prompt = f"""你是一位专业的财经分析师。请根据以下{confidence_label}财经事实，生成一份简洁的每日分析报告。

要求：
1. 提炼核心观点，不要逐条罗列
2. 分析这些事实之间的关联性和潜在影响
3. 给出2-3个值得关注的要点
4. 控制在300字以内

事实列表：
{context}

请用中文回答，格式为Markdown。"""
        
        if self.llm.is_available:
            try:
                analysis = self.llm._call_llm(prompt, max_tokens=500)
            except Exception as e:
                logger.warning(f"LLM 调用失败: {e}, 使用模板")
                analysis = self._template_analysis(analysis_facts, confidence_label)
        else:
            analysis = self._template_analysis(analysis_facts, confidence_label)
        
        # 组装完整报告
        report = f"""## 📊 {today} 财经日报

### 🎯 今日要点（{confidence_label}）

{analysis}

---

### 📈 事实来源（Top 5）

"""
        
        for i, fact in enumerate(analysis_facts[:5], 1):
            sources_str = "、".join(fact.sources)
            report += f"{i}. **{fact.canonical_title[:60]}**\n   - 来源: {sources_str}\n   - 置信度: {fact.confidence}\n\n"
        
        report += f"""---

📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔍 分析模式: {confidence_label}优先
"""
        
        return report
    
    def _template_analysis(self, facts: list, confidence_label: str) -> str:
        """模板分析（无 LLM 时使用）。"""
        if not facts:
            return "今日无重大财经新闻。"
        
        points = []
        for fact in facts[:3]:
            points.append(f"- **{fact.canonical_title[:50]}**: {fact.canonical_summary[:80]}...")
        
        return f"""基于{confidence_label}事实的简要分析：

{chr(10).join(points)}

这些事实反映了当前市场的主要关注点，建议持续追踪后续发展。"""
    
    def _push_report(self, report: str) -> dict:
        """推送报告到所有已配置的渠道。"""
        today = datetime.now().strftime("%Y-%m-%d")
        title = f"📊 {today} 财经日报 | 每日自动分析"
        
        # 使用统一的 send 方法
        results = self.notifier.send(title, report)
        
        # 如果没有配置任何渠道，保存到本地
        if results == {} or all(v == "未配置" for v in results.values()):
            logger.warning("⚠️  未配置任何推送渠道，报告仅保存到本地")
            report_file = f"reports/report_{today}.md"
            os.makedirs("reports", exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            results["local_file"] = f"已保存到 {report_file}"
        
        return results


def main():
    """主入口。"""
    logger.info("🚀 启动每日财经分析管道...")
    
    pipeline = DailyPipeline()
    result = pipeline.run()
    
    if result["status"] == "success":
        logger.info("\n🎉 管道执行成功！")
        logger.info(f"   - 爬取文章: {result['articles']} 篇")
        logger.info(f"   - 归一化事实: {result['facts']} 条")
        logger.info(f"   - 高置信度事实: {result['high_conf']} 条")
        logger.info(f"   - 推送结果: {result['push_result']}")
    else:
        logger.error(f"\n❌ 管道执行失败: {result}")
    
    return result


if __name__ == "__main__":
    main()
