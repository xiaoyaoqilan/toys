"""定时调度器 - 主动分析与推送。"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from loguru import logger

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False

from rag import FinanceRAG, LLMClient
from notify import Notifier


class AnalysisScheduler:
    """定时分析与推送调度器。"""
    
    def __init__(self, rag: FinanceRAG, notifier: Optional[Notifier] = None):
        self.rag = rag
        self.notifier = notifier or Notifier()
        self.scheduler = None
        
        if _APSCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()
            logger.info("APScheduler 初始化成功")
        else:
            logger.warning("APScheduler 未安装，请手动安装: pip install apscheduler")

    def add_daily_task(
        self,
        hour: int = 9,
        minute: int = 0,
        queries: Optional[list] = None,
    ):
        """添加每日定时分析任务。"""
        if not self.scheduler:
            logger.error("调度器未初始化")
            return
            
        default_queries = [
            "Stock market today",
            "Bitcoin price",
            "Oil price trend",
            "Federal Reserve policy",
        ]
        queries = queries or default_queries
        
        self.scheduler.add_job(
            self._run_daily_analysis,
            trigger="cron",
            hour=hour,
            minute=minute,
            kwargs={"queries": queries},
            id="daily_financial_analysis",
            replace_existing=True,
        )
        logger.info(f"已添加每日分析任务: 每天 {hour}:{minute:02d} 执行")

    def start(self):
        """启动调度器。"""
        if self.scheduler:
            self.scheduler.start()
            logger.info("调度器已启动")

    def stop(self):
        """停止调度器。"""
        if self.scheduler:
            self.scheduler.shutdown()
            logger.info("调度器已停止")

    def _run_daily_analysis(self, queries: list):
        """执行每日分析并推送。"""
        logger.info("=" * 50)
        logger.info("开始每日财经分析...")
        logger.info("=" * 50)
        
        today = datetime.now().strftime("%Y-%m-%d")
        all_reports = []
        
        for query in queries:
            try:
                logger.info(f"分析: {query}")
                answer = self.rag.answer(query, top_k=3)
                all_reports.append({
                    "query": query,
                    "answer": answer.answer,
                    "confidence": answer.confidence,
                    "sources_count": len(answer.sources),
                })
                time.sleep(1)  # 避免 API 限流
            except Exception as e:
                logger.error(f"分析 '{query}' 失败: {e}")
                all_reports.append({
                    "query": query,
                    "answer": f"分析失败: {e}",
                    "confidence": "error",
                    "sources_count": 0,
                })

        # 生成汇总报告
        summary = self._generate_summary(today, all_reports)
        
        # 推送
        self._push_report(today, summary)
        
        logger.info("每日分析完成")

    def _generate_summary(self, date: str, reports: list) -> str:
        """生成汇总报告。"""
        lines = [
            f"## 📊 每日财经分析报告 ({date})",
            "",
            "---",
            "",
        ]
        
        for i, report in enumerate(reports, 1):
            lines.extend([
                f"### {i}. {report['query']}",
                "",
                f"**置信度**: {report['confidence']}",
                f"**相关事实数**: {report['sources_count']}",
                "",
                "**核心分析**:",
                "",
                report['answer'][:500],  # 截取前 500 字
                "",
                "---",
                "",
            ])
            
        lines.extend([
            "",
            f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])
        
        return "\n".join(lines)

    def _push_report(self, date: str, content: str):
        """推送报告。"""
        title = f"📊 财经日报 | {date}"
        
        try:
            results = self.notifier.send(title, content)
            logger.info(f"推送结果: {results}")
        except Exception as e:
            logger.error(f"推送失败: {e}")
            
        return results
