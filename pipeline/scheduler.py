"""定时任务调度器 - 每天9点自动运行管道。"""
from __future__ import annotations

import os
import sys
import time
import signal
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

from pipeline.daily_pipeline import DailyPipeline


class Scheduler:
    """定时任务调度器。"""
    
    def __init__(self, run_at_hour: int = 9, run_at_minute: int = 0):
        self.run_at_hour = run_at_hour
        self.run_at_minute = run_at_minute
        self.pipeline = DailyPipeline()
        self.running = True
        self.last_run = None
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info("\n🛑 收到停止信号，正在退出...")
        self.running = False
    
    def _should_run_now(self) -> bool:
        """检查是否到了运行时间。"""
        now = datetime.now()
        if now.hour == self.run_at_hour and now.minute == self.run_at_minute:
            if self.last_run is None or (now.date() != self.last_run.date()):
                return True
        return False
    
    def _next_run_time(self) -> datetime:
        """计算下次运行时间。"""
        now = datetime.now()
        target = now.replace(hour=self.run_at_hour, minute=self.run_at_minute, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        return target
    
    def run_once(self):
        """立即运行一次（用于测试）。"""
        logger.info("▶️  立即执行一次管道...")
        self.last_run = datetime.now()
        return self.pipeline.run()
    
    def start(self):
        """启动调度器（持续运行）。"""
        logger.info("=" * 70)
        logger.info(f"🕐 定时任务调度器已启动")
        logger.info(f"   执行时间: 每天 {self.run_at_hour:02d}:{self.run_at_minute:02d}")
        logger.info(f"   下次运行: {self._next_run_time().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   按 Ctrl+C 停止")
        logger.info("=" * 70)
        
        while self.running:
            try:
                now = datetime.now()
                
                # 每分钟检查一次
                if self._should_run_now():
                    logger.info(f"\n⏰ 到达执行时间 {now.strftime('%H:%M:%S')}")
                    self.last_run = now
                    try:
                        result = self.pipeline.run()
                        logger.info(f"✅ 执行完成: {result['status']}")
                    except Exception as e:
                        logger.error(f"❌ 执行失败: {e}")
                
                # 显示状态（每小时一次）
                if now.minute == 0 and now.second < 5:
                    next_run = self._next_run_time()
                    remaining = next_run - now
                    hours, remainder = divmod(int(remaining.total_seconds()), 3600)
                    minutes, _ = divmod(remainder, 60)
                    logger.debug(f"⏳ 系统运行中，下次执行还有 {hours}小时{minutes}分钟")
                
                time.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"调度器异常: {e}")
                time.sleep(60)
        
        logger.info("👋 调度器已停止")


def main():
    """主入口。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="财经分析系统调度器")
    parser.add_argument("--run-now", action="store_true", help="立即运行一次")
    parser.add_argument("--hour", type=int, default=9, help="运行小时 (默认: 9)")
    parser.add_argument("--minute", type=int, default=0, help="运行分钟 (默认: 0)")
    parser.add_argument("--daemon", action="store_true", help="作为守护进程持续运行")
    
    args = parser.parse_args()
    
    scheduler = Scheduler(run_at_hour=args.hour, run_at_minute=args.minute)
    
    if args.run_now:
        result = scheduler.run_once()
        print("\n" + "=" * 70)
        print("📋 执行结果摘要:")
        print("=" * 70)
        if result["status"] == "success":
            print(f"  爬取文章: {result['articles']} 篇")
            print(f"  归一化事实: {result['facts']} 条")
            print(f"  高置信度事实: {result['high_conf']} 条")
            print(f"  推送结果: {result['push_result']}")
            print(f"\n✅ 报告预览:")
            print(result['report'][:500] + "...")
        else:
            print(f"  ❌ 状态: {result['status']}")
            if 'error' in result:
                print(f"  错误: {result['error']}")
        print("=" * 70)
    elif args.daemon:
        scheduler.start()
    else:
        parser.print_help()
        print("\n💡 提示: 使用 --run-now 立即执行，或 --daemon 持续运行")


if __name__ == "__main__":
    main()
