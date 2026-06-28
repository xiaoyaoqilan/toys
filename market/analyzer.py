"""市场分析器 - Binance(加密) + yfinance(美股) + 布林带。"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

STOCK_SYMBOLS = ["TSLA", "AAPL", "MSFT", "NVDA", "GOOGL", "QQQ", "SPY", "XLV", "GLD", "TLT"]


class MarketAnalyzer:
    """市场分析器。"""
    
    def __init__(self):
        self.binance_client = None
        self.yf_available = YFINANCE_AVAILABLE
        
        if BINANCE_AVAILABLE:
            try:
                self.binance_client = Client("", "")
            except Exception:
                self.binance_client = None
    
    def calculate_bollinger_bands(self, closes: List[float], period: int = 20, std_dev: float = 2.0) -> Dict:
        """计算布林带 - 固定思路：下轨做多，上轨看空。"""
        if len(closes) < period:
            return {"error": "数据不足"}
        
        closes_arr = np.array(closes[-period:])
        current_price = closes[-1]
        
        middle = np.mean(closes_arr)
        std = np.std(closes_arr)
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        # 固定交易思路
        if current_price <= lower:
            signal = "🟢 下轨附近 → 做多"
            action = "买入/做多"
        elif current_price >= upper:
            signal = "🔴 上轨附近 → 看空"
            action = "卖出/做空"
        else:
            signal = "⚪ 区间震荡 → 观望"
            action = "观望"
        
        return {
            "price": round(current_price, 2),
            "upper": round(upper, 2),
            "middle": round(middle, 2),
            "lower": round(lower, 2),
            "signal": signal,
            "action": action,
        }
    
    def analyze_crypto(self, symbol: str) -> Dict:
        """分析加密货币。"""
        result = {"symbol": symbol, "type": "crypto"}
        
        if not self.binance_client:
            result["error"] = "Binance 不可用"
            return result
        
        try:
            ticker = self.binance_client.get_ticker(symbol=symbol)
            result["price"] = float(ticker["lastPrice"])
            result["change_24h"] = float(ticker["priceChangePercent"])
        except Exception as e:
            result["error"] = str(e)
            return result
        
        try:
            klines = self.binance_client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_4HOUR,
                limit=50
            )
            closes = [float(k[4]) for k in klines]
            result["bollinger"] = self.calculate_bollinger_bands(closes)
        except Exception as e:
            result["error"] = f"K线获取失败: {e}"
        
        return result
    
    def analyze_stock(self, symbol: str) -> Dict:
        """分析美股。"""
        result = {"symbol": symbol, "type": "stock"}
        
        if not self.yf_available:
            result["error"] = "yfinance 不可用"
            return result
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                result["error"] = "无历史数据"
                return result
            
            # 去除 NaN 值
            hist = hist.dropna(subset=["Close"])
            
            if len(hist) < 20:
                result["error"] = "历史数据不足"
                return result
            
            closes = hist["Close"].values.tolist()
            current = closes[-1]
            prev = closes[-2]
            
            result["price"] = round(float(current), 2)
            result["change_1d"] = round((current - prev) / prev * 100, 2)
            
            result["bollinger"] = self.calculate_bollinger_bands(closes)
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_all_analysis(self) -> Dict:
        """获取所有分析。"""
        results = {
            "crypto": [],
            "stocks": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        logger.info("分析加密货币...")
        for sym in CRYPTO_SYMBOLS:
            results["crypto"].append(self.analyze_crypto(sym))
        
        logger.info("分析美股...")
        for sym in STOCK_SYMBOLS:
            results["stocks"].append(self.analyze_stock(sym))
        
        return results
    
    def format_report(self, data: Dict) -> str:
        """格式化报告。"""
        lines = []
        lines.append(f"# 📊 市场分析报告 - {data['timestamp']}")
        lines.append("")
        lines.append("**交易规则：布林带下轨做多，上轨看空**")
        lines.append("")
        
        # 加密货币
        lines.append("## 🔴 加密货币（Binance 4H）\n")
        for item in data["crypto"]:
            if "error" in item:
                continue
            
            bb = item.get("bollinger", {})
            price = item.get("price", 0)
            change = item.get("change_24h", 0)
            
            change_icon = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            lines.append(f"### {item['symbol']}")
            lines.append(f"- **价格**: ${price:,.2f} ({change_icon} {change:+.2f}%)")
            
            if "error" not in bb:
                lines.append(f"- **布林带(4H)**: 上${bb['upper']:,.2f} | 中${bb['middle']:,.2f} | 下${bb['lower']:,.2f}")
                lines.append(f"- **信号**: {bb.get('signal', 'N/A')}")
                lines.append(f"- **操作**: {bb.get('action', 'N/A')}")
            
            lines.append("")
        
        # 美股
        lines.append("## 🟢 美股（日线）\n")
        for item in data["stocks"]:
            if "error" in item:
                continue
            
            bb = item.get("bollinger", {})
            price = item.get("price", 0)
            change = item.get("change_1d", 0)
            
            change_icon = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            lines.append(f"### {item['symbol']}")
            lines.append(f"- **价格**: ${price:,.2f} ({change_icon} {change:+.2f}%)")
            
            if "error" not in bb:
                lines.append(f"- **布林带(日线)**: 上${bb['upper']:,.2f} | 中${bb['middle']:,.2f} | 下${bb['lower']:,.2f}")
                lines.append(f"- **信号**: {bb.get('signal', 'N/A')}")
                lines.append(f"- **操作**: {bb.get('action', 'N/A')}")
            
            lines.append("")
        
        return "\n".join(lines)


if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    data = analyzer.get_all_analysis()
    print(analyzer.format_report(data))
