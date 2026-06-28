"""Binance 实时行情 + 布林带分析。"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
from loguru import logger

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logger.warning("python-binance 未安装")


# 关注的交易对（加密货币 + 美股）
SYMBOLS = {
    "crypto": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
        "MATICUSDT", "LTCUSDT", "SHIBUSDT", "UNIUSDT", "ATOMUSDT",
    ],
    "us_stocks": [
        "TSLA", "AAPL", "MSFT", "NVDA", "GOOGL",
        "AMZN", "META", "JPM", "BAC", "GS",
        "XOM", "CVX", "GLD", "TLT", "DXY",
        "QQQ", "SPY", "XLV", "XLE", "XLK",
    ],
}


class BinanceAnalyzer:
    """Binance 行情分析器。"""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.available = BINANCE_AVAILABLE
        if self.available:
            try:
                self.client = Client(api_key, api_secret)
            except Exception as e:
                logger.error(f"Binance 客户端初始化失败: {e}")
                self.available = False
                self.client = None
        else:
            self.client = None
    
    def get_ticker_24h(self, symbol: str) -> Dict:
        """获取24小时行情。"""
        if not self.available:
            return {"error": "Binance 不可用"}
        
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return {
                "symbol": symbol,
                "price": float(ticker["lastPrice"]),
                "change_24h": float(ticker["priceChangePercent"]),
                "high_24h": float(ticker["highPrice"]),
                "low_24h": float(ticker["lowPrice"]),
                "volume_24h": float(ticker["volume"]),
                "quote_volume_24h": float(ticker["quoteVolume"]),
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}
    
    def get_klines_4h(self, symbol: str, limit: int = 50) -> List[Dict]:
        """获取4小时K线数据。"""
        if not self.available:
            return []
        
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_4HOUR,
                limit=limit
            )
            
            result = []
            for k in klines:
                result.append({
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                })
            
            return result
        except Exception as e:
            logger.error(f"获取 {symbol} K线失败: {e}")
            return []
    
    def calculate_bollinger_bands(self, klines: List[Dict], period: int = 20, std_dev: float = 2.0) -> Dict:
        """计算布林带。"""
        if len(klines) < period:
            return {"error": "K线数据不足"}
        
        closes = np.array([k["close"] for k in klines[-period:]])
        current_price = klines[-1]["close"]
        
        middle = np.mean(closes)
        std = np.std(closes)
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        # 信号判断
        if current_price > upper:
            signal = "⚠️ 超买，可能回调"
            position = "above_upper"
        elif current_price < lower:
            signal = "✅ 超卖，可能反弹"
            position = "below_lower"
        elif current_price > middle:
            signal = "📈 中轨上方，偏多"
            position = "above_middle"
        else:
            signal = "📉 中轨下方，偏空"
            position = "below_middle"
        
        # 带宽计算
        bandwidth = (upper - lower) / middle * 100
        
        return {
            "current_price": current_price,
            "middle": round(middle, 2),
            "upper": round(upper, 2),
            "lower": round(lower, 2),
            "bandwidth": round(bandwidth, 2),
            "signal": signal,
            "position": position,
        }
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """完整分析单个币种。"""
        result = {"symbol": symbol}
        
        # 24h 行情
        ticker = self.get_ticker_24h(symbol)
        if "error" in ticker:
            return result
        
        result["ticker"] = ticker
        
        # 4h K线 + 布林带
        klines = self.get_klines_4h(symbol, limit=50)
        if klines:
            bb = self.calculate_bollinger_bands(klines)
            result["bollinger"] = bb
        
        return result
    
    def get_full_report(self) -> List[Dict]:
        """生成完整报告。"""
        results = []
        
        for symbol in SYMBOLS:
            logger.info(f"    分析 {symbol}...")
            result = self.analyze_symbol(symbol)
            results.append(result)
        
        return results
    
    def format_report(self, results: List[Dict]) -> str:
        """格式化报告。"""
        lines = []
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"# 📈 Binance 技术分析报告 - {today}")
        lines.append("")
        
        for r in results:
            if "error" in r:
                continue
            
            symbol = r["symbol"]
            ticker = r.get("ticker", {})
            bb = r.get("bollinger", {})
            
            price = ticker.get("price", 0)
            change = ticker.get("change_24h", 0)
            change_icon = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            lines.append(f"## {symbol}")
            lines.append(f"- 价格: ${price:,.2f} ({change_icon} {change:+.2f}%)")
            
            if "error" not in bb:
                lines.append(f"- 布林带(4H):")
                lines.append(f"  - 上轨: ${bb['upper']:,.2f}")
                lines.append(f"  - 中轨: ${bb['middle']:,.2f}")
                lines.append(f"  - 下轨: ${bb['lower']:,.2f}")
                lines.append(f"  - 带宽: {bb['bandwidth']:.2f}%")
                lines.append(f"  - 信号: {bb['signal']}")
            
            lines.append("")
        
        return "\n".join(lines)


if __name__ == "__main__":
    analyzer = BinanceAnalyzer()
    if analyzer.available:
        results = analyzer.get_full_report()
        print(analyzer.format_report(results))
    else:
        print("请先安装: pip install python-binance")
