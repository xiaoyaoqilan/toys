"""立即发送一条测试消息到微信。"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def test_serverchan():
    logger.info("正在发送测试消息到微信...")
    
    try:
        import requests
        
        sendkey = "SCT371241Tmo9PuexweDuxplls3RzTA5D4"
        url = f"https://sctapi.ftqq.com/{sendkey}.send"
        
        data = {
            "title": "🔔 金融 RAG 系统 - 推送测试",
            "desp": """
## 测试消息

这是一条来自 **金融多信源校验 RAG 平台** 的测试消息。

---

### ✅ 配置成功

如果你收到了这条消息，说明：
1. **Server酱推送配置成功**
2. **微信通知通道已打通**
3. **系统可以随时向你推送财经分析报告**

---

### 📊 系统功能

本系统可以：
- 🤖 自动抓取财经新闻
- 🔍 多信源交叉验证
- 🧠 LLM 智能分析
- 📱 主动推送到微信

---

测试时间：""" + "现在"
        }
        
        response = requests.post(url, data=data, timeout=10)
        result = response.json()
        
        if result.get("code") == 0:
            logger.info("✅ 发送成功！请检查你的微信是否收到消息。")
            logger.info(f"   返回信息: {result.get('message', '成功')}")
        else:
            logger.error(f"❌ 发送失败: {result}")
            
        return result
        
    except Exception as e:
        logger.error(f"发送异常: {e}")
        return None

if __name__ == "__main__":
    test_serverchan()
