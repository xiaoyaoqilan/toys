"""测试邮件推送 - 使用配置好的 SMTP 发送测试邮件。"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def test_email():
    logger.info("正在发送测试邮件到 1131892323@qq.com...")
    
    try:
        from notify.notifier import Notifier
        
        notifier = Notifier(
            smtp_server="smtp.qq.com",
            smtp_port=465,
            smtp_user="1131892323@qq.com",
            smtp_password="wavivgtckrhjfijb",
            email_to="1131892323@qq.com",
        )
        
        test_content = """
## 🔔 金融 RAG 系统 - 邮件推送测试

如果你收到了这封邮件，说明邮件推送配置成功！

---

### ✅ 系统已具备双通道推送能力

1. **微信推送**（Server酱）：已配置 ✅
2. **邮件推送**（SMTP）：已配置 ✅

---

### 📊 系统功能

本系统可以：
- 🤖 自动抓取财经新闻
- 🔍 多信源交叉验证
- 🧠 LLM 智能分析
- 📱 推送到微信 + 📧 推送到邮箱

---

测试时间：现在
        """
        
        result = notifier._send_email("🔔 金融 RAG 系统测试 - 邮件推送", test_content)
        logger.info(f"✅ 发送成功！结果: {result}")
        logger.info("请检查你的邮箱是否收到了测试邮件。")
        
    except Exception as e:
        logger.error(f"❌ 发送失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_email()
