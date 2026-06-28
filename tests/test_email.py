"""邮件推送测试脚本 - 配置你的邮箱并测试发送。"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def test_email():
    logger.info("=== 邮件推送测试 ===")
    print()
    
    # 配置邮箱参数
    config = {
        "smtp_server": "smtp.qq.com",  # QQ 邮箱 SMTP 服务器
        "smtp_port": 465,              # SSL 端口
        "smtp_user": "你的邮箱@qq.com", # 发送邮箱
        "smtp_password": "你的授权码",  # 注意：这是授权码，不是邮箱密码
        "email_to": "接收邮箱@qq.com",   # 接收邮箱
    }
    
    logger.info("请修改以下配置信息：")
    print()
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    print()
    logger.info("提示：")
    logger.info("  1. QQ 邮箱: 需在 '设置 -> 账户' 中开启 SMTP 服务，并获取授权码")
    logger.info("  2. 163 邮箱: 同样需要开启 SMTP 并获取授权码")
    logger.info("  3. Gmail: 需要生成应用专用密码")
    print()
    
    # 创建 Notifier
    from notify.notifier import Notifier
    
    notifier = Notifier(
        smtp_server=config["smtp_server"],
        smtp_port=config["smtp_port"],
        smtp_user=config["smtp_user"],
        smtp_password=config["smtp_password"],
        email_to=config["email_to"],
    )
    
    # 发送测试邮件
    logger.info("正在发送测试邮件...")
    
    test_content = """
## 🔔 金融 RAG 系统 - 邮件推送测试

如果你收到了这封邮件，说明：
- ✅ 邮件推送配置成功
- ✅ 系统可以向你的邮箱发送分析报告
- ✅ 可以结合 Server酱实现双通道推送

---

### 📊 系统功能

本系统可以：
- 🤖 自动抓取财经新闻
- 🔍 多信源交叉验证
- 🧠 LLM 智能分析
- 📱 推送到微信 + 📧 推送到邮箱

---

测试时间：""" + "现在"
    
    try:
        result = notifier._send_email("🔔 金融 RAG 系统测试", test_content)
        logger.info(f"结果: {result}")
        logger.info("✅ 测试完成！")
    except Exception as e:
        logger.error(f"❌ 发送失败: {e}")
        print()
        logger.error("常见问题排查：")
        logger.error("  1. 检查 SMTP 服务器地址是否正确")
        logger.error("  2. 检查授权码是否正确（不是邮箱密码）")
        logger.error("  3. 检查邮箱是否已开启 SMTP 服务")
        logger.error("  4. 检查网络连接是否正常")

if __name__ == "__main__":
    test_email()
