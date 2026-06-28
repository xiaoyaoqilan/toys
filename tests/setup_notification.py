"""通知推送测试脚本 - 演示如何配置和使用推送功能。"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def setup_wechat():
    """配置企业微信推送。"""
    print("\n" + "=" * 60)
    print("📱 配置企业微信推送")
    print("=" * 60)
    print("""
要使用企业微信推送，你需要：
1. 在企业微信群聊中，点击右上角设置
2. 选择 "群机器人" -> "添加机器人"
3. 自定义机器人名称，如 "财经日报助手"
4. 创建后复制 Webhook 地址

示例 Webhook: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxx
""")
    webhook = input("请输入企业微信 Webhook 地址 (直接跳过则不配置): ").strip()
    return webhook if webhook else None

def setup_serverchan():
    """配置 Server酱 推送 (推送到个人微信)。"""
    print("\n" + "=" * 60)
    print("📱 配置 Server酱 推送 (推送到个人微信)")
    print("=" * 60)
    print("""
要使用 Server酱 推送到个人微信：
1. 访问 https://sct.ftqq.com/ 注册账号
2. 登录后获取 SendKey (类似 SCT123456T...)
3. 将 SendKey 填入下方

优点：可以推送到你的个人微信，不需要企业微信群
""")
    key = input("请输入 Server酱 SendKey (直接跳过则不配置): ").strip()
    return key if key else None

def setup_email():
    """配置邮件推送。"""
    print("\n" + "=" * 60)
    print("📧 配置邮件推送")
    print("=" * 60)
    print("""
要使用邮件推送，你需要：
1. 一个支持 SMTP 的邮箱账号
2. 邮箱的 SMTP 授权码 (不是登录密码)

常用邮箱 SMTP 配置：
- QQ 邮箱: smtp.qq.com:465
- 163 邮箱: smtp.163.com:465
- Gmail: smtp.gmail.com:587

获取授权码：
- QQ 邮箱: 设置 -> 账户 -> 开启 SMTP -> 获取授权码
""")
    
    config = {}
    smtp_server = input("SMTP 服务器 (默认 smtp.qq.com): ").strip() or "smtp.qq.com"
    smtp_user = input("发件邮箱地址: ").strip()
    smtp_password = input("邮箱授权码 (不是密码): ").strip()
    email_to = input("收件邮箱地址: ").strip()
    
    if smtp_user and smtp_password and email_to:
        config = {
            "smtp_server": smtp_server,
            "smtp_user": smtp_user,
            "smtp_password": smtp_password,
            "email_to": email_to,
        }
        return config
    return None

def test_notification(notifier, name):
    """测试发送通知。"""
    print("\n🔔 正在发送测试通知...")
    results = notifier.send(
        title="🔔 财经日报系统 - 推送测试",
        content=f"这是一条来自金融 RAG 系统的测试消息。\n\n如果你收到了这条消息，说明 **{name}** 推送配置成功！\n\n时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"   推送结果: {results}")
    return results

def main():
    print("=" * 60)
    print("🚀 金融 RAG 系统 - 通知推送配置向导")
    print("=" * 60)
    
    from notify import Notifier
    
    # 引导用户配置
    print("\n请选择你要配置的推送方式：")
    print("1. 企业微信 (推荐群聊使用)")
    print("2. Server酱 (推送到个人微信)")
    print("3. 邮件推送")
    print("4. 配置多种方式")
    
    choice = input("\n请输入选项 (1/2/3/4): ").strip()
    
    notifier = Notifier()
    configs = []
    
    if choice == "1" or choice == "4":
        webhook = setup_wechat()
        if webhook:
            notifier.wechat_webhook = webhook
            configs.append("企业微信")
    
    if choice == "2" or choice == "4":
        key = setup_serverchan()
        if key:
            notifier.serverchan_key = key
            configs.append("Server酱")
    
    if choice == "3" or choice == "4":
        email_config = setup_email()
        if email_config:
            notifier.smtp_server = email_config["smtp_server"]
            notifier.smtp_user = email_config["smtp_user"]
            notifier.smtp_password = email_config["smtp_password"]
            notifier.email_to = email_config["email_to"]
            configs.append("邮件")
    
    if not configs:
        print("\n⚠️ 没有配置任何推送方式。")
        return
    
    print(f"\n✅ 已配置的推送方式: {', '.join(configs)}")
    
    # 测试推送
    test = input("\n是否发送测试通知? (y/n): ").strip().lower()
    if test == "y":
        for name in configs:
            test_notification(notifier, name)
    
    # 保存配置到环境变量示例
    print("\n" + "=" * 60)
    print("💡 生产环境配置建议")
    print("=" * 60)
    print("""
你可以将以下配置保存为环境变量，供系统自动读取：

# Windows PowerShell
[Environment]::SetEnvironmentVariable("WECHAT_WEBHOOK", "你的webhook", "User")
[Environment]::SetEnvironmentVariable("SERVERCHAN_KEY", "你的sendkey", "User")
[Environment]::SetEnvironmentVariable("SMTP_USER", "你的邮箱", "User")
[Environment]::SetEnvironmentVariable("SMTP_PASSWORD", "你的授权码", "User")
[Environment]::SetEnvironmentVariable("EMAIL_TO", "收件邮箱", "User")

或者创建一个 .env 文件在项目根目录：
WECHAT_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
SERVERCHAN_KEY=SCT123456T...
SMTP_SERVER=smtp.qq.com
SMTP_USER=your@email.com
SMTP_PASSWORD=your_auth_code
EMAIL_TO=receiver@email.com
""")
    
    print("\n🎉 配置完成！你可以使用 Notifier 类来发送通知了。")
    print("\n示例代码:")
    print("""
from notify import Notifier

notifier = Notifier(
    wechat_webhook="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
)
notifier.send("标题", "内容")
""")

if __name__ == "__main__":
    main()
