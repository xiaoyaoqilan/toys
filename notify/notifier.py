"""通知推送模块 - 支持微信、邮件等多种渠道。"""
from __future__ import annotations

import json
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from loguru import logger

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class Notifier:
    """通用通知推送器，支持多种渠道。"""
    
    def __init__(
        self,
        wechat_webhook: str = "",
        serverchan_key: str = "",
        smtp_server: str = "",
        smtp_port: int = 465,
        smtp_user: str = "",
        smtp_password: str = "",
        email_to: str = "",
    ):
        self.wechat_webhook = wechat_webhook
        self.serverchan_key = serverchan_key
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.email_to = email_to

    def send(
        self,
        title: str,
        content: str,
        channels: Optional[list] = None,
    ) -> dict:
        """
        发送通知到指定渠道。
        channels: ["wechat", "serverchan", "email"]，如果为 None 则发送到所有已配置的渠道。
        """
        results = {}
        
        if channels is None:
            channels = self._get_configured_channels()
            
        for channel in channels:
            try:
                if channel == "wechat":
                    if self.wechat_webhook:
                        results["wechat"] = self._send_wechat(title, content)
                    else:
                        results["wechat"] = "未配置"
                        
                elif channel == "serverchan":
                    if self.serverchan_key:
                        results["serverchan"] = self._send_serverchan(title, content)
                    else:
                        results["serverchan"] = "未配置"
                        
                elif channel == "email":
                    if self.smtp_server and self.email_to:
                        results["email"] = self._send_email(title, content)
                    else:
                        results["email"] = "未配置"
            except Exception as e:
                logger.error(f"Failed to send via {channel}: {e}")
                results[channel] = f"失败: {e}"
                
        return results

    def _get_configured_channels(self) -> list:
        channels = []
        if self.wechat_webhook:
            channels.append("wechat")
        if self.serverchan_key:
            channels.append("serverchan")
        if self.smtp_server and self.email_to:
            channels.append("email")
        return channels

    # ---------- 微信推送 ----------
    def _send_wechat(self, title: str, content: str) -> str:
        """通过企业微信 Webhook 推送。"""
        if not _REQUESTS_AVAILABLE:
            return "requests 库未安装"
            
        data = {
            "msgtype": "markdown",
            "markdown": {
                "content": f"## {title}\n\n{content}"
            }
        }
        
        response = requests.post(
            self.wechat_webhook,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("errcode") == 0:
            logger.info("✅ 企业微信推送成功")
            return "成功"
        else:
            return f"失败: {result}"

    def _send_serverchan(self, title: str, content: str) -> str:
        """通过 Server酱 (ServerChan) 推送到微信。"""
        if not _REQUESTS_AVAILABLE:
            return "requests 库未安装"
            
        url = f"https://sctapi.ftqq.com/{self.serverchan_key}.send"
        data = {
            "title": title,
            "desp": content
        }
        
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if result.get("code") == 0:
            logger.info("✅ Server酱推送成功")
            return "成功"
        else:
            return f"失败: {result}"

    # ---------- 邮件推送 ----------
    def _send_email(self, title: str, content: str) -> str:
        """通过 SMTP 发送邮件。"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = title
        msg["From"] = self.smtp_user
        msg["To"] = self.email_to
        
        # 支持 Markdown 和纯文本
        html_content = self._markdown_to_html(content)
        msg.attach(MIMEText(html_content, "html", "utf-8"))
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.smtp_user, self.email_to, msg.as_string())
            
        logger.info("✅ 邮件推送成功")
        return "成功"

    @staticmethod
    def _markdown_to_html(md_content: str) -> str:
        """简单的 Markdown 转 HTML。"""
        import html
        
        # 基本转义
        text = html.escape(md_content)
        
        # 处理标题
        import re
        text = re.sub(r'## (.*)', r'<h2>\1</h2>', text)
        text = re.sub(r'### (.*)', r'<h3>\1</h3>', text)
        
        # 处理列表
        text = re.sub(r'^- (.*)', r'• \1', text, flags=re.MULTILINE)
        
        # 处理换行
        text = text.replace('\n', '<br>\n')
        
        # 处理加粗
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        return f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333;">
        {text}
        </body>
        </html>
        """


def create_notifier_from_config() -> Notifier:
    """从环境变量或默认配置创建 Notifier。"""
    import os
    
    return Notifier(
        wechat_webhook=os.getenv("WECHAT_WEBHOOK", ""),
        serverchan_key=os.getenv("SERVERCHAN_KEY", ""),
        smtp_server=os.getenv("SMTP_SERVER", "smtp.qq.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "465")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        email_to=os.getenv("EMAIL_TO", ""),
    )
