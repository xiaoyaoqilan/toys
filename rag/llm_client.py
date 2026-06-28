"""LLM 客户端 - 封装 DeepSeek API。"""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class LLMClient:
    """
    DeepSeek 大模型客户端。
    用于 RAG 系统的响应合成、查询改写和推理。
    """
    
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    DEFAULT_MODEL = "deepseek-chat"
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self._is_available = bool(self.api_key and _REQUESTS_AVAILABLE)
        
        if self._is_available:
            logger.info(f"LLM Client initialized (model: {model})")
        else:
            if not self.api_key:
                logger.warning("DEEPSEEK_API_KEY not set. LLM features will be disabled.")
            if not _REQUESTS_AVAILABLE:
                logger.warning("requests library not available. LLM features will be disabled.")

    @property
    def is_available(self) -> bool:
        return self._is_available

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """
        发送对话请求给 LLM。
        """
        if not self._is_available:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            response = requests.post(
                self.DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("LLM connection error")
            return None
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return None

    def rewrite_query(self, original_query: str) -> str:
        """
        查询改写：将用户的口语化问题改写为更适合检索的查询。
        """
        if not self._is_available:
            # Fallback: 简单的规则改写
            if len(original_query.split()) < 3:
                return f"{original_query} latest news analysis"
            return original_query
            
        system_prompt = "你是一个搜索优化专家。你的任务是将用户的口语化问题改写为更适合在财经新闻数据库中检索的精确查询。"
        user_prompt = f"""请将以下用户问题改写为适合搜索的查询语句：

用户问题：{original_query}

要求：
1. 提取核心实体和关键词
2. 补充可能相关的财经术语
3. 返回改写后的查询语句，不要解释"""
        
        result = self.chat(system_prompt, user_prompt, temperature=0.1)
        return result.strip() if result else original_query

    def synthesize_answer(self, question: str, context_text: str) -> Optional[str]:
        """
        响应合成：基于检索到的上下文，生成最终的分析报告。
        """
        if not self._is_available:
            return None
            
        system_prompt = """你是一名资深金融投研分析师。你的任务是基于提供的多信源财经资讯，生成专业的分析报告。

要求：
1. 严格基于提供的事实，不要编造信息
2. 对关键结论标注事实来源和置信度
3. 分析事件的潜在影响和趋势
4. 如证据不足，明确说明局限
5. 使用清晰的Markdown格式"""
        
        user_prompt = f"""请根据以下事实，回答用户的问题并生成分析报告。

用户问题：{question}

以下是从多个信源检索到的相关事实：

{context_text}

请生成一份专业的分析报告，包括：
1. 核心发现摘要
2. 详细分析
3. 置信度评估
4. 后续关注点"""
        
        return self.chat(system_prompt, user_prompt, temperature=0.3, max_tokens=3000)

    def predict_impact(self, event: str, affected_entities: list) -> Optional[str]:
        """
        影响预测：基于事件和受影响实体，进行因果推理。
        """
        if not self._is_available:
            return None
            
        entities_str = "、".join(affected_entities) if affected_entities else "相关领域"
        
        system_prompt = """你是一名金融市场分析师，擅长分析宏观和行业事件对市场的连锁影响。"""
        
        user_prompt = f"""请分析以下事件可能对 {entities_str} 产生的影响：

事件：{event}

请提供：
1. 影响传导路径分析
2. 潜在的利好/利空因素
3. 影响的时间周期判断
4. 风险提示"""
        
        return self.chat(system_prompt, user_prompt, temperature=0.4, max_tokens=2000)
