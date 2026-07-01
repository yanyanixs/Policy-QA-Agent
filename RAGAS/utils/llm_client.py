"""
LLM 调用封装 — 评估模块专用

复用 ai-chatkit 后端的 DeepSeek 模型，温度=0（评估需确定性输出）。
"""

import sys
import os
import asyncio
from typing import Optional

# 挂载 ai-chatkit 后端路径
_BACKEND_APP = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "app")
_BACKEND_APP = os.path.abspath(_BACKEND_APP)
if _BACKEND_APP not in sys.path:
    sys.path.insert(0, _BACKEND_APP)

# 加载 .env
from dotenv import load_dotenv
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env")
load_dotenv(os.path.abspath(_ENV_PATH))

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from ai.llm import get_model
from ai.models import DeepseekModelName


def _sync_call(messages: list[BaseMessage], model_name: str = DeepseekModelName.DEEPSEEK_V4_FLASH) -> str:
    """同步调用 LLM（内部用 asyncio.run）"""
    model = get_model(model_name)
    model.temperature = 0.0  # 评估场景：确定性输出
    response = model.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def llm_ask(
    system_prompt: str,
    user_prompt: str,
    model_name: str = DeepseekModelName.DEEPSEEK_V4_FLASH,
) -> str:
    """
    向 LLM 发送一条 system + user 消息，返回文本响应。

    Args:
        system_prompt: 系统提示词（角色定义）
        user_prompt: 用户提示词（具体任务）
        model_name: 模型名，默认 deepseek-v4-flash

    Returns:
        LLM 的文本回复
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    return _sync_call(messages, model_name)


def llm_ask_json(
    system_prompt: str,
    user_prompt: str,
    model_name: str = DeepseekModelName.DEEPSEEK_V4_FLASH,
) -> str:
    """
    同 llm_ask，但在 system_prompt 中追加 JSON 格式要求。
    返回原始文本（调用方自行 json.loads）。
    """
    json_instruction = "\n\n重要：你的回复必须是合法的 JSON，不要包含任何其他文字。"
    return llm_ask(system_prompt + json_instruction, user_prompt, model_name)
