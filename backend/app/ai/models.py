from enum import Enum
from typing import TypeAlias

DEFAULT_MODEL = "deepseek-chat"


class OpenAIModelName(str, Enum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

class DeepseekModelName(str, Enum):
    """https://api-docs.deepseek.com/quick_start/pricing"""
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_V4_FLASH = "deepseek-v4-flash"

class OllamaModelName(str, Enum):
    """https://ollama.com/search"""

    OLLAMA_GENERIC = "ollama"

class FakeModelName(str, Enum):
    """Fake model for testing."""
    FAKE = "fake"

class TongYiModelName(str, Enum):
    """TongYi model"""
    QWEN_PLUS = "qwen-plus"
    QWEN_MAX = "qwen-max"
    



AllModelEnum: TypeAlias = (
    OpenAIModelName
    | DeepseekModelName
    | OllamaModelName
    | FakeModelName
    | TongYiModelName
)
