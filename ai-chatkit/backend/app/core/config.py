"""
config settings
"""

# from pydantic import BaseSettings
from ctypes import DEFAULT_MODE
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    
    APP_NAME: str = "AI ChatKit"
    DEBUG: bool = True
    DATABASE_URL: str | None = None
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEV: bool = True
    
    DEEPSEEK_API_KEY: str | None = None
    OLLAMA_BASE_URL: str | None = None
    OLLAMA_MODEL: str | None = None
    
    DEFAULT_MODEL: str | None = None
    
    EMBEDDING_MODEL: str | None = None
    
    CHROMA_PATH: str | None = None

    # ── Reranker 配置 ──────────────────────────────────
    # 后端: "cross-encoder" (BGE-Reranker, ~2GB) | "llm" (DeepSeek, 无需额外模型) | "none" (跳过精排)
    RERANKER_BACKEND: str = "cross-encoder"
    # BGE-Reranker 模型路径: HuggingFace model id 或本地路径
    RERANKER_MODEL_PATH: str = "BAAI/bge-reranker-v2-m3"
    # HuggingFace 镜像 (国内用户建议设: https://hf-mirror.com)
    HF_ENDPOINT: str | None = None

    def is_dev(self):
        return self.DEV



settings = Settings()