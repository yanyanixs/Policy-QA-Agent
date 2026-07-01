"""
向量化封装 — 评估模块专用

复用 ai-chatkit 后端的 Ollama BGE-M3 Embedding，提供余弦相似度计算。
"""

import sys
import os
import numpy as np

# 确保后端路径已挂载
_BACKEND_APP = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "app")
_BACKEND_APP = os.path.abspath(_BACKEND_APP)
if _BACKEND_APP not in sys.path:
    sys.path.insert(0, _BACKEND_APP)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度 (0~1)"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedding(text: str) -> list[float]:
    """
    用 BGE-M3 (Ollama) 将文本转为向量。

    Args:
        text: 输入文本

    Returns:
        向量列表
    """
    from ai.rag.chromaClient import embeddings
    return embeddings.embed_query(text)


def pairwise_similarity(text_a: str, text_b: str) -> float:
    """
    计算两段文本的语义相似度 (0~1)。

    先分别向量化，再计算余弦相似度。
    """
    vec_a = get_embedding(text_a)
    vec_b = get_embedding(text_b)
    return cosine_similarity(vec_a, vec_b)
