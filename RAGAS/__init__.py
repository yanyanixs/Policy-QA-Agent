"""
RAGAS 评估模块 — 包入口

Usage:
    from ragas.evaluate import evaluate_rag

    result = evaluate_rag(
        question="北京的数据政策有哪些？",
        answer="根据现有政策，北京...",
        contexts=["文档1内容...", "文档2内容..."],
        ground_truths=["北京的数据政策包括..."],
    )
"""

from .evaluate import evaluate_rag, evaluate_rag_detailed
