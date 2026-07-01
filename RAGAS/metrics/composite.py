"""
综合指标

- ragas_score：对所选指标加权求和，得到综合评分 (0~1)
"""

from typing import Dict, List, Optional


# 默认权重配置
DEFAULT_WEIGHTS: Dict[str, float] = {
    # 检索质量
    "context_precision": 0.15,
    "context_recall": 0.15,
    "context_relevancy": 0.10,
    "context_entity_recall": 0.05,
    # 生成质量
    "answer_relevancy": 0.10,
    "answer_similarity": 0.10,
    "answer_correctness": 0.15,
    # 事实一致性
    "faithfulness": 0.15,
    "hallucination_score": 0.05,  # 反转：低幻觉 = 高分
}

# 指标含义说明
METRIC_LABELS: Dict[str, str] = {
    "context_precision": "上下文精确性",
    "context_recall": "上下文召回率",
    "context_relevancy": "上下文相关性",
    "context_entity_recall": "上下文实体召回率",
    "answer_relevancy": "答案相关性",
    "answer_similarity": "答案相似度",
    "answer_correctness": "答案正确性",
    "faithfulness": "忠实性",
    "hallucination_score": "幻觉评分",
}


def ragas_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    计算 RAGAS 综合得分。

    对各项指标按权重加权求和。hallucination_score 自动反转（1 - score），
    使高分 = 低幻觉。

    Args:
        metrics: 各项指标得分字典
        weights: 自定义权重，默认使用 DEFAULT_WEIGHTS

    Returns:
        综合得分 (0~1)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total = 0.0
    total_weight = 0.0

    for metric_name, score in metrics.items():
        if metric_name not in weights:
            continue

        weight = weights[metric_name]

        # hallucination_score 反转：低幻觉 = 高分
        if metric_name == "hallucination_score":
            score = 1.0 - score

        total += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return total / total_weight


def ragas_score_detailed(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    返回详细的综合评分信息，包含每个指标的贡献值。
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    detail = {}
    total = 0.0
    total_weight = 0.0

    for metric_name in weights:
        score = metrics.get(metric_name)
        if score is None:
            detail[metric_name] = {"raw_score": None, "contribution": 0, "label": METRIC_LABELS.get(metric_name, metric_name)}
            continue

        weight = weights[metric_name]

        # hallucination_score 反转
        adjusted = 1.0 - score if metric_name == "hallucination_score" else score
        contribution = adjusted * weight

        detail[metric_name] = {
            "raw_score": round(score, 4),
            "adjusted_score": round(adjusted, 4),
            "weight": weight,
            "contribution": round(contribution, 4),
            "label": METRIC_LABELS.get(metric_name, metric_name),
        }

        total += contribution
        total_weight += weight

    ragas = round(total / total_weight, 4) if total_weight > 0 else 0.0

    return {
        "ragas_score": ragas,
        "total_weight": round(total_weight, 4),
        "detail": detail,
    }
