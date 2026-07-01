"""
检索质量指标

- context_precision：检索到的上下文是否均为回答问题所必需
- context_recall：检索到的上下文是否包含回答所需的全部关键信息
- context_relevancy：检索到的上下文与问题的语义关联度
- context_entity_recall：检索到的上下文中关键实体的完整性
"""

import json
import sys
import os
from typing import List

from utils.llm_client import llm_ask, llm_ask_json
from utils.embedding import pairwise_similarity


# ─── context_precision ──────────────────────────────────

CONTEXT_PRECISION_SYSTEM = """你是一个 RAG 系统评估专家。你的任务是判断一条检索到的上下文片段是否对回答用户问题「必要」。

判断标准：
- 如果移除这条上下文，回答质量会明显下降 → "necessary"
- 如果这条上下文与问题无关或回答不需要它 → "unnecessary"

请严格按 JSON 格式输出：{"verdict": "necessary" 或 "unnecessary", "reason": "一句话说明理由"}"""


def context_precision(question: str, answer: str, contexts: List[str]) -> float:
    """
    上下文精确性：必要上下文数 / 总上下文数。

    对每条 context，LLM 判断是否必要。必要 = 移除后会降低答案质量。
    """
    if not contexts:
        return 0.0

    necessary_count = 0
    for ctx in contexts:
        user_prompt = f"问题：{question}\n答案：{answer}\n待判断的上下文：{ctx}"
        try:
            result = json.loads(llm_ask_json(CONTEXT_PRECISION_SYSTEM, user_prompt))
            if result.get("verdict") == "necessary":
                necessary_count += 1
        except (json.JSONDecodeError, Exception):
            # 解析失败时保守计为 unnecessary
            continue

    return necessary_count / len(contexts)


# ─── context_recall ────────────────────────────────────

CONTEXT_RECALL_SYSTEM = """你是一个 RAG 系统评估专家。你的任务是从真实答案中提取关键信息点，然后检查这些信息点是否在检索到的上下文中出现。

步骤：
1. 从 ground_truth 中提取关键信息点（核心事实、实体、逻辑关系），每点一行
2. 对每个信息点，判断它是否在上下文中出现（语义匹配即可，不要求文字完全一致）

输出 JSON：
{
  "key_points": ["信息点1", "信息点2", ...],
  "matched": [true, false, ...]
}"""


def context_recall(question: str, contexts: List[str], ground_truths: List[str]) -> float:
    """
    上下文召回率：被检索到的关键信息点数 / 所有必要关键信息点数。

    ground_truths: 预期标准答案列表（可能有多条）
    """
    if not contexts or not ground_truths:
        return 0.0

    ground_truth = "\n".join(ground_truths)
    context_text = "\n\n---\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))

    user_prompt = f"问题：{question}\n\n真实答案（ground truth）：\n{ground_truth}\n\n检索到的上下文：\n{context_text}"

    try:
        result = json.loads(llm_ask_json(CONTEXT_RECALL_SYSTEM, user_prompt))
        matched = result.get("matched", [])
        if not matched:
            return 0.0
        return sum(1 for m in matched if m) / len(matched)
    except (json.JSONDecodeError, Exception):
        return 0.0


# ─── context_relevancy ─────────────────────────────────

def context_relevancy(question: str, contexts: List[str]) -> float:
    """
    上下文相关性：每条 context 与 question 的余弦相似度均值。

    使用 BGE-M3 向量化后计算余弦相似度（轻量，不依赖 LLM）。
    """
    if not contexts:
        return 0.0

    scores = []
    for ctx in contexts:
        sim = pairwise_similarity(question, ctx)
        scores.append(sim)

    return sum(scores) / len(scores)


# ─── context_entity_recall ─────────────────────────────

CONTEXT_ENTITY_RECALL_SYSTEM = """你是一个信息抽取专家。你的任务是从真实答案中提取核心实体，然后检查这些实体是否在检索到的上下文中出现。

核心实体包括：人名、地名、机构名、术语、事件名、数值关键指标等。

注意：同义词、缩写、别称视作匹配（如 "国家能源局" 和 "能源局" 视为同一实体）。

输出 JSON：
{
  "entities": ["实体1", "实体2", ...],
  "matched": [true, false, ...]
}"""


def context_entity_recall(question: str, contexts: List[str], ground_truths: List[str]) -> float:
    """
    上下文实体召回率：检索到的关键实体数 / 所有必要关键实体数。
    """
    if not contexts or not ground_truths:
        return 0.0

    ground_truth = "\n".join(ground_truths)
    context_text = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))

    user_prompt = f"问题：{question}\n\n真实答案（ground truth）：\n{ground_truth}\n\n检索到的上下文（共 {len(contexts)} 条）：\n{context_text}"

    try:
        result = json.loads(llm_ask_json(CONTEXT_ENTITY_RECALL_SYSTEM, user_prompt))
        matched = result.get("matched", [])
        if not matched:
            return 0.0
        return sum(1 for m in matched if m) / len(matched)
    except (json.JSONDecodeError, Exception):
        return 0.0
