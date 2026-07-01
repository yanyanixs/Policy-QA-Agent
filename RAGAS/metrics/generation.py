"""
生成质量指标

- answer_relevancy：答案与问题的关联程度
- answer_similarity：生成答案与真实答案的语义相似度
- answer_correctness：答案与真实答案的事实匹配程度
"""

import json
from typing import List

from utils.llm_client import llm_ask, llm_ask_json
from utils.embedding import pairwise_similarity, get_embedding, cosine_similarity


# ─── answer_relevancy ──────────────────────────────────

ANSWER_RELEVANCY_SYSTEM = """你是一个 RAG 系统评估专家。你的任务是根据给定的答案，反推出用户可能问的问题。

请根据答案内容，生成 3 个可能的用户提问。这些提问应该能通过给定答案来回答。

输出 JSON：
{
  "generated_questions": ["问题1", "问题2", "问题3"]
}"""


def answer_relevancy(question: str, answer: str) -> float:
    """
    答案相关性：LLM 从 answer 反推可能的提问，再与原始 question 计算语义相似度。

    分数 = 反推问题与原问题的平均余弦相似度。
    """
    if not answer or not question:
        return 0.0

    user_prompt = f"答案：{answer}"

    try:
        result = json.loads(llm_ask_json(ANSWER_RELEVANCY_SYSTEM, user_prompt))
        generated = result.get("generated_questions", [])
    except (json.JSONDecodeError, Exception):
        return 0.0

    if not generated:
        return 0.0

    # 计算每个生成问题与原问题的相似度，取均值
    question_vec = get_embedding(question)
    scores = []
    for gq in generated:
        gq_vec = get_embedding(gq)
        scores.append(cosine_similarity(question_vec, gq_vec))

    return sum(scores) / len(scores)


# ─── answer_similarity ─────────────────────────────────

def answer_similarity(answer: str, ground_truths: List[str]) -> float:
    """
    答案相似度：answer 与 ground_truth 的语义向量余弦相似度。

    如果有多个 ground_truth，取最高相似度。
    使用 BGE-M3 向量化后计算（轻量，不依赖 LLM）。
    """
    if not answer or not ground_truths:
        return 0.0

    best_score = 0.0
    for gt in ground_truths:
        score = pairwise_similarity(answer, gt)
        if score > best_score:
            best_score = score

    return best_score


# ─── answer_correctness ────────────────────────────────

ANSWER_CORRECTNESS_SYSTEM = """你是一个 RAG 系统评估专家。你的任务是将答案和真实答案分别拆解为事实性陈述，然后逐一比对。

步骤：
1. 从生成答案（answer）中提取所有事实性陈述
2. 从真实答案（ground truth）中提取所有事实性陈述
3. 对 answer 的每条陈述，判断是否与 ground truth 中的某条一致
   - "full_match": 完全一致或有细微同义替换
   - "partial_match": 部分一致但缺少细节或有小偏差
   - "no_match": 与真实答案矛盾或完全不同

输出 JSON：
{
  "answer_statements": ["陈述1", "陈述2", ...],
  "ground_truth_statements": ["陈述1", "陈述2", ...],
  "matches": [
    {"answer_stmt": "陈述1", "verdict": "full_match", "matched_gt": "对应的真实陈述"},
    ...
  ]
}"""


def answer_correctness(answer: str, ground_truths: List[str]) -> float:
    """
    答案正确性：将 answer 和 ground_truth 拆解为事实陈述后逐一比对。

    加权计分：full_match = 1.0, partial_match = 0.5, no_match = 0.0
    """
    if not answer or not ground_truths:
        return 0.0

    ground_truth = "\n".join(ground_truths)
    user_prompt = (
        f"生成答案：\n{answer}\n\n"
        f"真实答案（ground truth）：\n{ground_truth}"
    )

    try:
        result = json.loads(llm_ask_json(ANSWER_CORRECTNESS_SYSTEM, user_prompt))
        matches = result.get("matches", [])
        if not matches:
            return 0.0

        total = len(matches)
        score = 0.0
        for m in matches:
            v = m.get("verdict", "no_match")
            if v == "full_match":
                score += 1.0
            elif v == "partial_match":
                score += 0.5
            # no_match → +0

        return score / total
    except (json.JSONDecodeError, Exception):
        return 0.0
