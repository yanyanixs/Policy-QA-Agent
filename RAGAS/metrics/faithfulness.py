"""
事实一致性指标

- faithfulness：答案是否完全基于检索到的上下文（无编造）
- hallucination_score：答案中编造信息的比例（与 faithfulness 互补）
"""

import json
from typing import List

from utils.llm_client import llm_ask_json


# ─── faithfulness ──────────────────────────────────────

FAITHFULNESS_SYSTEM = """你是一个 RAG 系统评估专家。你的任务是判断生成答案中的每条陈述是否能在检索到的上下文中找到依据。

步骤：
1. 将生成答案拆解为独立的事实性陈述
2. 对每条陈述，在上下文中寻找明确依据
3. 判断：有依据 → "supported"，无依据（编造）→ "unsupported"

注意：
- 同义表达、合理推论都算"有依据"
- 只有在上下文中完全找不到任何支持信息的才算"无依据"

输出 JSON：
{
  "statements": ["陈述1", "陈述2", ...],
  "verdicts": ["supported", "unsupported", ...]
}"""


def faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    """
    忠实性：有依据的陈述数 / 总陈述数。

    值越接近 1 表示答案越忠实于检索到的上下文。
    """
    if not answer or not contexts:
        return 0.0

    context_text = "\n\n---\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    user_prompt = (
        f"问题：{question}\n\n"
        f"检索到的上下文（共 {len(contexts)} 条）：\n{context_text}\n\n"
        f"生成答案：\n{answer}"
    )

    try:
        result = json.loads(llm_ask_json(FAITHFULNESS_SYSTEM, user_prompt))
        verdicts = result.get("verdicts", [])
        if not verdicts:
            return 0.0
        supported = sum(1 for v in verdicts if v == "supported")
        return supported / len(verdicts)
    except (json.JSONDecodeError, Exception):
        return 0.0


# ─── hallucination_score ───────────────────────────────

HALLUCINATION_SYSTEM = """你是一个 RAG 系统评估专家。你的任务是识别生成答案中所有「未在检索上下文中出现」的事实性陈述（幻觉）。

步骤：
1. 将生成答案拆解为独立的事实性陈述
2. 对每条陈述，判断是否能在上下文中找到依据
3. 如果在上下文中完全找不到任何支持 → 标记为 "hallucination"

重要区分：
- 上下文中明确提到 → 不是幻觉
- 上下文中可以合理推断 → 不是幻觉
- 答案编造了上下文中不存在的事实 → 是幻觉

输出 JSON：
{
  "statements": ["陈述1", "陈述2", ...],
  "is_hallucination": [true, false, ...]
}"""


def hallucination_score(question: str, answer: str, contexts: List[str]) -> float:
    """
    幻觉评分：幻觉陈述数 / 总陈述数。

    值越接近 0 表示幻觉越少（与 faithfulness 互补）。
    """
    if not answer or not contexts:
        return 1.0  # 无上下文时，所有内容都视为幻觉

    context_text = "\n\n---\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    user_prompt = (
        f"问题：{question}\n\n"
        f"检索到的上下文（共 {len(contexts)} 条）：\n{context_text}\n\n"
        f"生成答案：\n{answer}"
    )

    try:
        result = json.loads(llm_ask_json(HALLUCINATION_SYSTEM, user_prompt))
        is_hallucination = result.get("is_hallucination", [])
        if not is_hallucination:
            return 0.0
        hallucination_count = sum(1 for h in is_hallucination if h)
        return hallucination_count / len(is_hallucination)
    except (json.JSONDecodeError, Exception):
        return 0.0
