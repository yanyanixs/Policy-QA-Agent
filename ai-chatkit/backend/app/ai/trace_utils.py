"""
RAG 查询追踪工具

供 policy_assistant 和 trace_query.py 共用。
通过 contextvars 在每个请求内收集追踪事件，最后通过 SSE custom 事件输出。
"""

import time
import contextvars
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    """单个追踪步骤"""
    step: str           # 步骤名称，如 "embedding", "chroma_search", "format", "llm_call"
    label: str          # 展示标题，如 "查询文本 → Embedding 向量"
    duration_ms: float  # 耗时（毫秒）
    detail: dict | None = None  # 附加详情


@dataclass
class TraceCollector:
    """收集一个请求内的所有追踪步骤"""
    query: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    answer: str = ""                               # LLM 最终回答
    tool_calls: list[dict] = field(default_factory=list)  # LLM 调用的工具及参数

    def add_step(self, step: str, label: str, duration_ms: float, detail: dict | None = None):
        self.steps.append(TraceStep(step=step, label=label, duration_ms=duration_ms, detail=detail))

    def set_answer(self, answer: str):
        self.answer = answer

    def set_tool_calls(self, tool_calls: list[dict]):
        self.tool_calls = tool_calls

    def to_dict(self) -> dict:
        """序列化为可 JSON 化的 dict"""
        return {
            "query": self.query,
            "total_ms": round(sum(s.duration_ms for s in self.steps), 1),
            "steps": [
                {
                    "step": s.step,
                    "label": s.label,
                    "duration_ms": round(s.duration_ms, 1),
                    "detail": s.detail,
                }
                for s in self.steps
            ],
            "tool_calls": self.tool_calls,
            "answer": self.answer,
        }


# 每个请求独立的 trace 收集器，通过 contextvars 隔离（异步安全）
_current_trace: contextvars.ContextVar[TraceCollector | None] = contextvars.ContextVar(
    "rag_trace", default=None
)


def start_trace(query: str) -> TraceCollector:
    """开始一次追踪，返回收集器"""
    collector = TraceCollector(query=query)
    _current_trace.set(collector)
    return collector


def get_trace() -> TraceCollector | None:
    """获取当前请求的追踪收集器"""
    return _current_trace.get(None)


def end_trace():
    """结束追踪"""
    _current_trace.set(None)


class TimedStep:
    """上下文管理器，自动记录步骤耗时"""

    def __init__(self, step: str, label: str, detail: dict | None = None):
        self.step = step
        self.label = label
        self.detail = detail
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        duration_ms = (time.time() - self._start) * 1000
        trace = get_trace()
        if trace:
            trace.add_step(self.step, self.label, duration_ms, self.detail)
