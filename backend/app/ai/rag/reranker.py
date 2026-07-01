"""
Reranker 模块 — 对 ChromaDB 粗筛结果精排 + MMR 多样性控制

支持三种后端（按精度降序）：
  1. Cross-Encoder (BGE-Reranker-v2-m3) — 本地推理，精度最高，需 ~2GB 模型
  2. LLM-based — 用现有 DeepSeek 排序，无需额外模型
  3. None — 跳过精排，保留原始向量相似度顺序

自动降级策略：
  如果 Cross-Encoder 模型加载/推理失败 → 自动降级到 LLM
  如果 LLM 排序失败 → 自动降级到 None（无精排）
  降级后单例会记住新后端，本会话不再重试失败的后端。

MMR (Maximum Marginal Relevance):
  在精排分数基础上, 对同一政策的多个 chunk 施加多样性惩罚,
  避免 top_k 被单一政策的 chunk 占满, 提高政策覆盖面。
  λ=0.7 时: 70% 权重给相关性, 30% 给多样性。

用法：
  from ai.rag.reranker import get_reranker

  reranker = get_reranker(backend="cross-encoder")  # 或 "llm" / "none"
  reranked = await reranker.rerank(query, docs, top_n=24, use_mmr=True, lambda_param=0.7)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── 模块加载时自动配置 HF 镜像 ─────────────────────
# 必须在 FlagEmbedding 导入之前设置 os.environ，
# 否则底层 huggingface_hub 会在 flagembedding 初始化时就直连 hf.co 超时
try:
    from core.config import settings

    if settings.HF_ENDPOINT and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
        logger.info(f"HF_ENDPOINT 已自动设置为 {settings.HF_ENDPOINT}")
except Exception:
    pass  # 独立导入时 settings 可能不可用，由 _rerank_cross_encoder 兜底


class Reranker:
    """文档重排序器，支持三级降级"""

    def __init__(self, backend: str = "cross-encoder"):
        """
        Args:
            backend: "cross-encoder" | "llm" | "none"
        """
        self.backend = backend
        self._model = None
        self._degraded_from: str | None = None  # 记录降级来源，用于日志

    # ── 公共接口 ──────────────────────────────────────

    async def rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int = 24,
        use_mmr: bool = False,
        lambda_param: float = 0.7,
    ) -> list[Document]:
        """
        对文档列表重新排序，返回 top_n 个最相关的文档。

        自动降级：cross-encoder → llm → none

        Args:
            query: 用户查询文本
            docs: 粗筛后的文档列表（通常 30 个）
            top_n: 返回数量
            use_mmr: 是否启用 MMR 多样性重排
            lambda_param: MMR 参数，0-1 之间。越大越偏重相关性

        Returns:
            按相关性（可选多样性）降序排列的文档列表
        """
        if not docs:
            return []

        # 参数校验
        if not 0.0 <= lambda_param <= 1.0:
            logger.warning(f"MMR lambda_param={lambda_param} 越界，已 clamp 到 [0, 1]")
            lambda_param = max(0.0, min(1.0, lambda_param))

        return await self._try_rerank(query, docs, top_n, use_mmr, lambda_param)

    async def _try_rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int,
        use_mmr: bool,
        lambda_param: float,
    ) -> list[Document]:
        """三级降级链：cross-encoder → llm → none"""

        # ── Level 1: Cross-Encoder ──
        if self.backend == "cross-encoder":
            try:
                return await self._rerank_cross_encoder(
                    query, docs, top_n, use_mmr, lambda_param
                )
            except Exception as e:
                self._degraded_from = "cross-encoder"
                self.backend = "llm"
                self._model = None  # 释放可能部分加载的模型
                logger.warning(
                    f"Cross-Encoder 不可用: {e}\n"
                    f"  已自动降级到 LLM 后端（无需额外模型）。\n"
                    f"  如需永久切换，在 .env 中设 RERANKER_BACKEND=llm"
                )

        # ── Level 2: LLM ──
        if self.backend == "llm":
            try:
                return await self._rerank_llm(
                    query, docs, top_n, use_mmr, lambda_param
                )
            except Exception as e:
                self._degraded_from = "llm"
                self.backend = "none"
                logger.warning(
                    f"LLM 排序失败: {e}\n"
                    f"  已自动降级到无精排模式（保留原始向量相似度顺序）。"
                )

        # ── Level 3: None（最终兜底）──
        return self._rerank_none(docs, top_n, use_mmr, lambda_param)

    # ── MMR 多样性重排 ────────────────────────────────

    @staticmethod
    def _apply_mmr(
        docs_with_scores: list[tuple[Document, float]],
        top_n: int,
        lambda_param: float,
    ) -> list[Document]:
        """
        MMR (Maximum Marginal Relevance) 多样性重排。

        在 policy_id 粒度做多样性控制：
        - 相同 policy_id 的文档 → 相似度 = 1.0 → 惩罚 (1-λ)
        - 不同 policy_id 的文档 → 相似度 = 0.0 → 无惩罚
        - metadata 缺少 policy_id → 视为唯一文档，相似度 = 0

        Args:
            docs_with_scores: 已按分数降序排列的 (doc, score) 对
            top_n: 选择数量
            lambda_param: 相关性权重 (0-1)，越高越偏重分数

        Returns:
            按 MMR 排序的文档列表
        """
        if not docs_with_scores or top_n <= 0:
            return []

        # 快速路径：无需 MMR
        if lambda_param >= 1.0:
            return [doc for doc, _ in docs_with_scores[:top_n]]

        # 快速路径：所有文档 policy_id 都不同或都相同 → MMR 退化
        all_pids = set(
            doc.metadata.get("policy_id", id(doc))
            for doc, _ in docs_with_scores
        )
        if len(all_pids) <= 1:
            return [doc for doc, _ in docs_with_scores[:top_n]]

        if top_n >= len(docs_with_scores):
            top_n = len(docs_with_scores)

        selected: list[int] = []            # 已选中的索引
        selected_pids: set = set()          # 已选中的 policy_id 集合
        remaining = list(range(len(docs_with_scores)))

        # 第一轮：无条件选分数最高的
        first_idx = remaining[0]  # 列表已按分数降序
        selected.append(first_idx)
        remaining.remove(first_idx)
        first_pid = docs_with_scores[first_idx][0].metadata.get("policy_id")
        if first_pid is not None:
            selected_pids.add(first_pid)

        # 贪心迭代
        while remaining and len(selected) < top_n:
            best_idx = -1
            best_mmr = float("-inf")

            for idx in remaining:
                doc, score = docs_with_scores[idx]
                pid = doc.metadata.get("policy_id")

                # 计算与已选文档的最大相似度
                if pid is not None and pid in selected_pids:
                    max_sim = 1.0
                else:
                    max_sim = 0.0

                mmr_val = lambda_param * score - (1.0 - lambda_param) * max_sim

                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = idx

            if best_idx == -1:
                break  # 理论上不会到这里

            selected.append(best_idx)
            remaining.remove(best_idx)
            pid = docs_with_scores[best_idx][0].metadata.get("policy_id")
            if pid is not None:
                selected_pids.add(pid)

        return [docs_with_scores[i][0] for i in selected]

    # ── 后端 1: Cross-Encoder ─────────────────────────

    async def _rerank_cross_encoder(
        self,
        query: str,
        docs: list[Document],
        top_n: int,
        use_mmr: bool = False,
        lambda_param: float = 0.7,
    ) -> list[Document]:
        """BGE-Reranker-v2-m3 精排（从 HF / 镜像 / 本地加载模型）"""
        if self._model is None:
            from FlagEmbedding import FlagReranker
            from core.config import settings

            model_path = settings.RERANKER_MODEL_PATH

            # HF_ENDPOINT 已在模块加载时设置，此处无需重复

            logger.info(f"正在加载 Cross-Encoder 模型: {model_path} ...")
            self._model = FlagReranker(
                model_path,
                use_fp16=False,  # CPU 推理；有 GPU 可改为 True
            )
            logger.info("Cross-Encoder 模型加载完成")

        # 构建 (query, doc) 对
        pairs = [[query, doc.page_content] for doc in docs]

        # compute_score 返回 list[float]，每个元素是相关性分数
        scores = self._model.compute_score(pairs, normalize=True)

        # 按分数降序排列，构造 (doc, score) 元组列表
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        if use_mmr:
            return self._apply_mmr(ranked, top_n, lambda_param)
        return [doc for doc, _ in ranked[:top_n]]

    # ── 后端 2: LLM 列表级排序 ────────────────────────

    async def _rerank_llm(
        self,
        query: str,
        docs: list[Document],
        top_n: int,
        use_mmr: bool = False,
        lambda_param: float = 0.7,
    ) -> list[Document]:
        """用现有 LLM (DeepSeek) 进行列表级排序"""
        from langchain_core.messages import HumanMessage, SystemMessage

        from ai.llm import get_model
        from core.config import settings

        # 为每个 chunk 编号，展示标题和内容摘要
        entries = []
        for i, doc in enumerate(docs):
            title = doc.metadata.get("title", "?")
            snippet = doc.page_content[:300].replace("\n", " ")
            entries.append(f"[{i}] {title}\n    {snippet}...")

        numbered_list = "\n\n".join(entries)

        system = (
            "你是一个政策检索系统的排序专家。"
            "你会收到一个用户查询和一组候选文档片段（已编号）。"
            "请按与查询的相关性从高到低排列这些文档。"
            "输出格式：每行一个数字（文档编号），不要输出其他内容。"
            "只输出前 N 个最相关的编号。"
        )
        human = (
            f"用户查询：{query}\n\n"
            f"候选文档（共 {len(docs)} 个）：\n{numbered_list}\n\n"
            f"请输出最相关的前 {top_n} 个文档编号（每行一个数字）："
        )

        model = get_model(settings.DEFAULT_MODEL)
        response = await model.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])

        # 解析 LLM 返回的编号列表
        ranked_docs = []
        seen = set()
        for line in str(response.content).strip().split("\n"):
            try:
                idx = int(line.strip().rstrip("."))
                if 0 <= idx < len(docs) and idx not in seen:
                    ranked_docs.append(docs[idx])
                    seen.add(idx)
            except ValueError:
                continue
            if len(ranked_docs) >= top_n:
                break

        # 兜底：如果 LLM 解析失败，保留原始顺序的前 top_n
        if not ranked_docs:
            logger.warning("LLM 排序解析失败，使用原始向量相似度顺序")
            return docs[:top_n]

        if use_mmr:
            # LLM 无真实分数，用排名位置推导伪分数 (第 1 名 = 1.0, 递减)
            n = len(ranked_docs)
            docs_with_scores = [
                (doc, 1.0 - i / max(n, 1))
                for i, doc in enumerate(ranked_docs)
            ]
            return self._apply_mmr(docs_with_scores, top_n, lambda_param)

        return ranked_docs[:top_n]

    # ── 后端 3: 无精排（终极兜底）────────────────────

    @staticmethod
    def _rerank_none(
        docs: list[Document],
        top_n: int,
        use_mmr: bool = False,
        lambda_param: float = 0.7,
    ) -> list[Document]:
        """
        无精排模式：保留 ChromaDB 原始向量相似度顺序。
        如果启用 MMR，用排名位置推导伪分数后再做多样性重排。
        """
        if use_mmr and len(docs) > 1:
            # 用排名位置推导伪分数（第 1 名 ≈ 1.0，递减）
            n = len(docs)
            docs_with_scores = [
                (doc, 1.0 - i / max(n, 1))
                for i, doc in enumerate(docs)
            ]
            return Reranker._apply_mmr(docs_with_scores, top_n, lambda_param)

        return docs[:top_n]


# ── 模块级单例 ────────────────────────────────────────

_reranker: Reranker | None = None


def get_reranker(backend: str = "cross-encoder") -> Reranker:
    """
    获取 Reranker 单例。

    注意：如果运行时发生降级，单例的 backend 会被更新为降级后的值，
    后续调用不会重试失败的后端。

    Args:
        backend: "cross-encoder" | "llm" | "none"
    """
    global _reranker
    if _reranker is None or _reranker.backend != backend:
        _reranker = Reranker(backend=backend)
    return _reranker
