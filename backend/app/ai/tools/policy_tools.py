"""
政策问答智能体工具集

包含三个工具：
1. search_policies — 语义搜索 + 元数据过滤（ChromaDB）
2. get_policy_detail — 获取政策全文（SQLite）
3. list_policies_by_metadata — 按元数据精确浏览（SQLite）
"""

from langchain_core.tools import tool

from core.config import settings
from db.database import async_session_maker
from db.repository.policy_repo import PolicyRepository
from ai.rag.chromaClient import policy_vector_store, embeddings
from ai.trace_utils import TimedStep, get_trace


def _build_chroma_filter(
    issuing_authority: str | None = None,
    location: str | None = None,
    policy_tool: str | None = None,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict | None:
    """构建 ChromaDB where 过滤条件"""
    filter_parts = []

    if issuing_authority:
        filter_parts.append({"issuing_authority": {"$eq": issuing_authority}})
    if location:
        filter_parts.append({"location": {"$eq": location}})
    if policy_tool:
        filter_parts.append({"policy_tool": {"$eq": policy_tool}})
    if category:
        filter_parts.append({"category": {"$eq": category}})

    # 日期范围合并到一个条件中（ChromaDB 同字段不能分开放 $and）
    date_filter = {}
    if start_date:
        date_filter["$gte"] = start_date
    if end_date:
        date_filter["$lte"] = end_date
    if date_filter:
        filter_parts.append({"publish_date": date_filter})

    if not filter_parts:
        return None
    if len(filter_parts) == 1:
        return filter_parts[0]
    return {"$and": filter_parts}


# 单篇政策合并后最大展示字符数（避免单篇占比过高，保留检索多样性）
_MAX_CONTENT_PER_POLICY = 3000


def _format_chunk_results(docs: list, top_k: int) -> str:
    """将检索结果格式化为 LLM 友好的文本

    改进策略：
    1. 按 policy_id 分组，合并同一政策的所有命中 chunk
    2. 合并后的 chunk 按 chunk_index 排序，还原原文顺序
    3. 每篇政策最多展示 _MAX_CONTENT_PER_POLICY 字符
    4. 最终输出 top_k 篇政策
    """
    if not docs:
        return "未找到相关政策。请尝试更换查询词或放宽过滤条件。"

    # ── 按 policy_id 分组，收集命中 chunks ──
    grouped: dict[int, dict] = {}  # policy_id → {meta, chunks: [(chunk_index, text)]}
    for doc in docs:
        pid = doc.metadata.get("policy_id")
        if pid is None:
            continue

        if pid not in grouped:
            grouped[pid] = {
                "meta": doc.metadata,
                "chunks": [],
            }

        ci = doc.metadata.get("chunk_index", 0)
        grouped[pid]["chunks"].append((ci, doc.page_content))

    # ── 按 top_k 截取（保持原始 docs 的相似度顺序） ──
    seen_ids = set()
    ordered_pids = []
    for doc in docs:
        pid = doc.metadata.get("policy_id")
        if pid is None or pid in seen_ids or pid not in grouped:
            continue
        seen_ids.add(pid)
        ordered_pids.append(pid)
        if len(ordered_pids) >= top_k:
            break

    # ── 格式化输出 ──
    lines = []
    for count, pid in enumerate(ordered_pids, 1):
        info = grouped[pid]
        meta = info["meta"]

        lines.append(f"--- 政策 {count} ---")
        lines.append(f"标题：{meta.get('title', '未知')}")
        lines.append(f"发布机关：{meta.get('issuing_authority', '未知')}")
        lines.append(f"发布日期：{meta.get('publish_date', '未知')}")
        if meta.get("location"):
            lines.append(f"适用地区：{meta['location']}")
        if meta.get("policy_tool"):
            lines.append(f"政策工具：{meta['policy_tool']}")
        if meta.get("doc_number"):
            lines.append(f"发文字号：{meta['doc_number']}")
        if meta.get("source_url"):
            lines.append(f"原文链接：{meta['source_url']}")
        lines.append(f"政策ID：{pid}")

        # 合并 chunks：按 chunk_index 排序后拼接
        info["chunks"].sort(key=lambda x: x[0])
        merged = ""
        for _, chunk_text in info["chunks"]:
            merged += chunk_text
        total_len = len(merged)

        # 截断过长内容，优先保留前部（chunk_index 靠前的内容通常包含核心条款）
        if len(merged) > _MAX_CONTENT_PER_POLICY:
            merged = merged[:_MAX_CONTENT_PER_POLICY] + f"\n... [已截断，全文共 {total_len} 字符]"
            lines.append(f"命中片段：{len(info['chunks'])} 处，全文 {total_len} 字符，以下为前 {_MAX_CONTENT_PER_POLICY} 字符")
        else:
            lines.append(f"命中片段：{len(info['chunks'])} 处，合并后 {total_len} 字符")

        lines.append(f"相关内容：{merged}")
        lines.append("")

    return "\n".join(lines)


# -------------------- 工具定义 --------------------


@tool
async def search_policies(
    query: str,
    issuing_authority: str | None = None,
    location: str | None = None,
    policy_tool: str | None = None,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_k: int = 8,
) -> str:
    """
    在政策知识库中进行语义搜索，支持按元数据过滤。

    当用户问关于政策内容的问题时使用此工具（例如"关于新能源的补贴政策有哪些"）。
    语义搜索会根据含义匹配相关内容，即使关键词不完全一致也能找到。

    参数：
    - query: 搜索查询，描述用户想了解的政策内容（例如 "新能源汽车补贴政策"）
    - issuing_authority: 按发布机关过滤（如 "国务院"、"财政部"、"国家能源局"）
    - location: 按适用地区过滤（如 "全国"、"北京市"、"广东省"）
    - policy_tool: 按政策工具类型过滤（如 "财政补贴"、"税收优惠"、"行政审批"）
    - category: 按政策分类过滤（如 "能源"、"环保"、"科技"）
    - start_date: 起始日期（YYYY-MM-DD）
    - end_date: 截止日期（YYYY-MM-DD）
    - top_k: 返回结果数量（默认 8）

    返回包含来源标注（政策标题、发布机关、发布日期）的检索结果。
    """
    chroma_filter = _build_chroma_filter(
        issuing_authority=issuing_authority,
        location=location,
        policy_tool=policy_tool,
        category=category,
        start_date=start_date,
        end_date=end_date,
    )

    # 1. 查询向量化
    with TimedStep("embedding", "查询文本 → Embedding 向量"):
        query_vector = embeddings.embed_query(query)

    # 2. ChromaDB 向量检索
    raw_k = min(top_k * 3, 30)
    with TimedStep("chroma_search", "ChromaDB 向量相似度检索",
                   detail={"k": raw_k, "filter": bool(chroma_filter)}):
        results = policy_vector_store.similarity_search_by_vector(
            query_vector,
            k=raw_k,
            filter=chroma_filter,
        )

    # 3. Rerank + MMR 精排（粗筛 → 精排 → 多样性）
    with TimedStep("rerank", "Reranker 精排 + MMR 多样性",
                   detail={"input_chunks": len(results), "mmr": True, "lambda": 0.7}):
        from ai.rag.reranker import get_reranker
        reranker = get_reranker(backend=settings.RERANKER_BACKEND)  # 从 .env 读取，默认 cross-encoder
        rerank_top_n = min(top_k * 3, len(results))       # 精排后保留 top_k * 3 给去重留余量
        results = await reranker.rerank(
            query, results, top_n=rerank_top_n,
            use_mmr=True, lambda_param=0.7,
        )

    # 收集检索到的 policy 分布（用于 trace）
    from collections import Counter
    hit_titles = Counter(doc.metadata.get("title", "?") for doc in results)
    hit_summary = [{"title": t[:60], "chunks": c} for t, c in hit_titles.most_common()]

    # 4. 格式化
    unique_pids = len(set(doc.metadata.get("policy_id") for doc in results))
    with TimedStep("format", "去重与格式化",
                   detail={
                       "raw_chunks": len(results),
                       "unique_policies": unique_pids,
                       "top_k": top_k,
                       "policies": hit_summary,  # 新增：命中政策清单
                   }):
        formatted = _format_chunk_results(results, top_k)

    return formatted


@tool
async def get_policy_detail(policy_id: int) -> str:
    """
    获取某条政策的完整正文和元数据。

    当用户在搜索结果中看到某条政策，想要阅读全文时使用此工具。
    也可以用于对比多条政策的详细内容。

    参数：
    - policy_id: 政策在数据库中的 ID

    返回政策的完整元数据和正文全文。
    """
    async with async_session_maker() as session:
        policy = await PolicyRepository.get_policy(session, policy_id)

    if not policy:
        return f"未找到 ID 为 {policy_id} 的政策。"

    parts = [
        f"========== 政策详情 ==========",
        f"标题：{policy.title}",
        f"发布机关：{policy.issuing_authority}",
        f"发布日期：{policy.publish_date or '未知'}",
    ]
    if policy.doc_number:
        parts.append(f"发文字号：{policy.doc_number}")
    if policy.effective_date:
        parts.append(f"生效日期：{policy.effective_date}")
    if policy.location:
        parts.append(f"适用地区：{policy.location}")
    if policy.policy_tool:
        parts.append(f"政策工具：{policy.policy_tool}")
    if policy.category:
        parts.append(f"分类：{policy.category}")
    if policy.status:
        parts.append(f"状态：{policy.status}")
    if policy.source_url:
        parts.append(f"原文链接：{policy.source_url}")
    parts.append(f"\n---------- 正文 ----------")
    parts.append(policy.body_text)
    parts.append(f"\n==============================")

    return "\n".join(parts)


@tool
async def list_policies_by_metadata(
    issuing_authority: str | None = None,
    location: str | None = None,
    policy_tool: str | None = None,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    title_keyword: str | None = None,
    limit: int = 20,
) -> str:
    """
    按元数据字段精确浏览政策列表，不进行语义搜索。

    当用户的问题类似下列情况时使用此工具：
    - "财政部2024年发布了哪些政策？"
    - "列出所有适用上海的政策"
    - "有哪些税收优惠类的政策？"

    此工具直接查询结构化数据库，适合按条件筛选/列举场景。

    参数：
    - issuing_authority: 按发布机关过滤（如 "国务院"、"财政部"）
    - location: 按适用地区过滤（如 "全国"、"上海市"）
    - policy_tool: 按政策工具类型过滤（如 "税收优惠"、"财政补贴"）
    - category: 按分类过滤
    - start_date: 起始日期（YYYY-MM-DD）
    - end_date: 截止日期（YYYY-MM-DD）
    - title_keyword: 按标题关键词模糊匹配
    - limit: 最大返回条数（默认 20）

    返回匹配的政策列表（仅元数据，不含正文）。
    """
    async with async_session_maker() as session:
        policies = await PolicyRepository.search_policies(
            session,
            title_like=title_keyword,
            issuing_authority=issuing_authority,
            location=location,
            policy_tool=policy_tool,
            category=category,
            date_from=start_date,
            date_to=end_date,
            limit=limit,
        )

    if not policies:
        return "未找到符合条件的政策。"

    lines = [f"共找到 {len(policies)} 条政策：", ""]
    for i, p in enumerate(policies, 1):
        lines.append(f"{i}. [{p.publish_date or '日期未知'}] {p.title}")
        lines.append(f"   发布机关：{p.issuing_authority}  |  政策ID：{p.id}")
        if p.location:
            lines.append(f"   适用地区：{p.location}")
        if p.policy_tool:
            lines.append(f"   政策工具：{p.policy_tool}")
        lines.append("")

    return "\n".join(lines)
