"""
RAG 查询全流程诊断/追踪脚本

逐步骤展示一次查询在 RAG 管道中的完整路径：
  Query → Embedding → ChromaDB 检索 → 去重格式化 → LLM 生成

用法:
  cd backend
  python scripts/trace_query.py "新能源补贴有哪些政策"
  python scripts/trace_query.py "财政部2024年的税收政策" --authority 财政部 --year 2024
  python scripts/trace_query.py "上海AI产业政策" --top_k 5 --verbose
"""

import os
import sys
import json
import argparse
import time
import asyncio

# 确保 stdout 使用 UTF-8，避免 Windows GBK 编码错误
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.join(_SCRIPT_DIR, "..", "app")
sys.path.insert(0, os.path.abspath(_APP_DIR))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from db.database import async_session_maker
from db.repository.policy_repo import PolicyRepository
from ai.rag.chromaClient import policy_vector_store, embeddings
from ai.tools.policy_tools import (
    search_policies,
    get_policy_detail,
    list_policies_by_metadata,
    _build_chroma_filter,
    _format_chunk_results,
)
from ai.llm import get_model
from ai.trace_utils import TimedStep, start_trace, get_trace, end_trace
from ai.rag.reranker import get_reranker
from core.config import settings

# ─── 辅助函数 ───────────────────────────────────────────────

SEP = "=" * 72
SUB = "-" * 48


def print_header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(f"{SEP}")


def print_step(n: int, title: str):
    print(f"\n  [{n}] {title}")
    print(f"  {SUB}")


def print_kv(key: str, value, indent: int = 4):
    prefix = " " * indent
    if isinstance(value, str) and len(value) > 500:
        value = value[:500] + f"\n{prefix}... [截断，总长 {len(value)} 字符]"
    elif isinstance(value, list):
        print(f"{prefix}{key}: [{len(value)} 条]")
        for i, item in enumerate(value):
            s = str(item)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"{prefix}  [{i}] {s}")
        return
    elif isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            print(f"{prefix}  {k}: {v}")
        return
    print(f"{prefix}{key}: {value}")


# ─── 核心追踪 ───────────────────────────────────────────────


async def trace_query(
    query: str,
    *,
    issuing_authority: str | None = None,
    location: str | None = None,
    policy_tool: str | None = None,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_k: int = 8,
    verbose: bool = False,
    dry_run: bool = False,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
):
    """
    逐步追踪一次 RAG 查询的完整流程。

    流程与生产环境完全一致：
      Step 1: 系统提示词 + LLM 初次决策（可能改写 query 并决定调用工具）
      Step 2-6: 用 LLM 改写后的 query 跑完整 RAG 管线
      Step 7: 检索结果发给 LLM → 生成最终回答
    """

    # ══════════════════════════════════════════════════════════
    print_header("RAG 查询全流程追踪（生产环境等价）")
    start_trace(query)
    print_kv("用户查询", query)
    print_kv("过滤条件", {
        "issuing_authority": issuing_authority,
        "location": location,
        "policy_tool": policy_tool,
        "category": category,
        "start_date": start_date,
        "end_date": end_date,
    })
    print_kv("默认 top_k", top_k)
    print_kv("LLM 模型", settings.DEFAULT_MODEL)
    print_kv("Embedding 模型", settings.EMBEDDING_MODEL)
    print_kv("ChromaDB 路径", settings.CHROMA_PATH)
    print_kv("Reranker", settings.RERANKER_BACKEND)
    print_kv("MMR", f"启用={use_mmr}, λ={mmr_lambda}")
    if dry_run:
        print("\n  [DRY RUN] 跳过 LLM 调用\n")

    from datetime import datetime

    def _print_trace_summary():
        """输出追踪摘要（trace 已结束或提前退出时使用）"""
        trace = get_trace()
        if trace:
            td = trace.to_dict()
            print(f"  总耗时: {td['total_ms']:.0f}ms")
            for s in td["steps"]:
                print(f"    {s['label']}: {s['duration_ms']:.0f}ms")
            if td.get("tool_calls"):
                print(f"\n  LLM 工具调用:")
                for tc in td["tool_calls"]:
                    print(f"    - {tc['name']}({json.dumps(tc.get('args',{}), ensure_ascii=False)})")
            if td.get("answer"):
                print(f"\n  LLM 最终回答 ({len(td['answer'])} 字符):")
                preview = td["answer"][:500]
                for line in preview.split("\n"):
                    print(f"    | {line}")
                if len(td["answer"]) > 500:
                    print(f"    | ... [共 {len(td['answer'])} 字符，已截断]")
        print(f"{SEP}")

    # ══════════════════════════════════════════════════════════
    # Step 1: 系统提示词 + LLM 初次决策
    # ══════════════════════════════════════════════════════════
    print_step(1, "系统提示词 + LLM 初次决策")

    instructions = f"""
你是一个政策问答智能助手，帮助用户搜索和理解政府发布的各类政策法规。

你可以调用以下工具来获取信息：
1. **search_policies**：语义搜索政策知识库。适合开放性问答（如"新能源补贴有哪些政策"）。支持按发布机关、地区、政策工具类型、日期等过滤。
2. **get_policy_detail**：获取某条政策的完整正文。适合用户想看全文或确认细节时使用。
3. **list_policies_by_metadata**：按结构化字段精确浏览政策列表。适合"列出财政部2024年的所有政策"这类查询。

### 工作准则：
- **必须标注来源**：每次回答都必须注明政策名称、发布机关和发布日期。引用正文时需标明政策标题。
- **禁止编造**：不得凭空捏造政策条款。如知识库中无相关答案，请如实告知用户。
- **善用过滤**：用户提及具体机关（如"财政部"）、地区（如"上海"）、政策类型（如"税收优惠"）时，用对应参数精确过滤。
- **处理模糊问题**：用户问题不够明确时，可追问地区、机关、政策类型等来缩小范围。
- **结构化回答**：涉及多条政策时按相关度排列，每条列出标题、机关、日期和核心内容摘要。
- **日期处理**：用户说"今年"指 {datetime.now().year} 年，"去年"指 {datetime.now().year - 1} 年，"最近"指近半年内。

当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    print(f"\n{instructions.strip()}\n")

    if dry_run:
        print("  [SKIP] DRY RUN: 跳过 LLM 调用，直接使用原始 query 检索\n")
        search_query = query
        search_top_k = top_k
    else:
        print(f"  正在调用 LLM ({settings.DEFAULT_MODEL})...")
        model = get_model(settings.DEFAULT_MODEL)
        model_with_tools = model.bind_tools([search_policies, get_policy_detail, list_policies_by_metadata])
        messages = [SystemMessage(content=instructions), HumanMessage(content=query)]

        with TimedStep("llm_decision", "LLM 初次决策（选择工具 + 改写 query）"):
            response = await model_with_tools.ainvoke(messages)

        print(f"\n  ── LLM 响应 ──")

        if hasattr(response, "tool_calls") and response.tool_calls:
            print_kv("类型", "Tool Call（LLM 决定调用工具）")
            for tc in response.tool_calls:
                print(f"\n    工具: {tc['name']}")
                print(f"    参数: {json.dumps(tc['args'], ensure_ascii=False, indent=6)}")

            # 记录 tool_calls 到 trace
            trace = get_trace()
            if trace:
                trace.set_tool_calls([
                    {"name": tc["name"], "args": tc["args"]}
                    for tc in response.tool_calls
                ])

            # 从 LLM 的 tool_call 中提取 query 和 top_k
            search_tc = None
            for tc in response.tool_calls:
                if tc["name"] == "search_policies":
                    search_tc = tc
                    break

            if search_tc is None:
                print("\n  [INFO] LLM 未调用 search_policies，无需检索")
                # 直接返回 LLM 回答
                trace = get_trace()
                if trace and hasattr(response, "content"):
                    trace.set_answer(response.content or "")
                print_header("追踪完成")
                _print_trace_summary()
                return

            search_query = search_tc["args"].get("query", query)
            search_top_k = search_tc["args"].get("top_k", top_k)
            print(f"\n    → LLM 改写 query: \"{query}\" → \"{search_query}\"")
            print(f"    → LLM 指定 top_k: {search_top_k}")
        else:
            # 直接回答，无工具调用
            print_kv("类型", "直接回答（无工具调用）")
            print(f"\n    ── 回答内容 ──")
            print(f"    {response.content}")
            trace = get_trace()
            if trace and hasattr(response, "content"):
                trace.set_answer(response.content or "")
            print_header("追踪完成")
            _print_trace_summary()
            return

    # ══════════════════════════════════════════════════════════
    # Step 2-5: RAG 管线（与 search_policies 内部完全一致）
    # ══════════════════════════════════════════════════════════

    # ── Step 2: Embedding ────────────────────────────────
    print_step(2, f"查询文本 → Embedding 向量 (query=\"{search_query}\")")

    with TimedStep("embedding", "查询文本 → Embedding 向量"):
        query_vector = embeddings.embed_query(search_query)

    print_kv("Embedding 模型", "BGE-M3 (Ollama)")
    print_kv("向量维度", len(query_vector))
    print_kv("向量前 10 维", query_vector[:10])

    # ── Step 3: ChromaDB 过滤 + 检索 ───────────────────
    print_step(3, "ChromaDB 向量相似度检索")

    chroma_filter = _build_chroma_filter(
        issuing_authority=issuing_authority,
        location=location,
        policy_tool=policy_tool,
        category=category,
        start_date=start_date,
        end_date=end_date,
    )
    raw_k = min(search_top_k * 3, 30)
    print_kv("过滤条件", json.dumps(chroma_filter, ensure_ascii=False, indent=4) if chroma_filter else "无（全库搜索）")
    print_kv("检索策略", f"内部取 k={raw_k}（top_k={search_top_k} × 3，上限 30）")

    with TimedStep("chroma_search", "ChromaDB 向量相似度检索",
                   detail={"k": raw_k, "filter": bool(chroma_filter)}):
        raw_results = policy_vector_store.similarity_search_by_vector(
            query_vector,
            k=raw_k,
            filter=chroma_filter,
        )

    print_kv("原始返回 chunk 数", len(raw_results))

    if not raw_results:
        print("\n  [WARN] 未检索到任何结果！")
        return

    print(f"\n    原始 chunks（按向量相似度排序，共 {len(raw_results)}）:")
    for i, doc in enumerate(raw_results):
        pid = doc.metadata.get("policy_id")
        title = doc.metadata.get("title", "?")
        ci = doc.metadata.get("chunk_index", "?")
        if verbose:
            text_preview = doc.page_content[:80].replace("\n", " ")
            print(f"    [{i}] policy_id={pid} chunk={ci} | {title}")
            print(f"        text: {text_preview}...")
        else:
            print(f"    [{i}] policy_id={pid} chunk={ci} | {title}")

    # ── Step 4: Rerank + MMR ───────────────────────────
    print_step(4, f"Reranker ({settings.RERANKER_BACKEND}) + MMR 多样性 (λ={mmr_lambda})")

    rerank_top_n = min(search_top_k * 3, len(raw_results))
    with TimedStep("rerank_mmr", "Reranker 精排 + MMR 多样性",
                   detail={"input": len(raw_results), "output": rerank_top_n, "mmr": use_mmr}):
        reranker = get_reranker(backend=settings.RERANKER_BACKEND)
        results = await reranker.rerank(
            search_query, raw_results, top_n=rerank_top_n,
            use_mmr=use_mmr, lambda_param=mmr_lambda,
        )

    pre_rerank_pids = set(doc.metadata.get("policy_id") for doc in raw_results)
    post_rerank_pids = set(doc.metadata.get("policy_id") for doc in results)
    print_kv("精排输入 chunk 数", len(raw_results))
    print_kv("精排输出 chunk 数", len(results))
    print_kv("精排前不同 policy 数", len(pre_rerank_pids))
    print_kv("精排后不同 policy 数", len(post_rerank_pids))

    print(f"\n    精排后 chunks（按 Reranker 得分排序，共 {len(results)}）:")
    for i, doc in enumerate(results):
        title = doc.metadata.get("title", "?")
        pid = doc.metadata.get("policy_id", "?")
        ci = doc.metadata.get("chunk_index", "?")
        if verbose:
            text_preview = doc.page_content[:80].replace("\n", " ")
            print(f"    [{i}] policy_id={pid} chunk={ci} | {title}")
            print(f"        text: {text_preview}...")
        else:
            print(f"    [{i}] policy_id={pid} chunk={ci} | {title}")

    # ── Step 5: 去重 + 格式化 ──────────────────────────
    print_step(5, "去重与格式化（_format_chunk_results）")

    unique_pids = set()
    for doc in results:
        unique_pids.add(doc.metadata.get("policy_id"))
    with TimedStep("format", "去重与格式化",
                   detail={"raw_chunks": len(results), "unique_policies": len(unique_pids), "top_k": search_top_k}):
        formatted = _format_chunk_results(results, search_top_k)

    print_kv("去重前 chunk 数", len(results))
    print_kv("去重后 policy 数", len(unique_pids))
    print_kv("最终输出 policy 数", min(len(unique_pids), search_top_k))

    print(f"\n    ── 格式化输出（传给 LLM 的内容，共 {len(formatted)} 字符）──")
    print(f"    |")
    for line in formatted.split("\n"):
        print(f"    | {line}")
    print(f"    |")

    # ══════════════════════════════════════════════════════════
    # Step 6: LLM 生成最终回答
    # ══════════════════════════════════════════════════════════

    if dry_run:
        print_step(6, "LLM 生成最终回答 [SKIP — dry-run]")
    else:
        print_step(6, "LLM 生成最终回答（检索结果 + 系统提示词 → LLM）")

        with TimedStep("llm_generate", "LLM 综合生成最终回答"):
            messages.append(response)
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=formatted, tool_call_id=search_tc["id"]))
            final_response = await model_with_tools.ainvoke(messages)

        print(f"\n    ── 最终回答 ──")
        print(f"    {final_response.content}")

        trace = get_trace()
        if trace:
            trace.set_answer(final_response.content or "")

    # ══════════════════════════════════════════════════════════
    print_header("追踪完成")
    trace = get_trace()
    if trace:
        td = trace.to_dict()
        print(f"  总耗时: {td['total_ms']:.0f}ms")
        for s in td["steps"]:
            print(f"    {s['label']}: {s['duration_ms']:.0f}ms")
        if td.get("tool_calls"):
            print(f"\n  LLM 工具调用:")
            for tc in td["tool_calls"]:
                print(f"    - {tc['name']}({json.dumps(tc.get('args',{}), ensure_ascii=False)})")
        if td.get("answer"):
            print(f"\n  LLM 最终回答 ({len(td['answer'])} 字符):")
            preview = td["answer"][:500]
            for line in preview.split("\n"):
                print(f"    | {line}")
            if len(td["answer"]) > 500:
                print(f"    | ... [共 {len(td['answer'])} 字符，已截断]")
    print(f"{SEP}")


# ─── 入口 ───────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(
        description="RAG 查询全流程追踪工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/trace_query.py "新能源补贴有哪些政策"
  python scripts/trace_query.py "财政部税收政策" --authority 财政部
  python scripts/trace_query.py "上海AI产业" --location 上海市 --top_k 5
  python scripts/trace_query.py "新能源" --dry-run
        """,
    )
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--authority", default=None, help="发布机关过滤")
    parser.add_argument("--location", default=None, help="地区过滤")
    parser.add_argument("--tool", dest="policy_tool", default=None, help="政策工具过滤")
    parser.add_argument("--category", default=None, help="分类过滤")
    parser.add_argument("--start-date", default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="截止日期 YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=8, help="返回结果数 (默认 8)")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示每个 chunk 详情")
    parser.add_argument("--dry-run", action="store_true", help="仅追踪检索，不调用 LLM")
    parser.add_argument("--no-mmr", action="store_true", help="禁用 MMR 多样性，仅用 Reranker 相关性排序")
    parser.add_argument("--mmr-lambda", type=float, default=0.7,
                        help="MMR λ 参数，0-1 之间，越大越偏重相关性 (默认 0.7)")
    args = parser.parse_args()

    await trace_query(
        args.query,
        issuing_authority=args.authority,
        location=args.location,
        policy_tool=args.policy_tool,
        category=args.category,
        start_date=args.start_date,
        end_date=args.end_date,
        top_k=args.top_k,
        verbose=args.verbose,
        dry_run=args.dry_run,
        use_mmr=not args.no_mmr,
        mmr_lambda=args.mmr_lambda,
    )


if __name__ == "__main__":
    asyncio.run(main())
