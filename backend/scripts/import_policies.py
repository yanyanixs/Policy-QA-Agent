"""
政策数据导入脚本

功能：将 JSON 格式的政策数据导入到 SQLite（结构化存储）和 ChromaDB（向量检索）。
JSON 文件格式：一个数组，每个元素包含以下字段：
  {
    "标题": "政策标题",
    "日期": "YYYY-MM-DD",
    "发布机关": "国家机关名称",
    "正文": "政策正文全文...",
    "地点": "全国/北京/上海等（可选）",
    "政策工具": "财政补贴/税收优惠/行政审批等（可选）",
    "分类": "政策分类（可选）",
    "发文字号": "文号（可选）",
    "生效日期": "YYYY-MM-DD（可选）",
    "原文链接": "URL（可选）",
    "状态": "active/expired/repealed（可选，默认 active）"
  }

用法：
  python scripts/import_policies.py policies.json [--clean]
    --clean  清空旧数据后重新导入
"""

import os
import sys
import json
import argparse

# 将 backend/app/ 添加到 Python 路径（项目的导入根目录）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.join(_SCRIPT_DIR, "..", "app")
sys.path.insert(0, os.path.abspath(_APP_DIR))

from db.database import async_session_maker, async_engine, create_db_and_tables
from db.models.policy import Policy
from db.repository.policy_repo import PolicyRepository
from ai.rag.chromaClient import policy_vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import asyncio


# -------------------- 分块配置 --------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n\n", "\n", "。", ".", "；", ";", "，", ",", " ", ""]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=CHUNK_SEPARATORS,
    length_function=len,
    is_separator_regex=False,
)


# -------------------- 导入逻辑 --------------------

def load_policies_from_json(filepath: str) -> list[dict]:
    """从 JSON 文件加载政策列表"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件内容必须是一个数组")

    policies = []
    for idx, item in enumerate(data):
        policy = {
            "title": item.get("标题", f"未命名政策_{idx}"),
            "issuing_authority": item.get("发布机关", "未知"),
            "doc_number": item.get("发文字号"),
            "publish_date": item.get("日期"),
            "effective_date": item.get("生效日期"),
            "location": item.get("地点"),
            "policy_tool": item.get("政策工具"),
            "category": item.get("分类"),
            "status": item.get("状态", "active"),
            "source_url": item.get("原文链接"),
            "body_text": item.get("正文", ""),
        }
        policies.append(policy)

    return policies


async def clean_existing_data():
    """清空已有的政策和向量数据"""
    print("[clean] 清空旧数据...")

    # 清空 ChromaDB 的 policies 集合中的文档
    try:
        existing = policy_vector_store.get()
        ids = existing.get("ids", [])
        if ids:
            policy_vector_store.delete(ids=ids)
            print(f"  - ChromaDB 已清空 {len(ids)} 个向量")
        else:
            print("  - ChromaDB 集合为空，无需清理")
    except Exception as e:
        print(f"  - ChromaDB 清理失败: {e}")

    # 清空 SQLite 的 policy 表
    async with async_session_maker() as session:
        policies = await PolicyRepository.get_all_policies(session)
        for p in policies:
            await session.delete(p)
        await session.commit()
        print(f"  - SQLite policy 表已清空（删除了 {len(policies)} 条记录）")


async def import_policies(policies: list[dict], source_file: str) -> tuple[int, int]:
    """
    将政策数据写入 SQLite 并分块存入 ChromaDB。
    返回 (policy_count, chunk_count)
    """
    total_chunks = 0

    for idx, pdata in enumerate(policies):
        async with async_session_maker() as session:
            # 1. 写入 SQLite
            policy = Policy(
                title=pdata["title"],
                issuing_authority=pdata["issuing_authority"],
                doc_number=pdata.get("doc_number"),
                publish_date=pdata.get("publish_date"),
                effective_date=pdata.get("effective_date"),
                location=pdata.get("location"),
                policy_tool=pdata.get("policy_tool"),
                category=pdata.get("category"),
                status=pdata.get("status", "active"),
                source_url=pdata.get("source_url"),
                body_text=pdata["body_text"],
            )
            policy = await PolicyRepository.create_policy(session, policy)
            policy_id = policy.id

            print(f"[{idx + 1}/{len(policies)}] {policy.title} (id={policy_id})")

        # 2. 切分正文
        body = pdata["body_text"]
        if not body:
            print(f"  [WARN] 正文为空，跳过向量化")
            continue

        chunks = text_splitter.split_text(body)
        print(f"  正文 {len(body)} 字 → {len(chunks)} 个 chunk")

        # 3. 构建带元数据的 Document 并写入 ChromaDB
        documents = []
        for ci, chunk_text in enumerate(chunks):
            metadata = {
                "policy_id": policy_id,
                "title": pdata["title"],
                "issuing_authority": pdata["issuing_authority"],
                "publish_date": pdata.get("publish_date") or "",
                "location": pdata.get("location") or "",
                "policy_tool": pdata.get("policy_tool") or "",
                "category": pdata.get("category") or "",
                "doc_number": pdata.get("doc_number") or "",
                "source_file": source_file,
                "chunk_index": ci,
                "total_chunks": len(chunks),
            }
            documents.append(Document(page_content=chunk_text, metadata=metadata))

        policy_vector_store.add_documents(documents)
        total_chunks += len(chunks)

    return len(policies), total_chunks


async def main():
    parser = argparse.ArgumentParser(description="政策数据导入工具")
    parser.add_argument("file", help="JSON 格式的政策数据文件")
    parser.add_argument("--clean", action="store_true", help="导入前清空旧数据")
    args = parser.parse_args()

    json_path = os.path.abspath(args.file)
    if not os.path.exists(json_path):
        print(f"[ERROR] 文件不存在: {json_path}")
        sys.exit(1)

    print(f"[读取] {json_path}")
    policies = load_policies_from_json(json_path)
    print(f"[加载] 共 {len(policies)} 条政策\n")

    # 确保数据库表已创建
    await create_db_and_tables()
    print("[建表] 数据库表已就绪\n")

    if args.clean:
        await clean_existing_data()
        print()

    source_file = os.path.basename(json_path)
    policy_count, chunk_count = await import_policies(policies, source_file)

    print(f"\n[完成] 导入 {policy_count} 条政策，{chunk_count} 个向量片段")


if __name__ == "__main__":
    asyncio.run(main())
