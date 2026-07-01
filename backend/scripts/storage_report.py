"""
[存储报告] 政策数据库和向量库状态
"""
import os
import sys
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.join(_SCRIPT_DIR, "..", "app")
sys.path.insert(0, os.path.abspath(_APP_DIR))

from db.database import async_session_maker
from db.repository.policy_repo import PolicyRepository
from ai.rag.chromaClient import policy_vector_store
from collections import Counter
import asyncio


async def report():
    print("=" * 60)
    print("  政策问答智能体 — 存储状况报告")
    print("=" * 60)

    # ------------------ SQLite ------------------
    print("\n[数据库] SQLite")
    print("-" * 40)
    async with async_session_maker() as session:
        policies = await PolicyRepository.get_all_policies(session)

        # 文件信息
        db_path = os.path.join(_APP_DIR, "..", "data", "app.db")
        if os.path.exists(db_path):
            size_kb = os.path.getsize(db_path) / 1024
            print(f"  文件路径: {os.path.abspath(db_path)}")
            print(f"  文件大小: {size_kb:.1f} KB")
        print(f"  政策总数: {len(policies)}")

        if not policies:
            print("  (无数据)")
            return

        # 按发布机关统计
        authority_counts = Counter(p.issuing_authority for p in policies)
        print(f"\n  按发布机关统计:")
        for auth, cnt in authority_counts.most_common():
            print(f"    - {auth}: {cnt} 条")

        # 按政策工具统计
        tool_counts = Counter(p.policy_tool for p in policies if p.policy_tool)
        print(f"\n  按政策工具统计:")
        for tool, cnt in tool_counts.most_common():
            print(f"    - {tool}: {cnt} 条")

        # 按地区统计
        loc_counts = Counter(p.location for p in policies if p.location)
        print(f"\n  按适用地区统计:")
        for loc, cnt in loc_counts.most_common():
            print(f"    - {loc}: {cnt} 条")

        # 按分类统计
        cat_counts = Counter(p.category for p in policies if p.category)
        print(f"\n  按政策分类统计:")
        for cat, cnt in cat_counts.most_common():
            print(f"    - {cat}: {cnt} 条")

        # 详细清单
        print(f"\n  政策详细清单:")
        for i, p in enumerate(policies, 1):
            body_preview = p.body_text[:60].replace("\n", " ")
            print(f"  [{i}] id={p.id} | {p.title}")
            print(f"      发布机关: {p.issuing_authority} | 日期: {p.publish_date}")
            print(f"      地点: {p.location or '未指定'} | 工具: {p.policy_tool or '未指定'}")
            print(f"      分类: {p.category or '未指定'} | 字数: {len(p.body_text)}")
            print(f"      正文预览: {body_preview}...")
            print()

    # ------------------ ChromaDB ------------------
    print("\n[向量库] ChromaDB")
    print("-" * 40)
    chroma_path = os.path.join(_APP_DIR, "..", "data", "chroma_db")
    if os.path.exists(chroma_path):
        # 计算目录大小
        total_size = 0
        for root, dirs, files in os.walk(chroma_path):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
        print(f"  存储路径: {os.path.abspath(chroma_path)}")
        print(f"  存储大小: {total_size / 1024:.1f} KB")

    try:
        collection_data = policy_vector_store.get()
        chunk_ids = collection_data.get("ids", [])
        metadatas = collection_data.get("metadatas", [])
        print(f"  集合名称: policies")
        print(f"  文档片段总数: {len(chunk_ids)}")

        if metadatas:
            # 每条政策包含的 chunk 数
            policy_chunk_map = {}
            for meta in metadatas:
                pid = meta.get("policy_id")
                if pid not in policy_chunk_map:
                    policy_chunk_map[pid] = 0
                policy_chunk_map[pid] += 1

            print(f"  覆盖政策数: {len(policy_chunk_map)}")
            print(f"\n  每条政策的 chunk 分布:")
            for pid, cnt in sorted(policy_chunk_map.items()):
                # 找到对应的政策标题
                title = "?"
                for meta in metadatas:
                    if meta.get("policy_id") == pid:
                        title = meta.get("title", "?")
                        break
                print(f"    policy_id={pid}: {cnt} chunks — {title}")

            # 元数据覆盖率
            has_location = sum(1 for m in metadatas if m.get("location"))
            has_tool = sum(1 for m in metadatas if m.get("policy_tool"))
            has_date = sum(1 for m in metadatas if m.get("publish_date"))
            has_authority = sum(1 for m in metadatas if m.get("issuing_authority"))
            total = len(metadatas)
            print(f"\n  元数据覆盖率:")
            print(f"    发布机关: {has_authority}/{total} ({100*has_authority/total:.1f}%)")
            print(f"    发布日期: {has_date}/{total} ({100*has_date/total:.1f}%)")
            print(f"    适用地区: {has_location}/{total} ({100*has_location/total:.1f}%)")
            print(f"    政策工具: {has_tool}/{total} ({100*has_tool/total:.1f}%)")

    except Exception as e:
        print(f"  [ERROR] 读取 ChromaDB 失败: {e}")

    print("\n" + "=" * 60)
    print("  报告完毕")

if __name__ == "__main__":
    asyncio.run(report())
