from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from typing import Optional

from db.models.policy import Policy


class PolicyRepository:

    @classmethod
    async def create_policy(cls, session: AsyncSession, policy: Policy) -> Policy:
        """创建政策记录"""
        session.add(policy)
        await session.commit()
        await session.refresh(policy)
        return policy

    @classmethod
    async def get_policy(cls, session: AsyncSession, policy_id: int) -> Optional[Policy]:
        """根据 ID 获取政策"""
        result = await session.execute(
            select(Policy).where(Policy.id == policy_id)
        )
        return result.scalars().first()

    @classmethod
    async def get_policy_by_title(cls, session: AsyncSession, title: str) -> Optional[Policy]:
        """根据标题精确获取政策"""
        result = await session.execute(
            select(Policy).where(Policy.title == title)
        )
        return result.scalars().first()

    @classmethod
    async def get_all_policies(cls, session: AsyncSession) -> list[Policy]:
        """获取所有政策"""
        result = await session.execute(select(Policy))
        return list(result.scalars().all())

    @classmethod
    async def get_policies_by_authority(
        cls, session: AsyncSession, authority: str
    ) -> list[Policy]:
        """按发布机关查询"""
        result = await session.execute(
            select(Policy).where(Policy.issuing_authority == authority)
        )
        return list(result.scalars().all())

    @classmethod
    async def get_policies_by_location(
        cls, session: AsyncSession, location: str
    ) -> list[Policy]:
        """按地区查询"""
        result = await session.execute(
            select(Policy).where(Policy.location == location)
        )
        return list(result.scalars().all())

    @classmethod
    async def search_policies(
        cls,
        session: AsyncSession,
        *,
        title_like: str | None = None,
        issuing_authority: str | None = None,
        location: str | None = None,
        policy_tool: str | None = None,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str = "active",
        limit: int = 50,
    ) -> list[Policy]:
        """多字段组合查询，支持精确匹配和日期范围过滤"""
        stmt = select(Policy)

        if status:
            stmt = stmt.where(Policy.status == status)
        if title_like:
            stmt = stmt.where(Policy.title.contains(title_like))
        if issuing_authority:
            stmt = stmt.where(Policy.issuing_authority == issuing_authority)
        if location:
            stmt = stmt.where(Policy.location == location)
        if policy_tool:
            stmt = stmt.where(Policy.policy_tool == policy_tool)
        if category:
            stmt = stmt.where(Policy.category == category)
        if date_from:
            stmt = stmt.where(Policy.publish_date >= date_from)
        if date_to:
            stmt = stmt.where(Policy.publish_date <= date_to)

        stmt = stmt.order_by(Policy.publish_date.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @classmethod
    async def update_policy(
        cls, session: AsyncSession, policy_id: int, policy_data: dict
    ) -> Optional[Policy]:
        """更新政策"""
        policy = await cls.get_policy(session, policy_id)
        if policy:
            for key, value in policy_data.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            await session.commit()
            await session.refresh(policy)
        return policy

    @classmethod
    async def delete_policy(cls, session: AsyncSession, policy_id: int) -> bool:
        """删除政策"""
        policy = await cls.get_policy(session, policy_id)
        if policy:
            await session.delete(policy)
            await session.commit()
            return True
        return False
