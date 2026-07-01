from db.models.base import DBBaseModel
from sqlmodel import Field


class Policy(DBBaseModel, table=True):
    """
    政策数据模型
    """
    __tablename__ = "policy"

    id: int | None = Field(default=None, primary_key=True, index=True, description="主键")
    title: str = Field(max_length=500, index=True, description="政策标题")
    issuing_authority: str = Field(max_length=200, index=True, description="发布机关")
    doc_number: str | None = Field(default=None, max_length=100, description="发文字号")
    publish_date: str | None = Field(default=None, max_length=20, index=True, description="发布日期 YYYY-MM-DD")
    effective_date: str | None = Field(default=None, max_length=20, description="生效日期 YYYY-MM-DD")
    location: str | None = Field(default=None, max_length=200, index=True, description="适用地区")
    policy_tool: str | None = Field(default=None, max_length=100, index=True, description="政策工具类型（如财政补贴、税收优惠、行政审批）")
    category: str | None = Field(default=None, max_length=100, description="政策分类")
    status: str = Field(default="active", max_length=20, description="状态: active/expired/repealed")
    source_url: str | None = Field(default=None, max_length=1000, description="原文链接")
    body_text: str = Field(default="", description="正文全文")
