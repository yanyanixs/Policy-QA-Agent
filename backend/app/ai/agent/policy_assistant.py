from datetime import datetime
from typing import cast, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ai.llm import get_model, settings
from ai.tools.policy_tools import search_policies, get_policy_detail, list_policies_by_metadata
from ai.trace_utils import TimedStep, get_trace, end_trace


class AgentState(MessagesState):
    """Policy Q&A Agent 状态"""


tools = [search_policies, get_policy_detail, list_policies_by_metadata]


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """将工具绑定到模型，并注入系统提示词"""
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


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


async def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """调用 LLM 模型"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)

    label = "LLM 最终回答" if _has_tool_call_history(state) else "LLM 初次决策"
    with TimedStep("llm_call", label):
        response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}


def _has_tool_call_history(state: AgentState) -> bool:
    """检查消息历史中是否已有工具调用（即 LLM 是否已至少调过一次工具）"""
    from langchain_core.messages import ToolMessage
    return any(isinstance(m, ToolMessage) for m in state.get("messages", []))


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    """判断是否需要调用工具"""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


# 构建 LangGraph
agent = StateGraph(AgentState)
agent.add_node("model", call_model)
agent.add_node("tools", ToolNode(tools=tools))
agent.set_entry_point("model")
agent.add_edge("tools", "model")
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

policy_assistant = agent.compile(checkpointer=MemorySaver())
policy_assistant.name = "policy_assistant"
