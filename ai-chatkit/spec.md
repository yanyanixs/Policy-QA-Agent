# AI-CHATKIT RAG 全流程技术规格书

> **版本**: 1.2  
> **最后更新**: 2026-07-01  
> **项目路径**: `ai-chatkit/`

---

## 目录

1. [系统概览](#1-系统概览)
2. [技术栈](#2-技术栈)
3. [项目结构](#3-项目结构)
4. [文档摄入管道（离线）](#4-文档摄入管道离线)
5. [双层存储架构](#5-双层存储架构)
6. [检索管道（在线）](#6-检索管道在线)
7. [生成管道（在线）](#7-生成管道在线)
8. [完整查询流程追踪](#8-完整查询流程追踪)
9. [前端流式消费](#9-前端流式消费)
10. [关键参数速查表](#10-关键参数速查表)
11. [API 接口规范](#11-api-接口规范)
12. [数据模型](#12-数据模型)
13. [设计决策与约束](#13-设计决策与约束)
14. [文件索引](#14-文件索引)

---

## 1. 系统概览

AI-CHATKIT 是一个**政策问答 RAG（检索增强生成）智能助手**，核心技术流程分为两条管道：

```
┌──────────────────────────────────────────────────────────────────┐
│                        离线管道（Ingestion）                       │
│  JSON 政策文件 → 文本分块 → BGE-M3 向量化 → ChromaDB + SQLite     │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                        在线管道（Query）                           │
│  用户提问 → LLM 决策工具调用 → 向量检索 → Rerank 精排 → MMR 多样性 → 格式化 → 生成回答  │
└──────────────────────────────────────────────────────────────────┘
```

系统架构图：

```
┌──────────────┐     SSE Stream      ┌───────────────┐
│   Next.js    │ ◄────────────────── │   FastAPI      │
│   Frontend   │                     │   Backend      │
└──────────────┘                     └───────┬───────┘
                                              │
                          ┌───────────────────┼───────────────────────┐
                          │                   │                       │
                    ┌─────▼─────┐    ┌───────▼───────┐   ┌───────▼───────────┐
                    │ LangGraph  │    │   ChromaDB     │   │    SQLite         │
                    │   Agent    │    │  (向量检索)     │   │  (结构化存储)      │
                    └─────┬─────┘    └───────┬───────┘   └───────┬───────────┘
                          │                  │                    │
                          │         ┌────────▼────────┐          │
                          │    ┌────┤  OllamaEmbeddings│          │
                          │    │    │    (BGE-M3)      │          │
                          │    │    └──────────────────┘          │
                          │    │              │                    │
                          │    │    ┌─────────▼─────────┐         │
                          │    │    │   Ollama (本地)    │         │
                          │    │    │   bge-m3 模型     │         │
                          │    │    └───────────────────┘         │
                          │    │                                  │
                          │    │    ┌───────────────────────┐     │
                          │    └───►│  BGE-Reranker-v2-m3    │     │
                          │         │  (Cross-Encoder 精排)  │     │
                          │         │  + MMR 多样性重排      │     │
                          │         └───────────────────────┘     │
                          │                                       │
                          ┌────────────────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │  DeepSeek  │
                    │ v4-flash   │
                    │  (LLM)     │
                    └───────────┘
```

---

## 2. 技术栈

| 层级 | 技术 | 版本/型号 |
|------|------|-----------|
| **前端框架** | Next.js (React + TypeScript) | — |
| **UI 库** | Ant Design + Tailwind CSS | — |
| **后端框架** | FastAPI (Python) | — |
| **Agent 编排** | LangGraph | — |
| **LLM** | DeepSeek V4 Flash | deepseek-v4-flash |
| **LLM 温度** | 0.5 | — |
| **Embedding 模型** | BGE-M3（通过 Ollama 本地部署） | bge-m3 |
| **Embedding 维度** | 1024 | — |
| **向量数据库** | ChromaDB（持久化模式） | — |
| **关系数据库** | SQLite（异步驱动 aiosqlite） | — |
| **ORM** | SQLModel（兼容 SQLAlchemy 异步） | — |
| **文本分割器** | RecursiveCharacterTextSplitter (LangChain) | — |
| **流式协议** | Server-Sent Events (SSE) | — |

---

## 3. 项目结构

```
ai-chatkit/
├── spec.md                          # 本文件
├── README_zh.md                     # 项目说明（中文）
│
├── backend/
│   ├── .env                         # 运行时配置
│   ├── pyproject.toml               # Python 依赖
│   ├── resource/
│   │   ├── database.db              # SQLite 数据库文件
│   │   ├── chroma_db/               # ChromaDB 持久化目录
│   │   ├── sample_policies.json     # 示例政策数据（3 条）
│   │   └── 1_policies.json         # 主政策数据集
│   │
│   ├── scripts/
│   │   ├── import_policies.py       # ★ 文档摄入主脚本
│   │   └── storage_report.py       # 存储状态报告
│   │
│   ├── tests/rag/
│   │   ├── importRag.py            # PDF 手册摄入测试
│   │   └── queryChroma.py          # ChromaDB 查询测试
│   │
│   └── app/
│       ├── main.py                  # FastAPI 入口
│       ├── run_server.py            # Uvicorn 启动器
│       │
│       ├── core/
│       │   └── config.py            # Settings 配置类
│       │
│       ├── db/
│       │   ├── database.py          # 异步 SQLAlchemy 引擎
│       │   └── models/
│       │       ├── base.py          # DBBaseModel（create_time/edit_time）
│       │       └── policy.py        # ★ Policy ORM 模型
│       │   └── repository/
│       │       └── policy_repo.py   # ★ Policy CRUD 仓库
│       │
│       ├── ai/
│       │   ├── models.py            # 模型名称枚举
│       │   ├── llm.py               # ★ LLM 工厂函数
│       │   ├── rag/
│       │   │   ├── chromaClient.py  # ★ ChromaDB 客户端 + 向量存储
│       │   │   └── reranker.py       # ★ Reranker 精排 + MMR 多样性
│       │   ├── tools/
│       │   │   └── policy_tools.py  # ★ 三个 RAG 工具定义
│       │   └── agent/
│       │       ├── agents.py        # Agent 注册表
│       │       ├── policy_assistant.py # ★ 政策问答 Agent
│       │       └── multi_agent.py   # 多智能体协作（非 RAG）
│       │
│       ├── api/
│       │   ├── chat_routes.py       # ★ /chat/invoke + /chat/stream
│       │   └── schema/
│       │       └── chatSchema.py    # 请求/响应 Pydantic 模型
│       │
│       └── utils/
│           └── chat_utils.py        # 消息格式转换工具
│
└── frontend/
    └── app/
        ├── page.tsx                 # 首页
        ├── layout-context.ts        # Agent/Thread 上下文
        ├── chat/
        │   ├── page.tsx             # 聊天页
        │   ├── [threadId]/page.tsx  # 指定会话的聊天页
        │   ├── types/chat.types.ts  # 前端类型定义
        │   ├── hooks/
        │   │   ├── useStreamChat.ts # ★ SSE 流式消费 Hook
        │   │   ├── useChatActions.ts
        │   │   └── useScrollToBottom.ts
        │   └── components/
        │       ├── ChatComponent.tsx # ★ 主聊天组件
        │       ├── MessageBubble.tsx # 消息气泡渲染
        │       └── MessageInput.tsx  # 输入框
        └── components/
            ├── SiderComponent.tsx    # 侧边栏
            ├── SessionListItem.tsx
            └── NewChatButton.tsx
```

---

## 4. 文档摄入管道（离线）

### 4.1 入口

**文件**: [`backend/scripts/import_policies.py`](backend/scripts/import_policies.py)

**用法**:
```bash
cd backend
python scripts/import_policies.py resource/sample_policies.json [--clean]
```

- `--clean`：先清空 ChromaDB 和 SQLite 中已有数据，再导入

### 4.2 流程图

```
JSON 文件 (policy 数组)
        │
        ▼
┌────────────────────────────────┐
│ Step 1: JSON 解析               │
│ load_policies_from_json()       │
│ • 中文字段名 → 内部 key 映射     │
│   标题→title, 发布机关→          │
│   issuing_authority,            │
│   正文→body_text, etc.          │
└───────────────┬────────────────┘
                │
        ┌───────┴───────┐
        │  for each     │
        │  policy       │
        └───────┬───────┘
                │
┌───────────────▼────────────────┐
│ Step 2: 写入 SQLite            │
│ PolicyRepository.create_policy │
│ • 完整元数据 + 正文全文          │
│ • 返回自增 policy_id           │
└───────────────┬────────────────┘
                │
┌───────────────▼────────────────┐
│ Step 3: 文本分块                │
│ RecursiveCharacterTextSplitter │
│ • chunk_size: 500 字符          │
│ • chunk_overlap: 50 字符        │
│ • 分隔符优先级:                  │
│   \n\n → \n → 。→ . → ；→ ;    │
│   → ，→ , → 空格 → 无          │
└───────────────┬────────────────┘
                │
┌───────────────▼────────────────┐
│ Step 4: 构建 Document           │
│ • page_content = chunk_text    │
│ • metadata = {                 │
│     policy_id, title,          │
│     issuing_authority,         │
│     publish_date, location,    │
│     policy_tool, category,     │
│     doc_number, source_file,   │
│     chunk_index, total_chunks  │
│   }                            │
└───────────────┬────────────────┘
                │
┌───────────────▼────────────────┐
│ Step 5: 向量化 + 存储           │
│ policy_vector_store            │
│   .add_documents(documents)    │
│                                │
│ → OllamaEmbeddings(bge-m3)     │
│   生成 1024 维向量              │
│ → ChromaDB PersistentClient    │
│   存入 "policies" 集合          │
│ → 磁盘: resource/chroma_db/    │
└────────────────────────────────┘
```

### 4.3 字段映射表

| JSON 中文字段 | 内部 key | 说明 |
|---|---|---|
| `标题` | `title` | 政策标题 |
| `发布机关` | `issuing_authority` | 如"财政部"、"国务院" |
| `发文字号` | `doc_number` | 如"国能发科技〔2026〕12号" |
| `日期` | `publish_date` | YYYY-MM-DD |
| `生效日期` | `effective_date` | YYYY-MM-DD |
| `地点` | `location` | 如"全国"、"上海市" |
| `政策工具` | `policy_tool` | 如"财政补贴"、"税收优惠" |
| `分类` | `category` | 如"能源"、"财税"、"科技" |
| `状态` | `status` | active/expired/repealed |
| `原文链接` | `source_url` | URL |
| `正文` | `body_text` | 政策全文 |

### 4.4 分块参数详解

| 参数 | 值 | 说明 |
|------|------|------|
| `chunk_size` | 500 | 每个文本块最大 500 字符 |
| `chunk_overlap` | 50 | 相邻块重叠 50 字符，防止信息在边界断裂 |
| `separators` | `["\n\n", "\n", "。", ".", "；", ";", "，", ",", " ", ""]` | 按优先级依次尝试分割：优先在段落边界切分，其次句子，最后字符级 |
| `length_function` | `len` | 按字符数计算长度 |
| `is_separator_regex` | `False` | 分隔符为普通字符串匹配 |

### 4.5 备选摄入路径（PDF 手册）

**文件**: [`backend/tests/rag/importRag.py`](backend/tests/rag/importRag.py)

```
PDF 文件 → PyPDFLoader → RecursiveCharacterTextSplitter(chunk=100, overlap=20)
         → hand_book_vector_store.add_documents()
         → ChromaDB "handbook" 集合
```

此路径使用更小的分块参数（100 字符 / 20 重叠），存储到独立集合 `"handbook"`，当前未被 Agent 工具引用。

---

## 5. 双层存储架构

### 5.1 设计原则

```
┌──────────────────────────────────────────────────────┐
│                   SQLite (source of truth)            │
│  完整结构化数据：元数据 + 全文正文                      │
│  • 精确查询 (SQL WHERE)                               │
│  • 完整正文获取 (get_policy_detail)                    │
│  • 元数据浏览 (list_policies_by_metadata)              │
├──────────────────────────────────────────────────────┤
│                ChromaDB (search index)                 │
│  分块向量 + 元数据副本                                  │
│  • 语义相似度搜索 (search_policies)                    │
│  • 元数据过滤 (where filter)                           │
│  • 向量余弦相似度排序                                   │
└──────────────────────────────────────────────────────┘
```

关键点：
- **SQLite 是主存储**，保存政策完整信息
- **ChromaDB 是检索索引**，仅保存分块后的向量和关键元数据副本
- 两者通过 `policy_id` 桥接

### 5.2 SQLite 数据库

**文件**: `resource/database.db`

**表**: `policy`（由 SQLModel ORM 管理）

| 列名 | 类型 | 索引 | 说明 |
|------|------|------|------|
| `id` | INTEGER | PK | 自增主键 |
| `title` | VARCHAR(500) | ✓ | 政策标题 |
| `issuing_authority` | VARCHAR(200) | ✓ | 发布机关 |
| `doc_number` | VARCHAR(100) | | 发文字号 |
| `publish_date` | VARCHAR(20) | ✓ | 发布日期 |
| `effective_date` | VARCHAR(20) | | 生效日期 |
| `location` | VARCHAR(200) | ✓ | 适用地区 |
| `policy_tool` | VARCHAR(100) | ✓ | 政策工具类型 |
| `category` | VARCHAR(100) | | 政策分类 |
| `status` | VARCHAR(20) | | active/expired/repealed |
| `source_url` | VARCHAR(1000) | | 原文链接 |
| `body_text` | TEXT | | 正文全文 |
| `create_time` | DATETIME | | 创建时间 |
| `edit_time` | DATETIME | | 更新时间 |

**引擎**: `create_async_engine("sqlite+aiosqlite:///resource/database.db")`

### 5.3 ChromaDB 向量数据库

**文件**: [`backend/app/ai/rag/chromaClient.py`](backend/app/ai/rag/chromaClient.py)

**配置**:
```python
# 客户端
client = chromadb.PersistentClient(
    path="resource/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Embedding 函数
embeddings = OllamaEmbeddings(model="bge-m3")
```

**两个集合**:

| 集合名 | 用途 | 是否被 Agent 使用 |
|--------|------|:---:|
| `policies` | 政策分块向量存储 | ✓ |
| `handbook` | PDF 员工手册向量存储 | ✗ |

**集合 `policies` 中的每条向量（一个 chunk）的元数据**:

| 元数据字段 | 类型 | 说明 |
|---|---|---|
| `policy_id` | int | 关联 SQLite 的 policy.id |
| `title` | str | 政策标题 |
| `issuing_authority` | str | 发布机关 |
| `publish_date` | str | 发布日期 |
| `location` | str | 适用地区 |
| `policy_tool` | str | 政策工具类型 |
| `category` | str | 分类 |
| `doc_number` | str | 发文字号 |
| `source_file` | str | 来源 JSON 文件名 |
| `chunk_index` | int | 块序号 |
| `total_chunks` | int | 该政策总分块数 |

**持久化存储**: `resource/chroma_db/` 目录下包含：
- 以 UUID 命名的子目录（每个集合一个），含 `data_level0.bin`、`header.bin`、`length.bin`、`link_lists.bin`
- `chroma.sqlite3`：元数据数据库

**相似度度量**: ChromaDB 默认余弦相似度（cosine similarity）

---

## 6. 检索管道（在线）

### 6.1 工具概览

定义于 [`backend/app/ai/tools/policy_tools.py`](backend/app/ai/tools/policy_tools.py)，共三个 `@tool`：

```
┌─────────────────────────────────────────────────────────────┐
│ Tool 1: search_policies        │ 语义向量搜索 + 元数据过滤     │
│ 数据源: ChromaDB "policies"    │ 返回: 合并同政策命中 chunk 后  │
│                                │ 按 chunk_index 排序，上限      │
│                                │ 3000 字符/篇，取前 top_k 条    │
├─────────────────────────────────────────────────────────────┤
│ Tool 2: get_policy_detail      │ 获取单条政策的完整正文        │
│ 数据源: SQLite policy 表       │ 返回: 完整元数据 + 正文全文   │
├─────────────────────────────────────────────────────────────┤
│ Tool 3: list_policies_by_metadata│ 按元数据字段精确浏览       │
│ 数据源: SQLite policy 表       │ 返回: 匹配的元数据列表        │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Tool 1: `search_policies` — 语义搜索

```
调用参数
├── query: str                        # 必填，搜索查询文本
├── issuing_authority: str | None     # 按发布机关过滤
├── location: str | None              # 按地区过滤
├── policy_tool: str | None           # 按政策工具过滤
├── category: str | None              # 按分类过滤
├── start_date: str | None            # 起始日期 YYYY-MM-DD
├── end_date: str | None              # 截止日期 YYYY-MM-DD
└── top_k: int = 8                    # 返回结果数量

执行流程:
┌──────────────────────────────────────────────────┐
│ 1. _build_chroma_filter() 构建 Chroma 过滤条件    │
│    • 元数据字段 → {"$eq": value}                  │
│    • 日期范围 → {"publish_date": {"$gte":..., "$lte":...}}│
│    • 多条件 → {"$and": [...]}                     │
│    • 无条件 → None                                │
├──────────────────────────────────────────────────┤
│ 2. policy_vector_store.similarity_search()        │
│    • 查询文本 → OllamaEmbeddings(bge-m3) → 向量   │
│    • 余弦相似度搜索                                │
│    • k = min(top_k * 3, 30)  # 多取 3x 用于精排  │
│    • filter = chroma_filter                       │
├──────────────────────────────────────────────────┤
│ 3. BGE-Reranker-v2-m3 Cross-Encoder 精排          │
│    • 对 30 个粗筛 chunk 逐对打分 (query, chunk)    │
│    • 文件: app/ai/rag/reranker.py                  │
├──────────────────────────────────────────────────┤
│ 4. MMR 多样性重排 (λ=0.7)                         │
│    • 同 policy_id chunk → 相似度=1, 扣 0.3 分     │
│    • 不同 policy_id → 相似度=0, 不扣分             │
│    • 贪心迭代选出 top_n=top_k×3 个多样 chunk       │
├──────────────────────────────────────────────────┤
│ 5. _format_chunk_results() 格式化                 │
│    • 按 policy_id 分组，合并同政策的所有命中 chunk   │
│    • 每组内按 chunk_index 排序，还原原文顺序        │
│    • 单篇合并上限 _MAX_CONTENT_PER_POLICY = 3000   │
│    • 截取前 top_k 篇                               │
│    • 每条显示: 标题、发布机关、日期、地区、         │
│      政策工具、发文字号、ID、命中片段数、合并正文    │
└──────────────────────────────────────────────────┘
```

**过滤条件构建逻辑** (`_build_chroma_filter`):

```python
# 单字段精确匹配
{"issuing_authority": {"$eq": "财政部"}}

# 日期范围
{"publish_date": {"$gte": "2024-01-01", "$lte": "2024-12-31"}}

# 多字段组合
{"$and": [
    {"issuing_authority": {"$eq": "财政部"}},
    {"location": {"$eq": "全国"}},
    {"publish_date": {"$gte": "2024-01-01"}}
]}
```

**去重与合并逻辑** (`_format_chunk_results`):

1. 遍历 Rerank+MMR 精排后的所有 chunk（最多 24 个），按 `policy_id` 分组
2. 每组内按 `chunk_index` 升序排序，按序拼接所有命中 chunk 的正文
3. 单篇政策总字符数超过 `_MAX_CONTENT_PER_POLICY`（3000）时截断，并标注「已截断，全文共 N 字符」
4. 保持精排后的 MMR 顺序截取前 `top_k`（默认 8）篇
5. 每篇展示：命中片段数 + 合并后总字符数 + 合并正文

**重要**: 已实施 BGE-Reranker-v2-m3 Cross-Encoder 精排 + MMR 多样性重排。粗筛结果的余弦相似度分数被 Cross-Encoder 精排分数替代，再经 MMR（λ=0.7）多样性重排后送入格式化。可通过 `use_mmr=False` 禁用 MMR。

**自动降级**: Reranker 支持三级自动降级（`reranker.py:_try_rerank`）：
1. `cross-encoder`（默认）→ 失败时自动降级到 `llm`
2. `llm` → 失败时自动降级到 `none`（保留原始向量相似度顺序）
3. 降级后会记录警告日志，本会话不再重试失败的后端。

可通过 `.env` 中的 `RERANKER_BACKEND` 显式选择后端，避免自动降级。

### 6.3 Tool 2: `get_policy_detail` — 全文获取

```
调用参数
└── policy_id: int     # 政策 ID

执行流程:
┌──────────────────────────────────────┐
│ 1. PolicyRepository.get_policy(session, policy_id) │
│    SELECT * FROM policy WHERE id = ?│
├──────────────────────────────────────┤
│ 2. 格式化输出:                       │
│    • 元数据块（标题、机关、日期、      │
│      文号、地区、工具、分类、状态、链接）│
│    • 正文全文                         │
└──────────────────────────────────────┘
```

### 6.4 Tool 3: `list_policies_by_metadata` — 结构化浏览

```
调用参数
├── issuing_authority: str | None
├── location: str | None
├── policy_tool: str | None
├── category: str | None
├── start_date: str | None
├── end_date: str | None
├── title_keyword: str | None     # 标题模糊匹配 (LIKE %keyword%)
└── limit: int = 20

执行流程:
┌──────────────────────────────────────┐
│ PolicyRepository.search_policies()    │
│ • 组合多个 WHERE 条件                 │
│ • title_keyword → .contains() (LIKE) │
│ • 其他字段 → 精确匹配 (==)             │
│ • 日期 → >= / <=                     │
│ • 默认只查 status="active"           │
│ • ORDER BY publish_date DESC         │
│ • LIMIT limit                        │
├──────────────────────────────────────┤
│ 格式化: 编号列表，仅元数据无正文       │
└──────────────────────────────────────┘
```

---

## 7. 生成管道（在线）

### 7.1 Agent 定义

**文件**: [`backend/app/ai/agent/policy_assistant.py`](backend/app/ai/agent/policy_assistant.py)

**架构**: LangGraph 状态图

```
                 ┌──────────┐
          START──►  model   │◄──────────────┐
                 └────┬─────┘               │
                      │                     │
              pending_tool_calls()          │
                      │                     │
              ┌───────┴───────┐             │
              │               │             │
           tools           done ──► END     │
              │                             │
              └─────────────────────────────┘
```

**节点定义**:
- `model`：`call_model` 函数 —— 注入系统提示词 + 绑定工具，调用 LLM
- `tools`：LangGraph `ToolNode` —— 执行工具调用
- 路由：`pending_tool_calls` —— 如果 AIMessage 包含 `tool_calls` → `"tools"`，否则 → `"done"`

**状态**: `AgentState` 继承自 `MessagesState`，即 `{"messages": [...]}`

**记忆**: `MemorySaver()` —— 内存级检查点，支持多轮对话

### 7.2 系统提示词

```text
你是一个政策问答智能助手，帮助用户搜索和理解政府发布的各类政策法规。

你可以调用以下工具来获取信息：
1. **search_policies**：语义搜索政策知识库。适合开放性问答（如"新能源补贴有哪些政策"）。
2. **get_policy_detail**：获取某条政策的完整正文。适合用户想看全文或确认细节时使用。
3. **list_policies_by_metadata**：按结构化字段精确浏览政策列表。

### 工作准则：
- **必须标注来源**：每次回答都必须注明政策名称、发布机关和发布日期。
- **禁止编造**：不得凭空捏造政策条款。如知识库中无相关答案，请如实告知用户。
- **善用过滤**：用户提及具体机关、地区、政策类型时，用对应参数精确过滤。
- **处理模糊问题**：用户问题不够明确时，可追问地区、机关、政策类型等来缩小范围。
- **结构化回答**：涉及多条政策时按相关度排列，每条列出标题、机关、日期和核心内容摘要。
- **日期处理**：用户说"今年"指 {当前年} 年，"去年"指 {当前年-1} 年，"最近"指近半年内。

当前时间：{YYYY-MM-DD HH:MM:SS}
```

**动态注入**:
- `datetime.now().year`、`datetime.now().year - 1` → 处理相对时间
- `datetime.now().strftime('%Y-%m-%d %H:%M:%S')` → 提供当前时间上下文

**注入方式**:
```python
def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model
```
每次调用 LLM 之前，`SystemMessage` 被前置到消息列表头部。

### 7.3 LLM 配置

**文件**: [`backend/app/ai/llm.py`](backend/app/ai/llm.py)

```python
# 默认配置
ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0.5,
    streaming=True,
    api_key=settings.DEEPSEEK_API_KEY,
)
```

**可用模型** (`_MODEL_TABLE`):

| 枚举值 | API 模型名 | 提供商 |
|--------|-----------|--------|
| `OpenAIModelName.GPT_4O_MINI` | `gpt-4o-mini` | OpenAI |
| `OpenAIModelName.GPT_4O` | `gpt-4o` | OpenAI |
| `DeepseekModelName.DEEPSEEK_CHAT` | `deepseek-chat` | DeepSeek |
| `DeepseekModelName.DEEPSEEK_V4_FLASH` | `deepseek-v4-flash` | DeepSeek |
| `OllamaModelName.OLLAMA_GENERIC` | `ollama` (任意模型) | Ollama 本地 |
| `FakeModelName.FAKE` | `fake` | 测试桩 |
| `TongYiModelName.QWEN_PLUS` | `qwen-plus` | 阿里 TongYi |

配置通过 `.env` 的 `DEFAULT_MODEL` 控制，运行时也可通过 `RunnableConfig.configurable.model` 动态指定。

### 7.4 知识注入方式

系统采用 **Tool-Calling 模式** 而非静态 RAG 模板注入：

```
静态 RAG 模板（非本项目）:
  Prompt = "已知以下文档: {retrieved_docs}\n\n请回答: {user_query}"
  → 一次性注入，无交互

Tool-Calling 模式（本项目）:
  1. LLM 接收用户问题 + 工具列表
  2. LLM 自主决策调用哪个工具、传什么参数
  3. 工具返回格式化文本（检索结果）
  4. LLM 基于检索结果生成最终回答
  → 多步推理，可多次调用工具
```

**优势**: LLM 可以先 search 找到候选，再 get_detail 深入查看某条政策，然后综合回答。

---

## 8. 完整查询流程追踪

以用户提问「新能源补贴有哪些政策？」为例：

```
──────────────────────────────────────────────────────────────────
时间线                      组件                      数据
──────────────────────────────────────────────────────────────────
T0  用户输入             ChatComponent              "新能源补贴有哪些政策？"
                         handleSend()
                              │
T1  POST /chat/stream    chat_routes.py:stream()    {message, thread_id,
                              │                      agent_id, stream_tokens}
T2  创建 RunnableConfig  _handle_input()            configurable={thread_id, model}
                              │
T3  启动 LangGraph       message_generator()
                         agent.astream(stream_mode=["updates","messages","custom"])
                              │
T4  进入 "model" 节点    call_model()               SystemMessage(instructions)
                              │                     + HumanMessage("新能源...")
T5  LLM 推理             ChatDeepSeek               → AIMessage: tool_calls=[
                              │                        {name:"search_policies",
T6  LLM 决定调用工具                                         args:{query:"新能源补贴政策"}}]
                              │
T7  路由 → "tools"       pending_tool_calls()
                              │
T8  执行工具             search_policies()          1. _build_chroma_filter(None)
                              │                     2. policy_vector_store
                              │                        .similarity_search(
                              │                          "新能源补贴政策",
                              │                          k=24,
                              │                          filter=None)
                              │                     3. Query → BGE-M3 Embedding
                              │                     4. ChromaDB 余弦相似度检索
                              │                     5. 返回 30 个 raw chunks
                              │                     6. BGE-Reranker Cross-Encoder 精排
                              │                     7. MMR 多样性重排 (λ=0.7)
                              │                     8. _format_chunk_results()
                              │                        → 去重 → 截取 top 8
                              │                     → ToolMessage(格式化文本)
T9  路由 → "model"       图结构 edge                       │
                              │                            │
T10 再次进入 "model"     call_model()               SystemMessage
                              │                     + HumanMessage
                              │                     + ToolMessage(检索结果)
T11 LLM 生成最终回答     ChatDeepSeek               → AIMessage("根据检索结果，
                              │                        以下是相关的政策...")
T12 路由 → "done"        pending_tool_calls()       无 tool_calls → END
                              │
T13 SSE 事件发送         message_generator()        ← 流式输出
                              │
    ┌─ stream_mode="messages":
    │   data: {"type":"token","content":"根"}       ← token 级流式
    │   data: {"type":"token","content":"据"}       
    │   ...
    │   data: {"type":"message","content":{...}}    ← 完整消息（含 tool_calls）
    │
    └─ stream_mode="updates":
        data: {"type":"message","content":{...}}    ← 节点更新（含工具结果）

T14 最终事件              finally 块                 data: {"type":"end"}
                              │
T15 前端渲染             useStreamChat.handleStream()
                         • "token" → 逐字追加文本
                         • "message" → 渲染完整消息/工具调用
                         • "end" → 停止流式，关闭 reader
──────────────────────────────────────────────────────────────────
```

### SSE 事件类型

| type | 来源 | 内容 | 前端处理 |
|------|------|------|----------|
| `token` | `stream_mode="messages"` | LLM 输出的单个 token 文本 | 逐字追加到 AI 消息 |
| `message` | `stream_mode="updates"` | 完整消息对象（含 tool_calls） | 替换/更新消息气泡 |
| `end` | `finally` 块 | 无 | 停止流式，重置状态 |
| `error` | `except` 块 | 错误信息 | 显示错误提示 |

---

## 9. 前端流式消费

### 9.1 核心 Hook

**文件**: [`frontend/app/chat/hooks/useStreamChat.ts`](frontend/app/chat/hooks/useStreamChat.ts)

```typescript
// 流式消费流程
fetch("/chat/stream", { body: JSON.stringify(requestMsg) })
  → response.body.getReader()
  → 逐行解析 "data: " 前缀
  → 根据 type 分发处理
```

**三种事件处理**:

| 事件 | 处理函数 | 逻辑 |
|------|---------|------|
| `token` | `handleTokenData` | `content + token` 逐字追加 |
| `message` (ai + tool_calls) | `handleMessageData` | 追加 tool_calls 到消息的 `toolCall.calls` 数组 |
| `message` (ai + content) | `handleMessageData` | 替换消息 content |
| `message` (tool) | `handleMessageData` | 匹配 `tool_call_id`，注入 `result` 到对应 tool_call |
| `end` | — | `setIsStreaming(false)`, `reader.cancel()` |

### 9.2 消息持久化

```typescript
// localStorage 按 threadId 持久化消息
localStorage.setItem("chatMessages-" + currentThreadId, JSON.stringify(messages))

// 页面加载时恢复
const stored = localStorage.getItem("chatMessages-" + currentThreadId)
if (stored) setMessages(JSON.parse(stored))
```

---

## 10. 关键参数速查表

| 类别 | 参数 | 值 | 所在文件 |
|------|------|------|----------|
| **分块** | chunk_size | 500 字符 | `scripts/import_policies.py:46` |
| | chunk_overlap | 50 字符 | `scripts/import_policies.py:47` |
| | 分隔符 | `\n\n, \n, 。, ., ；, ;, ，, ,, 空格, 空` | `scripts/import_policies.py:48` |
| **Embedding** | 模型 | BGE-M3 | `.env` → `config.py:34` |
| | 维度 | 1024 | BGE-M3 默认 |
| | 提供方 | Ollama (本地) | `chromaClient.py:12` |
| **ChromaDB** | 集合 (policies) | `policies` | `chromaClient.py:28` |
| | 集合 (handbook) | `handbook` | `chromaClient.py:19` |
| | 持久化路径 | `resource/chroma_db` | `.env` → `config.py:36` |
| | 客户端类型 | PersistentClient | `chromaClient.py:10` |
| | 相似度度量 | 余弦相似度 (默认) | ChromaDB 默认 |
| **检索** | 默认 top_k | 8 | `policy_tools.py:102` |
| | 内部检索数 | `min(top_k * 3, 30)` | `policy_tools.py:134` |
| | 单篇合并上限 | `_MAX_CONTENT_PER_POLICY` = 3000 字符 | `policy_tools.py:53` |
| | 去重策略 | 按 policy_id 分组，合并命中 chunks，按 chunk_index 排序 | `policy_tools.py:53-117` |
| | 过滤方式 | ChromaDB where ($eq / $gte / $lte / $and) | `policy_tools.py:17-50` |
| **Rerank+MMR** | 精排模型 | BGE-Reranker-v2-m3 (Cross-Encoder) | `reranker.py:177` |
| | MMR λ | 0.7 (70% 相关性, 30% 多样性) | `policy_tools.py:199` |
| | 精排候选数 | `min(top_k * 3, 30)` | `policy_tools.py:198` |
| | MMR 相似度粒度 | policy_id 级别 (同政策=1, 不同政策=0) | `reranker.py:144` |
| | 可选后端 | `cross-encoder` / `llm` / `none` (三级自动降级) | `reranker.py:57-60` |
| | 模型路径 | 可配置: HF id 或本地目录 | `.env` → `RERANKER_MODEL_PATH` |
| | HF 镜像 | `HF_ENDPOINT` (国内默认 `https://hf-mirror.com`) | `.env` / `config.py` |
| **LLM** | 默认模型 | deepseek-v4-flash | `.env` → `config.py:32` |
| | 温度 | 0.5 | `llm.py:74` |
| | 流式 | 开启 | `llm.py:74` |
| | API Key | `sk-5e6eb16ac1a7434284c5524a89986862` | `.env:23` |
| **Agent** | 检查点 | MemorySaver (内存) | `policy_assistant.py:78` |
| | 最大工具调用轮次 | 无限制 (由 LLM 决策) | LangGraph 默认 |
| **SQLite** | 数据库路径 | `resource/database.db` | `.env:5` |
| | 驱动 | aiosqlite | `.env:5` |

---

## 11. API 接口规范

### 11.1 POST /chat/invoke

非流式，一次返回完整结果。

**Request**:
```json
{
    "message": "新能源补贴有哪些政策？",
    "thread_id": "uuid-string (optional)",
    "agent_id": "policy-assistant (default)",
    "agent_config": { }
}
```

**Response**: `ChatMessage`
```json
{
    "type": "ai",
    "content": "根据检索结果...",
    "tool_calls": [],
    "run_id": "uuid"
}
```

### 11.2 POST /chat/stream

SSE 流式返回。

**Request**:
```json
{
    "message": "新能源补贴有哪些政策？",
    "thread_id": "uuid-string (optional)",
    "agent_id": "policy-assistant (default)",
    "agent_config": { },
    "stream_tokens": true
}
```

**Response**: `text/event-stream`

```
data: {"type": "message", "content": {"type": "ai", "tool_calls": [...], ...}}

data: {"type": "token", "content": "根"}

data: {"type": "token", "content": "据"}

...

data: {"type": "end"}
```

### 11.3 Agent 注册

**当前已注册 Agent**:

| agent_id | 描述 | Graph |
|----------|------|-------|
| `policy-assistant` | A Policy Q&A intelligent assistant. | `policy_assistant` |
| `multi-agent-supervisor` | A supervisor for multi-agent assistant. | `supervisor_agent` |

默认使用 `policy-assistant`。

---

## 12. 数据模型

### 12.1 Policy (SQLModel)

**文件**: [`backend/app/db/models/policy.py`](backend/app/db/models/policy.py)

```python
class Policy(DBBaseModel, table=True):
    __tablename__ = "policy"
    id: int | None              # PK, auto-increment
    title: str                  # VARCHAR(500), INDEXED
    issuing_authority: str      # VARCHAR(200), INDEXED
    doc_number: str | None      # VARCHAR(100)
    publish_date: str | None    # VARCHAR(20), INDEXED
    effective_date: str | None  # VARCHAR(20)
    location: str | None        # VARCHAR(200), INDEXED
    policy_tool: str | None     # VARCHAR(100), INDEXED
    category: str | None        # VARCHAR(100)
    status: str                 # VARCHAR(20), default "active"
    source_url: str | None      # VARCHAR(1000)
    body_text: str              # TEXT
    create_time: datetime | None
    edit_time: datetime | None
```

### 12.2 请求/响应模型

**文件**: [`backend/app/api/schema/chatSchema.py`](backend/app/api/schema/chatSchema.py)

| 模型 | 字段 | 类型 | 说明 |
|------|------|------|------|
| `UserInput` | message | str | 用户输入 |
| | thread_id | str\|None | 会话线程 ID |
| | agent_id | str\|None | 智能体 ID，默认 policy-assistant |
| | agent_config | dict | 额外配置 |
| `StreamInput` | (继承 UserInput) | | |
| | stream_tokens | bool | 是否流式 token，默认 true |
| `ChatMessage` | type | "human"\|"ai"\|"tool"\|"custom" | 消息角色 |
| | content | str | 消息内容 |
| | tool_calls | list[ToolCall] | 工具调用列表 |
| | tool_call_id | str\|None | 对应的工具调用 ID |
| | run_id | str\|None | 运行 ID |
| `ToolCall` | name | str | 工具名称 |
| | args | dict | 工具参数 |
| | id | str\|None | 调用 ID |

---

## 13. 设计决策与约束

### 13.1 架构层面

| # | 决策 | 理由/影响 |
|---|------|----------|
| 1 | **双层存储（SQLite + ChromaDB）** | SQLite 作为 Source of Truth 存储完整数据；ChromaDB 作为检索加速层仅存向量。两者通过 `policy_id` 关联。 |
| 2 | **Tool-Calling 而非静态 RAG 模板** | LLM 可以多步推理：先搜索、再查看详情、再综合回答。比一次性注入上下文更灵活。 |
| 3 | **Rerank + MMR 精排** | Cross-Encoder (BGE-Reranker-v2-m3) 对粗筛 30 个 chunk 逐对精排，再通过 MMR（λ=0.7）在 policy_id 粒度做多样性重排，避免单政策 chunk 占据高位。 |
| 4 | **3x 过度检索** | `k_raw = min(top_k * 3, 30)`，多取然后去重，补偿 chunk-level 粒度导致的政策重复。 |
| 5 | **分块级检索，政策级去重** | 检索返回的是 chunk，但通过 `policy_id` 去重，确保返回给 LLM 的是去重后的政策列表。 |

### 13.2 当前约束与限制

| # | 限制 | 详情 |
|---|------|------|
| 1 | **无文档上传 API** | 所有文档摄入通过 CLI 脚本离线完成。前端无文件上传 UI。 |
| 2 | **仅支持 JSON 摄入** | 主摄入脚本仅解析 JSON 格式。PDF 支持仅存在于测试代码且未被 Agent 引用。 |
| 3 | **无认证/鉴权** | API 完全开放，无用户认证、权限控制。 |
| 4 | **内存级对话记忆** | `MemorySaver()` 在进程重启后丢失所有对话历史。无持久化 checkpoint。 |
| 5 | **无分数阈值过滤** | Reranker 精排后仍未设置最低分阈值，低相关度 chunk 不会被自动过滤（因 top_k × 3 已经截断，实际影响小）。 |
| 10 | **Reranker 需首次下载模型** | BGE-Reranker-v2-m3 约 2GB，首次运行 `FlagEmbedding` 会自动下载至 HuggingFace 缓存。国内需设 `HF_ENDPOINT=https://hf-mirror.com`。若模型不可用，自动降级到 LLM 或 None 后端。 |
| 6 | **文本分块不计 Token** | 使用 `len()` 按字符数分块，与 LLM 的 Token 计量方式不一致。 |
| 7 | **Ollama 依赖** | Embedding 需要本地运行 Ollama 服务并已拉取 `bge-m3` 模型。 |
| 8 | **ChromaDB 过滤局限性** | 仅支持元数据精确匹配 (`$eq`) 和日期范围、不支持全文搜索过滤 chunk 内容。 |
| 9 | **无增量更新** | 导入脚本只支持全量覆盖 (`--clean`)，不支持增删单个文档。 |
| 10 | **handbook 集合未被使用** | `chromaClient.py` 中定义了 `handbook` 集合和 `importRag.py` 测试，但 Agent 工具未引用。 |
| 11 | **硬编码 API Key** | `.env` 中的 `DEEPSEEK_API_KEY` 直接硬编码在仓库中。 |
| 12 | **持久化仅靠 localStorage** | 前端用 localStorage 按 threadId 存储消息，无后端持久化。 |
| 13 | **无引用高亮** | 检索结果以纯文本拼接返回 LLM；前端无法识别哪段回答来自哪篇政策。 |

### 13.3 扩展方向

- ✅ Cross-Encoder Reranker + MMR 已实施（见 6.2）
- 支持 PDF / Word / Web 等多格式文档摄入
- 后端持久化对话历史（PostgreSQL / MySQL）
- 集成引用追踪和前端高亮
- 支持增量文档更新和版本管理
- 添加检索质量评估（MRR, NDCG）
- 增加 Hybrid Search（向量 + BM25）
- 添加用户认证和 API Key 管理

---

## 14. 文件索引

| 文件路径 | 功能 | RAG 关联度 |
|----------|------|:---:|
| `backend/.env` | 运行时配置 | ★★★ |
| `backend/app/core/config.py` | Settings 配置类 | ★★★ |
| `backend/app/ai/rag/chromaClient.py` | ChromaDB 客户端 + 向量存储初始化 | ★★★ |
| `backend/app/ai/rag/reranker.py` | Reranker 精排 (Cross-Encoder / LLM / None) + MMR 多样性 + 三级自动降级 | ★★★ |
| `backend/app/ai/tools/policy_tools.py` | 三个 RAG 工具（检索 + 全文 + 列表） | ★★★ |
| `backend/app/ai/agent/policy_assistant.py` | 政策问答 Agent (LangGraph + 系统提示词) | ★★★ |
| `backend/app/ai/llm.py` | LLM 工厂函数 | ★★★ |
| `backend/app/db/models/policy.py` | Policy SQLModel 定义 | ★★★ |
| `backend/app/db/repository/policy_repo.py` | Policy CRUD 操作 | ★★★ |
| `backend/app/db/database.py` | 异步数据库引擎 | ★★☆ |
| `backend/app/db/models/base.py` | 基础模型（时间戳） | ★☆☆ |
| `backend/app/api/chat_routes.py` | /chat/invoke, /chat/stream | ★★★ |
| `backend/app/api/schema/chatSchema.py` | 请求/响应模型 | ★★☆ |
| `backend/app/ai/agent/agents.py` | Agent 注册表 | ★★☆ |
| `backend/scripts/import_policies.py` | 文档摄入主脚本 | ★★★ |
| `backend/resource/sample_policies.json` | 示例政策数据 | ★★☆ |
| `backend/tests/rag/importRag.py` | PDF 手册摄入测试 | ★☆☆ |
| `backend/tests/rag/queryChroma.py` | ChromaDB 查询测试 | ★☆☆ |
| `frontend/app/chat/hooks/useStreamChat.ts` | SSE 流式消费 Hook | ★★☆ |
| `frontend/app/chat/components/ChatComponent.tsx` | 主聊天组件 | ★★☆ |
| `frontend/app/chat/types/chat.types.ts` | 前端类型定义 | ★☆☆ |
| `backend/app/ai/agent/multi_agent.py` | 多智能体（非 RAG） | ☆☆☆ |
| `backend/app/utils/chat_utils.py` | 消息转换工具 | ★☆☆ |
