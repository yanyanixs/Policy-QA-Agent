<h1 align="center"> 政策问答智能体 </h1>
<p align="center">
  <a href="./README.md"  target="_Self">English</a> |
  <strong style="background-color: green;">中文</strong>
</p>

政策问答智能体是一个面向政府政策领域的 AI 智能问答系统，基于 LangGraph、FastAPI、NextJS、ChromaDB 等组件搭建，支持 RAG 增强检索、多智能体协作和流式对话。

<img src="./pictures/chat_img.png" width="700"/>

多智能体聊天:

<img src="./pictures/chat_multi_agent_img.png" width="700"/>

## 特性

1. 基于 LangGraph 框架搭建的智能体聊天应用，支持自定义智能体的行为逻辑编排。
2. 支持自定义智能体的知识库问答能力，基于 ChromaDB 来存储和查询知识库。
3. 支持自定义智能体的工具调用。
4. Python 后端接口 API，基于 FastAPI 实现，支持全异步调用。
5. 支持自定义智能体的前端应用，基于 NextJS 实现。
6. 支持聊天 Streaming 流输出，前端支持 SSE 流输出。
7. 支持自定义多个智能体。
8. 支持多智能体协作。
9. 聊天历史记录保存在本地浏览器缓存中。

## 前置依赖

在开始之前，请确保已安装以下工具：

| 依赖 | 版本要求 | 用途 | 安装方式 |
|------|---------|------|---------|
| Python | **≥ 3.13** | 后端运行时 | [python.org](https://www.python.org/downloads/) |
| Ollama | 最新版 | 本地 Embedding 模型 | [ollama.com](https://ollama.com/download) |
| pnpm | 最新版 | 前端包管理 | `npm install -g pnpm` |
| uv | 最新版 | Python 依赖管理 | `pip install uv` |

## 结构

- `backend` : 后端服务代码
- `frontend`: 前端服务代码

---

## 快速开始

以下步骤按顺序完成，预计耗时 **10-15 分钟**（不含模型下载时间）。

### 第一步：后端环境配置

```bash
# 进入后端目录
cd backend

# 复制并编辑环境变量
# Linux / macOS:
cp .env.example .env
# Windows (PowerShell):
copy .env.example .env
# ➜ 编辑 .env：填入你的 DEEPSEEK_API_KEY
```

**.env 关键配置说明**：

```properties
# ── 必填 ──────────────────────────────
DEEPSEEK_API_KEY=sk-your-key-here      # DeepSeek API Key（从 platform.deepseek.com 获取）
DEFAULT_MODEL=deepseek-v4-flash

# ── Embedding（需本地 Ollama）─────────
EMBEDDING_MODEL=bge-m3

# ── 数据库 ────────────────────────────
DATABASE_URL=sqlite+aiosqlite:///resource/database.db

# ── Reranker 精排（可选，模型不可用时自动降级）──
RERANKER_BACKEND=cross-encoder          # cross-encoder | llm | none
RERANKER_MODEL_PATH=BAAI/bge-reranker-v2-m3
HF_ENDPOINT=https://hf-mirror.com       # 国内必设，否则模型下载超时
```

### 第二步：安装依赖 & 启动后端

```bash
cd backend

# 安装 Python 依赖
uv sync --frozen

# 激活虚拟环境
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 启动后端服务（默认 http://localhost:8000）
python app/run_server.py
```

首次启动时会输出类似信息：
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 第三步：部署 Embedding 模型

后端依赖本地 Ollama 运行 BGE-M3 向量化模型（~1.2GB）：

```bash
# 安装并启动 Ollama（https://ollama.com/download）
ollama pull bge-m3
```

验证：
```bash
ollama list | grep bge-m3
# 输出: bge-m3:latest    xxx GB    ...
```

### 第四步：导入政策数据

数据导入是必须步骤 —— 没有数据，ChromaDB 为空，搜索不会有结果。

```bash
cd backend
# 确保虚拟环境已激活

# 导入示例政策数据（58 条北京市大数据/政务服务政策）
python scripts/import_policies.py resource/sample_policies.json

# 如果需要清空重新导入：
python scripts/import_policies.py resource/sample_policies.json --clean
```

导入成功后会打印：
```
导入完成: 58 条政策, xxx 个 chunk
```

### 第五步：配置 Reranker 精排（可选）

Reranker 用于对粗筛结果进行二次精排 + MMR 多样性重排：

| 后端 | 模型 | 精度 | 额外依赖 |
|------|------|------|---------|
| `cross-encoder` | BGE-Reranker-v2-m3 (~2GB) | ⭐⭐⭐ | 需下载模型 |
| `llm` | DeepSeek V4 Flash | ⭐⭐ | 无，用现有 API |
| `none` | 无 | ⭐ | 无，纯向量相似度 |

**国内用户重要提示**：`.env` 中 `HF_ENDPOINT` 已默认设为 `https://hf-mirror.com`，模型会自动从镜像站下载。若网络仍不通，改为 `RERANKER_BACKEND=llm` 即可跳过模型下载。

首次使用 `cross-encoder` 会下载约 2GB 模型文件，需要 1-3 分钟。之后会缓存到本地。

**自动降级**：如果模型下载失败或推理报错，系统会自动降级到 LLM 后端 → None 后端，不会阻塞使用。

### 第六步：配置前端

```bash
cd frontend

# 前端 API 地址（已默认配置为 localhost:8000，端口不同时需修改）
# .env.local 中: NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# 安装依赖
pnpm install

# 启动前端
pnpm dev
```

访问 http://localhost:3000/ 即可开始对话。

---

## 验证检索管道

使用追踪脚本验证完整的 RAG 链路（不调用 LLM，纯检索诊断）：

```bash
cd backend

# 基础追踪
python scripts/trace_query.py "新能源补贴" --dry-run

# 详细模式：查看每个 chunk 的详情和 MMR 排序
python scripts/trace_query.py "大数据跨境流动" --dry-run --verbose

# 对比 MMR 开关效果
python scripts/trace_query.py "大数据跨境流动" --dry-run --verbose --no-mmr
```

---

## 故障排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| `ModuleNotFoundError: No module named 'xxx'` | 虚拟环境未激活或依赖未安装 | `source .venv/bin/activate` 然后 `uv sync --frozen` |
| ChromaDB 返回空结果 | 未导入数据 | 执行第四步导入政策数据 |
| Ollama 连接失败 | Ollama 未启动 | 运行 `ollama serve` 启动服务 |
| Embedding 报错 | 未拉取 bge-m3 模型 | `ollama pull bge-m3` |
| Reranker 下载超时 | 国内无法访问 HuggingFace | `.env` 中已设 `HF_ENDPOINT=https://hf-mirror.com`，或改为 `RERANKER_BACKEND=llm` |
| `pip install uv` 失败 | pip 版本过旧 | `pip install --upgrade pip` |
| `python: command not found` | 需要 Python 3.13+ | 安装 [Python 3.13+](https://www.python.org/downloads/) 并确保在 PATH 中 |
| 前端请求 500 | 后端未启动或端口冲突 | 检查 `http://localhost:8000/docs` 是否可访问 |

---

## 自带智能体

本项目支持通过 LangGraph 创建并编排多个智能体，智能体编排逻辑在 `backend/app/ai/agent` 目录。前端可切换不同智能体进行对话。

### 1. 政策问答助手（POLICY-ASSISTANT）

面向政府政策的 RAG 问答智能体，支持语义搜索、全文获取、元数据精确筛选三种工具调用。

```
用户提问 → LLM 决策工具 → ChromaDB 检索 → Rerank+MMR 精排 → LLM 综合回答
```

具体参考：`backend/app/ai/agent/policy_assistant.py`

### 2. OA 助手（OA-ASSISTANT）

演示 OA 助手智能体，支持员工信息查询和员工手册知识库检索。

具体参考：`backend/app/ai/agent/oa_assistant.py`

### 3. 多智能体协作（MULTI_AGENT）

演示多智能体协作，包含三个子智能体：
- `math_agent`：数学计算
- `code_agent`：代码生成
- `general_agent`：通用问答

三个智能体通过 Supervisor 进行协作管理。

具体参考：`backend/app/ai/agent/multi_agent.py`
