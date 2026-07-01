# RAGAS 评估模块

对 RAG 系统进行多维度自动评估，基于大模型 + 语义向量计算 9 项核心指标，并输出综合评分。

## 目录结构

```
RAGAS/
├── evaluate.py              # 主入口（CLI + 编程调用）
├── metrics/
│   ├── retrieval.py         # 检索质量指标（4 项）
│   ├── generation.py        # 生成质量指标（3 项）
│   ├── faithfulness.py      # 事实一致性指标（2 项）
│   └── composite.py         # 综合评分
└── utils/
    ├── llm_client.py        # LLM 调用封装（复用后端 DeepSeek）
    └── embedding.py         # 向量化封装（复用后端 BGE-M3）
```

## 四要素

RAGAS 评估需要以下四项输入：

| 要素 | 含义 | 来源 |
|------|------|------|
| `question` | 用户问题 | 用户输入 |
| `answer` | RAG 系统生成的答案 | LLM 输出 |
| `contexts` | 检索到的上下文文档列表 | ChromaDB 检索结果 |
| `ground_truths` | 人工标注的真实答案 | **唯一需要人工提供** |

## 评估指标（9 项）

### 检索质量

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| `context_precision` | 上下文精确性 | LLM 判断每条上下文是否「必要」→ 必要数 / 总数 |
| `context_recall` | 上下文召回率 | LLM 从真实答案提取关键信息点 → 检查是否被检索到 |
| `context_relevancy` | 上下文相关性 | 语义向量计算上下文与问题的余弦相似度 |
| `context_entity_recall` | 实体召回率 | LLM 提取关键实体 → 统计检索覆盖率 |

### 生成质量

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| `answer_relevancy` | 答案相关性 | LLM 从答案反推提问 → 与原始问题计算语义相似度 |
| `answer_similarity` | 答案相似度 | 生成答案与真实答案的语义向量余弦相似度 |
| `answer_correctness` | 答案正确性 | LLM 拆解为事实陈述 → 逐一比对匹配程度 |

### 事实一致性

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| `faithfulness` | 忠实性 | LLM 拆解答案陈述 → 逐条验证能否在上下文中找到依据 |
| `hallucination_score` | 幻觉评分 | 无依据陈述数 / 总陈述数（越低越好，RAGAS 综合分中自动反转） |

### 综合评分

`ragas_score` = 以上 9 项指标按默认权重加权求和（0~1），默认权重：

| 指标 | 权重 |
|------|------|
| context_precision | 0.15 |
| context_recall | 0.15 |
| context_relevancy | 0.10 |
| context_entity_recall | 0.05 |
| answer_relevancy | 0.10 |
| answer_similarity | 0.10 |
| answer_correctness | 0.15 |
| faithfulness | 0.15 |
| hallucination_score | 0.05 |

## 前置依赖

| 依赖 | 用途 |
|------|------|
| Python ≥ 3.10 | 运行环境 |
| 后端 `.env` | LLM API Key（DeepSeek）和 Embedding 配置 |
| Ollama + BGE-M3 | 本地 Embedding 模型 |

确保 `backend/.env` 已正确配置 `DEEPSEEK_API_KEY`，且 Ollama 已拉取 `bge-m3` 模型。

## 使用方式

### 方式一：从 trace_query.py 输出直接评估（推荐）

```bash
# Step 1: 在 backend 目录跑一次追踪，输出 RAGAS 四要素 JSON
cd backend
python scripts/trace_query.py "新能源补贴有哪些政策" -o ragas_input.json -g "国家针对新能源汽车的补贴政策主要包括..."

# Step 2: 在 RAGAS 目录直接评估
cd ../RAGAS
python evaluate.py --from-trace ../backend/ragas_input.json -v
```

### 方式二：命令行传参

```bash
python evaluate.py \
    --question "新能源补贴有哪些政策" \
    --answer "根据现行政策，新能源补贴主要包括..." \
    --ground_truth "国家针对新能源汽车..." \
    --contexts "context1内容" "context2内容" \
    --verbose
```

从文件读取：

```bash
python evaluate.py \
    --question "新能源补贴有哪些政策" \
    --answer-file answer.txt \
    --ground-truth-file gt1.txt gt2.txt \
    --context-files ctx1.txt ctx2.txt \
    --verbose
```

### 方式三：交互式输入

```bash
python evaluate.py -i
```

按提示依次输入 question、answer、contexts、ground_truths。

### 方式四：编程调用

```python
from evaluate import evaluate_rag, evaluate_rag_detailed

result = evaluate_rag(
    question="新能源补贴有哪些政策",
    answer="根据现行政策，新能源补贴主要包括...",
    contexts=["北京市2024年新能源汽车推广方案...", "财政部关于新能源补贴的通知..."],
    ground_truths=["国家针对新能源汽车的补贴政策包括..."],
)

print(f"综合得分: {result['ragas_score']}")
print(f"检索质量: {result['retrieval']}")
print(f"生成质量: {result['generation']}")
print(f"事实一致性: {result['faithfulness']}")
```

## CLI 参数

| 参数 | 说明 |
|------|------|
| `--question` | 用户问题 |
| `--answer` / `--answer-file` | 生成答案（直接传 / 从文件读） |
| `--contexts` / `--context-files` | 上下文（直接传 / 从文件读，支持 .json） |
| `--ground-truth` / `--ground-truth-file` | 真实答案（直接传 / 从文件读） |
| `--from-trace` | 从 `trace_query.py` 输出的 JSON 直接加载四要素 |
| `-i`, `--interactive` | 交互式输入模式 |
| `-v`, `--verbose` | 打印每项指标的计算详情 |
| `-d`, `--detailed` | 输出综合评分的逐项贡献明细 |
| `-o`, `--output` | 保存结果为 JSON 文件 |
| `--weight` | 自定义权重，如 `--weight faithfulness=0.3 context_precision=0.2` |
| `--list-metrics` | 列出所有可用指标及默认权重 |

## 输出格式

```json
{
  "question": "新能源补贴有哪些政策",
  "retrieval": {
    "context_precision": 0.85,
    "context_recall": 0.72,
    "context_relevancy": 0.91,
    "context_entity_recall": 0.68
  },
  "generation": {
    "answer_relevancy": 0.88,
    "answer_similarity": 0.76,
    "answer_correctness": 0.81
  },
  "faithfulness": {
    "faithfulness": 0.90,
    "hallucination_score": 0.08
  },
  "ragas_score": 0.823,
  "elapsed_seconds": 45.3
}
```

## 自定义权重

```bash
# 提高事实一致性的权重
python evaluate.py --from-trace ragas_input.json \
    --weight faithfulness=0.25 context_precision=0.10 hallucination_score=0.10
```

## 注意事项

- 评估依赖 LLM 和 Embedding 模型，**需保持网络连接**
- LLM 调用温度设为 0（确定性输出），确保评估结果可复现
- 完整评估 9 项指标约需 30~120 秒（取决于 contexts 数量）
- `--from-trace` 要求 trace JSON 中包含非空的 `answer`（即不能用 `--dry-run` 生成）
- 首次运行会自动下载 Reranker 模型（如果后端启用 cross-encoder）
