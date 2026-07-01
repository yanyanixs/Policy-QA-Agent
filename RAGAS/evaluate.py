"""
RAGAS 评估主入口

接收 question / answer / contexts / ground_truths 四要素，
输出全部 9 项指标得分 + ragas_score 综合评分。

用法:
    # 命令行
    cd RAGAS
    python evaluate.py \\
        --question "新能源补贴有哪些政策" \\
        --answer "根据现行政策，新能源补贴主要包括..." \\
        --ground_truth "国家针对新能源汽车..." \\
        --contexts context1.txt context2.txt

    # 编程调用
    from evaluate import evaluate_rag

    result = evaluate_rag(
        question="...",
        answer="...",
        contexts=["...", "..."],
        ground_truths=["..."]
    )
"""

import sys
import os
import json
import argparse
import time
from typing import List, Dict, Optional

# Windows GBK → UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保当前目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics.retrieval import (
    context_precision,
    context_recall,
    context_relevancy,
    context_entity_recall,
)
from metrics.generation import (
    answer_relevancy,
    answer_similarity,
    answer_correctness,
)
from metrics.faithfulness import (
    faithfulness,
    hallucination_score,
)
from metrics.composite import (
    ragas_score,
    ragas_score_detailed,
    DEFAULT_WEIGHTS,
    METRIC_LABELS,
)


def evaluate_rag(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truths: List[str],
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> Dict:
    """
    评估 RAG 系统的完整性能。

    Args:
        question: 用户输入的问题
        answer: RAG 系统生成的答案
        contexts: 检索到的上下文文档列表
        ground_truths: 人工标注的真实答案列表
        weights: 自定义指标权重（默认使用 DEFAULT_WEIGHTS）
        verbose: 是否打印每项指标的计算结果

    Returns:
        {
            "question": str,
            "retrieval": {...},
            "generation": {...},
            "faithfulness": {...},
            "ragas_score": float,
            "elapsed_seconds": float,
        }
    """
    start_time = time.time()
    result = {"question": question}

    if verbose:
        print(f"[evaluate_rag] 评估开始: {question[:50]}...")

    def _run(label, fn, *args):
        t0 = time.time()
        try:
            score = fn(*args)
        except Exception as e:
            if verbose:
                print(f"  ✗ {label}: ERROR — {e}")
            score = None
        elapsed = time.time() - t0
        if verbose:
            status = f"{score:.4f}" if score is not None else "N/A"
            print(f"  {'✓' if score is not None else '✗'} {label}: {status} ({elapsed:.1f}s)")
        return score

    # ── 检索质量 ──
    retrieval = {}

    if verbose:
        print("\n── 检索质量 ──")
    retrieval["context_precision"] = _run("context_precision", context_precision, question, answer, contexts)
    retrieval["context_recall"] = _run("context_recall", context_recall, question, contexts, ground_truths)
    retrieval["context_relevancy"] = _run("context_relevancy", context_relevancy, question, contexts)
    retrieval["context_entity_recall"] = _run("context_entity_recall", context_entity_recall, question, contexts, ground_truths)

    result["retrieval"] = retrieval

    # ── 生成质量 ──
    generation = {}

    if verbose:
        print("\n── 生成质量 ──")
    generation["answer_relevancy"] = _run("answer_relevancy", answer_relevancy, question, answer)
    generation["answer_similarity"] = _run("answer_similarity", answer_similarity, answer, ground_truths)
    generation["answer_correctness"] = _run("answer_correctness", answer_correctness, answer, ground_truths)

    result["generation"] = generation

    # ── 事实一致性 ──
    faithfulness_metrics = {}

    if verbose:
        print("\n── 事实一致性 ──")
    faithfulness_metrics["faithfulness"] = _run("faithfulness", faithfulness, question, answer, contexts)
    faithfulness_metrics["hallucination_score"] = _run("hallucination_score", hallucination_score, question, answer, contexts)

    result["faithfulness"] = faithfulness_metrics

    # ── 综合得分 ──
    all_metrics = {}
    all_metrics.update(retrieval)
    all_metrics.update(generation)
    all_metrics.update(faithfulness_metrics)

    # 过滤掉 None 值
    valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}

    ragas = ragas_score(valid_metrics, weights)
    result["ragas_score"] = round(ragas, 4)

    elapsed = time.time() - start_time
    result["elapsed_seconds"] = round(elapsed, 1)

    if verbose:
        print(f"\n── 综合评分: {ragas:.4f} | 总耗时: {elapsed:.1f}s ──\n")

    return result


def evaluate_rag_detailed(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truths: List[str],
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> Dict:
    """
    同 evaluate_rag，但综合评分包含每个指标的详细贡献。
    """
    result = evaluate_rag(question, answer, contexts, ground_truths, weights, verbose)

    all_metrics = {}
    all_metrics.update(result["retrieval"])
    all_metrics.update(result["generation"])
    all_metrics.update(result["faithfulness"])
    valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}

    result["ragas_detail"] = ragas_score_detailed(valid_metrics, weights)
    return result


# ─── 交互式输入 ──────────────────────────────────────────


def _read_multiline(prompt: str, end_marker: str = "END") -> str:
    """读取多行输入，以单独一行的 end_marker 结束。"""
    print(f"\n{prompt}")
    print(f"  (输入完成后，单独一行输入 {end_marker} 结束)")
    print()
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == end_marker:
            break
        lines.append(line)
    return "\n".join(lines)


def _read_list_multiline(prompt: str, end_marker: str = "END") -> list:
    """读取多项多行输入，每项以 end_marker 分隔，全部以 DONE 结束。"""
    print(f"\n{prompt}")
    print(f"  (每条输入完成后单独一行输入 {end_marker} 分隔，全部完成后输入 DONE)")
    print()
    items = []
    current_lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        stripped = line.strip()
        if stripped == "DONE":
            if current_lines:
                items.append("\n".join(current_lines))
            break
        if stripped == end_marker:
            items.append("\n".join(current_lines))
            current_lines = []
        else:
            current_lines.append(line)
    return items


def _interactive_input():
    """交互式收集四要素"""
    print("\n" + "=" * 60)
    print("  RAGAS 评估 — 交互式输入模式")
    print("=" * 60)

    # 1. question
    print("\n" + "-" * 40)
    question = input("请输入 question（用户问题）: ").strip()
    while not question:
        print("  ⚠ question 不能为空，请重新输入")
        question = input("请输入 question（用户问题）: ").strip()

    # 2. answer
    answer = _read_multiline(
        "请输入 answer（RAG 系统生成的答案）:",
        end_marker="END",
    )
    while not answer.strip():
        print("  ⚠ answer 不能为空，请重新输入")
        answer = _read_multiline(
            "请输入 answer（RAG 系统生成的答案）:",
            end_marker="END",
        )

    # 3. contexts
    contexts = _read_list_multiline(
        "请输入 contexts（检索到的上下文文档，每条以 END 分隔）:",
        end_marker="END",
    )
    while not contexts:
        print("  ⚠ 至少需要一条 context，请重新输入")
        contexts = _read_list_multiline(
            "请输入 contexts（检索到的上下文文档，每条以 END 分隔）:",
            end_marker="END",
        )

    # 4. ground_truths
    ground_truths = _read_list_multiline(
        "请输入 ground_truths（人工标注的真实答案，每条以 END 分隔）:",
        end_marker="END",
    )
    while not ground_truths:
        print("  ⚠ 至少需要一条 ground_truth，请重新输入")
        ground_truths = _read_list_multiline(
            "请输入 ground_truths（人工标注的真实答案，每条以 END 分隔）:",
            end_marker="END",
        )

    # 确认
    print("\n" + "=" * 60)
    print("  输入汇总")
    print("=" * 60)
    print(f"\n  question: {question[:80]}{'...' if len(question) > 80 else ''}")
    print(f"  answer 长度: {len(answer)} 字符")
    print(f"  contexts: {len(contexts)} 条")
    print(f"  ground_truths: {len(ground_truths)} 条")

    confirm = input("\n确认开始评估? (Y/n): ").strip().lower()
    if confirm and confirm != "y":
        print("已取消。")
        sys.exit(0)

    return question, answer, contexts, ground_truths


# ─── CLI ────────────────────────────────────────────────


def _read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_weights(weight_strs: List[str]) -> Dict[str, float]:
    """解析权重参数: --weight context_precision=0.2 --weight faithfulness=0.3"""
    weights = {}
    for w in weight_strs:
        name, val = w.split("=", 1)
        weights[name.strip()] = float(val.strip())
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="RAGAS 评估 — 对 RAG 系统进行多维度自动评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate.py \\
      --question "新能源补贴有哪些政策" \\
      --answer "根据现行政策..." \\
      --ground_truth "国家针对新能源汽车..." \\
      --contexts ctx1.txt ctx2.txt

  python evaluate.py \\
      --question "北京数据政策" \\
      --answer "..." \\
      --ground_truth "..." \\
      --context-files contexts.json \\
      --verbose
        """,
    )
    parser.add_argument("--question", default=None, help="用户输入的问题")
    parser.add_argument("--answer", help="RAG 系统生成的答案（如不提供则从文件读取 --answer-file）")
    parser.add_argument("--answer-file", help="从文件读取答案")
    parser.add_argument("--ground-truth", nargs="+", help="人工标注的真实答案（一条或多条）")
    parser.add_argument("--ground-truth-file", nargs="+", help="从文件读取真实答案")
    parser.add_argument("--contexts", nargs="+", help="上下文文档（直接传入文本）")
    parser.add_argument("--context-files", nargs="+", help="上下文文档（从文件读取）")
    parser.add_argument("--weight", nargs="*", default=[], help="自定义权重，如 --weight context_precision=0.2")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    parser.add_argument("--detailed", "-d", action="store_true", help="输出综合评分的详细分解")
    parser.add_argument("--output", "-o", help="保存结果为 JSON 文件")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式模式：手动输入 question/answer/contexts/ground_truths")
    parser.add_argument("--list-metrics", action="store_true", help="列出所有可用指标及其默认权重")

    args = parser.parse_args()

    # 列出指标
    if args.list_metrics:
        print("可用指标（默认权重）:")
        for name, weight in DEFAULT_WEIGHTS.items():
            label = METRIC_LABELS.get(name, name)
            print(f"  {name:30s} = {weight:.2f}  ({label})")
        return

    # 交互式模式
    if args.interactive:
        question, answer, contexts, ground_truths = _interactive_input()
        args.question = question
        args.verbose = args.verbose if args.verbose else True  # 交互式默认 verbose
    else:
        # 解析四要素
        if not args.question:
            parser.error("必须提供 --question（或用 --interactive 进入交互模式）")
        question = args.question
        answer = args.answer or (args.answer_file and _read_file(args.answer_file))
        if not answer:
            parser.error("必须提供 --answer 或 --answer-file")

        ground_truths = args.ground_truth or []
        if args.ground_truth_file:
            for gf in args.ground_truth_file:
                ground_truths.append(_read_file(gf))
        if not ground_truths:
            parser.error("必须提供 --ground-truth 或 --ground-truth-file")

        contexts = args.contexts or []
        if args.context_files:
            for cf in args.context_files:
                if cf.endswith(".json"):
                    data = json.loads(_read_file(cf))
                    if isinstance(data, list):
                        contexts.extend(data)
                    else:
                        contexts.append(str(data))
                else:
                    contexts.append(_read_file(cf))
        if not contexts:
            parser.error("必须提供 --contexts 或 --context-files")

    # 权重
    weights = _parse_weights(args.weight) if args.weight else None

    # 执行评估
    fn = evaluate_rag_detailed if args.detailed else evaluate_rag
    result = fn(question, answer, contexts, ground_truths, weights, verbose=args.verbose)

    # 输出
    output_json = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"结果已保存到: {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
