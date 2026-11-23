import argparse
import json
from pathlib import Path


def format_speedup(value, digits=3):
    text = f"{value:.{digits}f}"
    if value > 1.05:
        return f"**{text}**"
    if value < 0.95:
        return f"*{text}*"
    return text


def describe(data, source):
    kind = data.get("comparison_type", "unknown")
    if kind == "generation":
        result = (f"token {data['token_identical']}/{data['n_prompts']}；"
                  f"text {data['text_identical']}/{data['n_prompts']}；"
                  f"logprob 容差 {data.get('logprob_within_tolerance')}")
    elif kind == "embedding":
        cosine = min(row["cosine"] for row in data["embedding_metrics"])
        result = f"最低余弦 {cosine:.6f}；容差内 {data['within_tolerance']}"
    elif kind == "task_accuracy":
        result = "；".join(f"{task} 差值 {values['difference_percentage_points']:+.3f} pp"
                          for task, values in data["tasks"].items())
        sample_changes = [f"{task} 对→错 {values['sample_comparison']['correct_to_wrong']}，"
                          f"错→对 {values['sample_comparison']['wrong_to_correct']}"
                          for task, values in data["tasks"].items()
                          if values.get("sample_comparison", {}).get("available")]
        if sample_changes:
            result += "；逐样本 " + "；".join(sample_changes)
    elif kind == "performance":
        result = (f"吞吐加速比 {format_speedup(data['output_throughput_speedup'])}；"
                  f"TPOT 加速比 {format_speedup(data['tpot_speedup'])}；"
                  f"TTFT 加速比 {format_speedup(data['ttft_speedup'])}；"
                  f"阈值内 {data['within_thresholds']}")
    elif kind == "inductor_evidence":
        evidence = data["evidence"]
        result = f"有 Inductor 证据 {evidence['has_inductor_evidence']}"
        actual = evidence.get("actual_compilation_config")
        if actual:
            result += f"；实际 backend {actual.get('backend')}；图模式 {actual.get('cudagraph_mode')}"
    else:
        result = "未识别的结果格式"
    return f"| {kind} | {source} | {result} |"


def main():
    parser = argparse.ArgumentParser(description="Summarize explicitly selected comparison JSON files.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    lines = ["# 测试结果汇总", "", "本表只汇总命令行明确指定的文件，不代表未列出的测试已通过。", "",
             "| 类型 | 结果文件 | 摘要 |", "|---|---|---|"]
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(describe(data, path))
    output = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"SUMMARY_UPDATED {args.out}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
