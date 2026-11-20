import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

NIGHTLY = {
    "Qwen3-0.6B-W8A8", "Qwen3.5-0.8B", "Qwen3-8B-W8A8", "Qwen3.8-27B",
    "Qwen3.8-27B-w8a8", "Qwen3.8-27B-FP8", "Qwen3.6-35B-A3B",
    "Qwen3.6-35B-A3B-w8a8", "Qwen3.6-35B-A3B-w4a8-mxfp",
    "Qwen3.6-35B-A3B-w8a8-mxfp8",
}

MODEL_ORDER = [
    "Qwen3-0.6B", "Qwen3-0.6B-W8A8", "Qwen3-Embedding-0.6B", "Qwen3.5-0.8B",
    "Qwen2.5-VL-7B-Instruct", "Qwen3-8B-W8A8", "DeepSeek-V2-Lite-W8A8",
    "DeepSeek-V2-Lite", "Qwen3.8-27B-w8a8", "Qwen3.8-27B-w8a8-mxfp8",
    "Qwen3.8-27B-FP8", "Qwen3-30B-A3B-W8A8", "Qwen3.8-27B",
    "Qwen3.6-35B-A3B-w4a8-mxfp", "Qwen3.6-35B-A3B-w8a8",
    "Qwen3.6-35B-A3B-w8a8-mxfp8", "Qwen3.6-35B-A3B",
    "Kimi-K3-expert16-top8-100B-CountBenchQA", "Qwen3.8-Flash-Next",
    "GLM-5.3-Flash-w8a8", "GLM-5.2-w4a8c8", "DeepSeek-V4.1-Flash",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def correctness_dir(root, name):
    path = root / "correctness" / name
    if name == "Qwen3.5-0.8B" and not (path / "compare.json").is_file():
        path = path / name
    return path


def score(root, name, track, task):
    data = read_json(root / "accuracy" / name / track / "accuracy_results.json")
    if not data or task not in data:
        return "—"
    value = data[task]["score"]
    if isinstance(value, dict):
        value = value.get("accuracy", value.get("score"))
    return f"{value:.2f}%" if isinstance(value, (int, float)) else str(value)


def performance(root, name, track):
    directory = root / "performance" / name / track
    rows = {}
    for path in directory.glob("*_repeat*.json"):
        match = re.match(r"(.+)_repeat(\d+)\.json$", path.name)
        if not match:
            continue
        data = read_json(path)
        if data:
            rows.setdefault(match.group(1), []).append(data)
    return rows


def median(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def ratio(default, inductor, latency=False):
    if default is None or inductor is None or not default or not inductor:
        return None
    return default / inductor if latency else inductor / default


def speedup(value):
    if value is None:
        return "—"
    text = f"{value:.3f}×"
    return f"**{text}**" if value > 1.05 or value < 0.95 else text


def model_status(root, name):
    data = read_json(root / "status" / f"{name}.json")
    if not data:
        return "运行中"
    if data.get("requeue"):
        return "权重已确认，待补跑"
    stages = data.get("stages", {})
    failed = []
    for stage, result in stages.items():
        if result == 0:
            continue
        failed.append(f"{stage}={result}")
    if failed:
        return "；".join(f"**{item}**" for item in failed)
    if "correctness" not in stages:
        return "运行中"
    if name in NIGHTLY and any(f"accuracy_{track}" not in stages for track in ("eager", "inductor")):
        return "正确性运行中"
    if any(stage.startswith("performance_") for stage in stages):
        if all(stages.get(f"performance_{track}") == 0 for track in ("default", "inductor")):
            return "完成"
        return "性能运行中"
    if name == "Qwen3-Embedding-0.6B":
        return "完成"
    return "正确性完成，性能待运行"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    lines = [
        "# A5 Inductor 扩展 Nightly 测试结果", "",
        f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "本轮使用原容器环境、MRV2 和默认 O2。正确性以同权重 eager 为参考；性能比较生产 default 与 Inductor。每个性能值为同一负载重复测量的中位数。", "",
        "| 模型 | token 一致 | 文本一致 | MMLU eager → Inductor | GSM8K eager → Inductor | QPS1 | QPS4 | QPS16 | QPS inf | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    details = []
    for name in MODEL_ORDER:
        comparison = read_json(correctness_dir(args.root, name) / "compare.json")
        token_equal = text_equal = "—"
        if comparison:
            if "token_identical" in comparison:
                token_equal = f"{comparison['token_identical']}/{comparison['n_prompts']}"
                text_equal = f"{comparison['text_identical']}/{comparison['n_prompts']}"
            elif "embedding_metrics" in comparison:
                token_equal = f"最低余弦 {min(x['cosine'] for x in comparison['embedding_metrics']):.6f}"
        default = performance(args.root, name, "default")
        inductor = performance(args.root, name, "inductor")
        qps = []
        for rate in ("1", "4", "16", "inf"):
            load = f"sharegpt_qps_{rate}"
            d = median(default.get(load, []), "output_throughput")
            i = median(inductor.get(load, []), "output_throughput")
            qps.append(speedup(ratio(d, i)))
        lines.append(
            f"| {name} | {token_equal} | {text_equal} | {score(args.root, name, 'eager', 'mmlu')} → "
            f"{score(args.root, name, 'inductor', 'mmlu')} | {score(args.root, name, 'eager', 'gsm8k')} → "
            f"{score(args.root, name, 'inductor', 'gsm8k')} | {' | '.join(qps)} | {model_status(args.root, name)} |")
        loads = sorted(set(default) | set(inductor))
        if not loads:
            continue
        details.extend(["", f"## {name}", "",
                        "| 负载 | default 输出吞吐 | Inductor 输出吞吐 | 吞吐加速比 | TTFT 加速比 | TPOT 加速比 |",
                        "|---|---:|---:|---:|---:|---:|"])
        for load in loads:
            drows, irows = default.get(load, []), inductor.get(load, [])
            d = median(drows, "output_throughput")
            i = median(irows, "output_throughput")
            dttft = median(drows, "mean_ttft_ms")
            ittft = median(irows, "mean_ttft_ms")
            dtpot = median(drows, "mean_tpot_ms")
            itpot = median(irows, "mean_tpot_ms")
            details.append(
                f"| {load} | {f'{d:.2f}' if d is not None else '—'} | {f'{i:.2f}' if i is not None else '—'} | "
                f"{speedup(ratio(d, i))} | {speedup(ratio(dttft, ittft, True))} | "
                f"{speedup(ratio(dtpot, itpot, True))} |")
    output = args.root / "SUMMARY.md"
    output.write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    print(f"SUMMARY_UPDATED {output}")


if __name__ == "__main__":
    main()
