#!/usr/bin/env python3
"""Summarize capture-1024 batch results (performance and accuracy) as Markdown and JSON."""
import argparse
import json
import statistics
from pathlib import Path

LOADS = (("sharegpt_qps_1", "ShareGPT QPS1"), ("sharegpt_qps_4", "ShareGPT QPS4"),
         ("sharegpt_qps_16", "ShareGPT QPS16"), ("sharegpt_qps_inf", "ShareGPT QPSinf"),
         ("random_8192_1024", "随机 8192/1024"), ("random_32768_1024", "随机 32768/1024"))
SPEEDUPS = (("output_throughput_speedup", "output_throughput", "吞吐加速比"),
            ("tpot_speedup", "mean_tpot_ms", "TPOT 加速比"),
            ("ttft_speedup", "mean_ttft_ms", "TTFT 加速比"))
OFFICIAL_KEYS = ("completed", "failed", "num_prompts", "output_throughput", "mean_ttft_ms",
                 "mean_tpot_ms", "p99_ttft_ms", "p99_tpot_ms", "duration")
TRACKS = ("default", "inductor")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path):
    return load(path) if path.is_file() else None


def num(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def mark(value):
    text = f"{value:.3f}"
    if value > 1.05:
        return f"**{text}**"
    if value < 0.95:
        return f"*{text}*"
    return text


def spread(values):
    if not values:
        return "—"
    return f"{mark(statistics.median(values))} [{min(values):.3f}, {max(values):.3f}]"


def pair(values):
    default = [v["default"] for v in values if v.get("default")]
    inductor = [v["inductor"] for v in values if v.get("inductor")]
    if not default or not inductor:
        return "—"
    digits = 3 if max(default + inductor) < 100 else 1
    return f"{statistics.median(default):.{digits}f} → {statistics.median(inductor):.{digits}f}"


def track_info(root, track, rounds):
    rows = {}
    for repeat in range(1, rounds + 1):
        server = optional(root / f"{track}.repeat{repeat}.server.json")
        runtime = optional(root / f"{track}.repeat{repeat}.runtime.json")
        if server is None and runtime is None:
            continue
        requested = (server or {}).get("requested_compilation_config") or {}
        actual = (server or {}).get("actual_compilation_config") or {}
        rows[repeat] = {"capture_size": requested.get("max_cudagraph_capture_size"),
                        "capture_sizes": requested.get("cudagraph_capture_sizes"),
                        "backend": actual.get("backend"), "cudagraph_mode": actual.get("cudagraph_mode"),
                        "level": actual.get("level"), "cache_state": (runtime or {}).get("cache_state"),
                        "return_code": (runtime or {}).get("return_code"),
                        "command": (server or {}).get("command")}
    return rows


def performance(name, root, rounds):
    result = {"name": name, "root": str(root),
              "tracks": {track: track_info(root, track, rounds) for track in TRACKS},
              "loads": {}, "evidence": {}}
    for repeat in range(1, rounds + 1):
        evidence = optional(root / f"inductor.repeat{repeat}.evidence.json")
        if evidence:
            inner = evidence["evidence"]
            result["evidence"][repeat] = {
                "has_inductor_evidence": inner["has_inductor_evidence"],
                "actual_compilation_config": inner.get("actual_compilation_config"),
                "artifact_count": len(inner.get("artifacts", []))}
    for key, _ in LOADS:
        rows = {}
        for repeat in range(1, rounds + 1):
            compare_path = root / "compare" / f"{key}_repeat{repeat}.json"
            compare = optional(compare_path)
            if compare is None:
                continue
            row = {"within_thresholds": compare["within_thresholds"],
                   "speedups": {s: compare[s] for s, _, _ in SPEEDUPS if s in compare}, "metrics": {}}
            missing = [s for s, _, _ in SPEEDUPS if s not in row["speedups"]]
            if missing:
                raise SystemExit(f"{compare_path}: missing speedup fields {missing}")
            for track in TRACKS:
                official = optional(root / track / f"{key}_repeat{repeat}.json")
                row["metrics"][track] = ({k: official.get(k) for k in OFFICIAL_KEYS}
                                         if official else None)
            rows[repeat] = row
        result["loads"][key] = rows
    return result


def accuracy(name, root):
    compare = optional(root / "compare.json")
    if compare is None:
        return None
    result = {"name": name, "root": str(root), "n_prompts": compare["n_prompts"],
              "token_identical": compare["token_identical"], "text_identical": compare["text_identical"],
              "logprob_within_tolerance": compare["logprob_within_tolerance"],
              "logprob_summary": compare["logprob_summary"],
              "logprob_tolerance": compare["logprob_tolerance"], "prompts": [], "tracks": {}}
    first_diff = {item["idx"]: item["first_diff_pos"] for item in compare.get("diff_details", [])}
    for item in compare["logprob_by_prompt"]:
        result["prompts"].append({
            "idx": item["idx"], "first_diff_pos": first_diff.get(item["idx"]),
            "pr_failure": item.get("pr_failure"),
            "mean_abs_logprob_diff": item.get("mean_abs_logprob_diff"),
            "max_abs_logprob_diff": item.get("max_abs_logprob_diff")})
    for track in ("eager", "inductor"):
        run = optional(root / track / "run_config.json")
        runtime = optional(root / f"{track}.runtime.json")
        result["tracks"][track] = {
            "engine_init_s": run.get("engine_init_s") if run else None,
            "actual_compilation_config": run.get("actual_compilation_config") if run else None,
            "engine_kwargs": run.get("engine_kwargs") if run else None,
            "cache_state": runtime.get("cache_state") if runtime else None,
            "return_code": runtime.get("return_code") if runtime else None}
    evidence = optional(root / "inductor_evidence.json")
    result["evidence"] = evidence["evidence"] if evidence else None
    return result


def describe_failure(failure):
    if failure is None:
        return "通过"
    reason = failure["reason"]
    position = failure.get("position")
    detail = ""
    if reason.endswith("token_mismatch") and "reference_token" in failure:
        detail = f"（eager {failure['reference_token']} → Inductor {failure['candidate_token']}）"
    elif "logprob_diff" in failure:
        detail = f"（差值 {failure['logprob_diff']:.6f}）"
    return f"`{reason}`@{position}{detail}" if position is not None else f"`{reason}`{detail}"


def render_accuracy(data):
    lines = [f"### {data['name']}：固定输入 token/logprob", ""]
    summary = data["logprob_summary"]
    tolerance = data["logprob_tolerance"]
    passed = sum(p["pr_failure"] is None for p in data["prompts"])
    lines.append(f"- token 完全一致：{data['token_identical']}/{data['n_prompts']}；"
                 f"PR 规则通过：{passed}/{data['n_prompts']}；"
                 f"`logprob_within_tolerance`={data['logprob_within_tolerance']}。")
    lines.append(f"- 分叉前的 logprob 绝对差：平均 {num(summary['mean_abs_logprob_diff'], 6)}，"
                 f"最大 {num(summary['max_abs_logprob_diff'], 6)}；容差：prefill "
                 f"{tolerance['prefill_abs_logprob_diff_max']}，decode "
                 f"{tolerance['decode_abs_logprob_diff_max']}，Top-{tolerance['decode_topk']}。")
    for track in ("eager", "inductor"):
        info = data["tracks"][track]
        actual = info["actual_compilation_config"] or {}
        lines.append(
            f"- {track}：引擎初始化 {num(info['engine_init_s'], 0)} 秒；缓存状态 {info['cache_state']}；"
            f"返回码 {info['return_code']}；实际 backend {actual.get('backend')}，"
            f"图模式 {actual.get('cudagraph_mode')}，级别 {actual.get('level')}。")
    evidence = data["evidence"] or {}
    lines.append(f"- Inductor 路径证据：{evidence.get('has_inductor_evidence')}"
                 f"（编译产物 {len(evidence.get('artifacts', []))} 个）。")
    lines += ["", "| prompt | 输入类型 | 首次分叉位置 | 首个未通过的规则 |", "|---:|---|---:|---|"]
    for prompt in data["prompts"]:
        first = prompt["first_diff_pos"]
        lines.append(f"| p{prompt['idx']} | {'短' if prompt['idx'] < 4 else '长'} | "
                     f"{'—' if first is None else first} | {describe_failure(prompt['pr_failure'])} |")
    return lines


def render_performance(data, rounds):
    lines = [f"### {data['name']}：default → Inductor 性能", ""]
    captures = sorted({row["capture_size"] for rows in data["tracks"].values()
                       for row in rows.values() if row["capture_size"] is not None})
    states = [data["tracks"]["inductor"][r]["cache_state"] for r in sorted(data["tracks"]["inductor"])]
    evidence = data["evidence"]
    lines.append(f"- 捕获上限：{'、'.join(map(str, captures)) or '未记录'}（两条路径相同）；"
                 f"Inductor 缓存状态（逐轮）：{'、'.join(states) or '无记录'}；"
                 f"Inductor 证据：{sum(1 for e in evidence.values() if e['has_inductor_evidence'])}/{rounds} 轮。")
    lines += ["", "每格 default → Inductor 和加速比均取各轮中位数，方括号内是各轮的最小值和最大值。", "",
              "| 负载 | 轮数 | 吞吐 (tokens/s) | TPOT (ms) | TTFT (ms) | 吞吐加速比 | TPOT 加速比 | TTFT 加速比 | 阈值内 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in LOADS:
        rows = data["loads"].get(key) or {}
        if not rows:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | 0/{rounds} |")
            continue
        cells = [label, str(len(rows))]
        for _, metric, _ in SPEEDUPS:
            cells.append(pair([{"default": row["metrics"]["default"][metric],
                                "inductor": row["metrics"]["inductor"][metric]}
                               if row["metrics"]["default"] and row["metrics"]["inductor"] else {}
                               for row in rows.values()]))
        for speedup, _, _ in SPEEDUPS:
            cells.append(spread([row["speedups"][speedup] for row in rows.values()]))
        cells.append(f"{sum(row['within_thresholds'] for row in rows.values())}/{rounds}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "逐轮明细：", "",
              "| 负载 | 轮次 | 吞吐加速比 | TPOT 加速比 | TTFT 加速比 | 阈值内 | 完成请求 default/Inductor |",
              "|---|---:|---:|---:|---:|---|---:|"]
    for key, label in LOADS:
        for repeat, row in sorted((data["loads"].get(key) or {}).items()):
            metrics = row["metrics"]
            done = "—"
            if metrics["default"] and metrics["inductor"]:
                done = (f"{metrics['default']['completed']}/{metrics['default']['num_prompts']} : "
                        f"{metrics['inductor']['completed']}/{metrics['inductor']['num_prompts']}")
            lines.append(f"| {label} | {repeat} | " + " | ".join(mark(row["speedups"][s]) for s, _, _ in SPEEDUPS)
                         + f" | {'是' if row['within_thresholds'] else '否'} | {done} |")
    return lines


def render_overview(entries, rounds):
    lines = ["| 模型 | 有效轮数 | 吞吐加速比中位数范围 | TPOT 加速比中位数范围 | TTFT 加速比中位数范围 | 阈值外（轮次×负载） |",
             "|---|---:|---:|---:|---:|---:|"]
    for data in entries:
        have_rounds = {r for rows in data["loads"].values() for r in rows}
        medians = {s: [statistics.median(row["speedups"][s] for row in rows.values())
                       for rows in data["loads"].values() if rows] for s, _, _ in SPEEDUPS}
        outside = sum(not row["within_thresholds"] for rows in data["loads"].values()
                      for row in rows.values())
        total = sum(len(rows) for rows in data["loads"].values())
        cells = [data["name"], f"{len(have_rounds)}/{rounds}"]
        for s, _, _ in SPEEDUPS:
            values = medians[s]
            cells.append(f"{min(values):.3f} – {max(values):.3f}" if values else "—")
        cells.append(f"{outside}/{total}" if total else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--entry", action="append", default=[], help="NAME=PERFORMANCE_DIR")
    parser.add_argument("--acc", action="append", default=[], help="NAME=CORRECTNESS_DIR")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    performances, accuracies = [], []
    seen_acc = set()
    if args.batch:
        for entry in sorted(p for p in args.batch.iterdir() if p.is_dir()):
            if (entry / "perf" / "performance").is_dir():
                performances.append(performance(entry.name, entry / "perf" / "performance", args.rounds))
            if (entry / "acc" / "correctness" / "compare.json").is_file():
                accuracies.append(accuracy(entry.name, entry / "acc" / "correctness"))
                seen_acc.add(entry.name)
    for item in args.entry:
        name, _, path = item.partition("=")
        performances.append(performance(name, Path(path), args.rounds))
    for item in args.acc:
        name, _, path = item.partition("=")
        if name not in seen_acc:
            accuracies.append(accuracy(name, Path(path)))

    lines = ["## 精度（固定输入）", ""]
    for data in accuracies:
        lines += render_accuracy(data) + [""]
    lines += ["## 性能总览", "",
              "加速比均以 Inductor 相对 default 计算，三项都是越高越好；小于 0.95 用斜体、大于 1.05 用粗体。", ""]
    lines += render_overview(performances, args.rounds) + [""]
    for data in performances:
        lines += render_performance(data, args.rounds) + [""]
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.json:
        args.json.write_text(json.dumps({"performance": performances, "accuracy": accuracies},
                                        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
