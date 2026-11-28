#!/usr/bin/env python3
"""Render the Markdown tables for the cap-1024 batch report from summary.json."""
import json
import re
import statistics
import sys
from pathlib import Path

LOADS = (("sharegpt_qps_1", "ShareGPT QPS1"), ("sharegpt_qps_4", "ShareGPT QPS4"),
         ("sharegpt_qps_16", "ShareGPT QPS16"), ("sharegpt_qps_inf", "ShareGPT QPSinf"),
         ("random_8192_1024", "随机 8192/1024"), ("random_32768_1024", "随机 32768/1024"))
SPEEDUPS = (("output_throughput_speedup", "output_throughput", "吞吐加速比"),
            ("tpot_speedup", "mean_tpot_ms", "TPOT 加速比"),
            ("ttft_speedup", "mean_ttft_ms", "TTFT 加速比"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mark(value):
    text = f"{value:.3f}"
    if value > 1.05:
        return f"**{text}**"
    if value < 0.95:
        return f"*{text}*"
    return text


def spread(values):
    return f"{mark(statistics.median(values))} [{min(values):.3f}, {max(values):.3f}]"


def pair(values):
    default = [v["default"] for v in values if v.get("default")]
    inductor = [v["inductor"] for v in values if v.get("inductor")]
    if not default or not inductor:
        return "—"
    digits = 3 if max(default + inductor) < 100 else 1
    return f"{statistics.median(default):.{digits}f} → {statistics.median(inductor):.{digits}f}"


def rows_of(entry):
    return {key: entry["loads"].get(key) or {} for key, _ in LOADS}


def overview(entries):
    lines = ["| 模型 | 负载 | 吞吐加速比中位数 | TPOT 加速比中位数 | TTFT 加速比中位数 | 阈值内轮数 |",
             "|---|---|---:|---:|---:|---:|"]
    for entry in entries:
        for key, label in LOADS:
            rows = rows_of(entry)[key]
            if not rows:
                continue
            cells = [entry["name"], label]
            for s, _, _ in SPEEDUPS:
                cells.append(mark(statistics.median(r['speedups'][s] for r in rows.values())))
            cells.append(f"{sum(r['within_thresholds'] for r in rows.values())}/{len(rows)}")
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def detail(entry):
    lines = [f"### {entry['name']}：default → Inductor 性能", ""]
    rounds = sorted({r for rows in entry["loads"].values() for r in rows})
    caps = sorted({row["capture_size"] for t in entry["tracks"].values() for row in t.values() if row["capture_size"] is not None})
    states = [entry["tracks"]["inductor"][r]["cache_state"] for r in rounds if r in entry["tracks"]["inductor"]]
    evidence = entry["evidence"]
    lines.append(f"- 捕获上限：{'、'.join(map(str, caps)) or '未记录'}（两条路径相同）；Inductor 缓存状态（逐轮）：{'、'.join(states) or '无记录'}；"
                 f"Inductor 证据：{sum(1 for e in evidence.values() if e['has_inductor_evidence'])}/{len(rounds)} 轮。")
    lines += ["", "每格 default → Inductor 和加速比均取各轮中位数，方括号内是各轮的最小值和最大值。", "",
              "| 负载 | 轮数 | 吞吐 (tokens/s) | TPOT (ms) | TTFT (ms) | 吞吐加速比 | TPOT 加速比 | TTFT 加速比 | 阈值内 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in LOADS:
        rows = rows_of(entry)[key]
        if not rows:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | 0/{len(rounds)} |")
            continue
        cells = [label, str(len(rows))]
        for _, metric, _ in SPEEDUPS:
            cells.append(pair([{"default": row["metrics"]["default"][metric], "inductor": row["metrics"]["inductor"][metric]}
                               if row["metrics"]["default"] and row["metrics"]["inductor"] else {} for row in rows.values()]))
        for s, _, _ in SPEEDUPS:
            cells.append(spread([row["speedups"][s] for row in rows.values()]))
        cells.append(f"{sum(row['within_thresholds'] for row in rows.values())}/{len(rows)}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "逐轮明细：", "",
              "| 负载 | 轮次 | 吞吐加速比 | TPOT 加速比 | TTFT 加速比 | 阈值内 | 完成请求 default/Inductor |",
              "|---|---:|---:|---:|---:|---|---:|"]
    for key, label in LOADS:
        for repeat, row in sorted(rows_of(entry)[key].items()):
            metrics = row["metrics"]
            done = "—"
            if metrics["default"] and metrics["inductor"]:
                done = (f"{metrics['default']['completed']}/{metrics['default']['num_prompts']} : "
                        f"{metrics['inductor']['completed']}/{metrics['inductor']['num_prompts']}")
            lines.append(f"| {label} | {repeat} | " + " | ".join(mark(row["speedups"][s]) for s, _, _ in SPEEDUPS)
                         + f" | {'是' if row['within_thresholds'] else '否'} | {done} |")
    return lines


def versus_previous(entries, previous):
    prev = {e["name"]: e for e in previous}
    lines = ["| 模型 | 负载 | 上一批 吞吐/TPOT/TTFT | 本批 吞吐/TPOT/TTFT |", "|---|---|---|---|"]
    for entry in entries:
        old = prev.get(entry["name"])
        if old is None:
            continue
        for key, label in LOADS:
            new_rows = rows_of(entry)[key]
            old_rows = {r: row for r, row in (old["loads"].get(key) or {}).items()}
            if not new_rows or not old_rows:
                continue
            def med(rows, s):
                return statistics.median(row["speedups"][s] for row in rows.values())
            o = "/".join(f"{med(old_rows, s):.3f}" for s, _, _ in SPEEDUPS)
            n = "/".join(f"{med(new_rows, s):.3f}" for s, _, _ in SPEEDUPS)
            lines.append(f"| {entry['name']} | {label} | {o} | {n} |")
    return lines


def compile_times(root, entries):
    lines = ["| 模型 | 第 1 轮 Inductor 引擎初始化（cold） | 第 2 轮 Inductor 引擎初始化（warm） | 第 2 轮日志含 `Directly load the compiled graph` |",
             "|---|---:|---:|---:|"]
    pattern = re.compile(r"torch.compile took ([0-9.]+) s in total")
    init = re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) s")
    for entry in entries:
        # summary.json records server paths; map "<batch>/<model>/perf/performance" to the local copy
        base = root / Path(entry["root"]).parts[-3] / "perf" / "performance"
        r1 = base / "inductor.repeat1.server.log"
        r2 = base / "inductor.repeat2.server.log"
        text1 = r1.read_text(encoding="utf-8", errors="replace") if r1.is_file() else ""
        text2 = r2.read_text(encoding="utf-8", errors="replace") if r2.is_file() else ""
        m1 = pattern.search(text1)
        i1 = init.search(text1)
        i2 = init.search(text2)
        lines.append(f"| {entry['name']} | {i1.group(1) if i1 else '—'} s | "
                     f"{i2.group(1) if i2 else '—'} s | {'是' if 'Directly load the compiled graph' in text2 else '未检出'} |")
    return lines


def main():
    batch = Path(sys.argv[1])
    summary = load(batch / "summary.json")
    previous = load(sys.argv[2])["performance"] if len(sys.argv) > 2 else []
    entries = summary["performance"]
    print("## OVERVIEW")
    print("\n".join(overview(entries)))
    print("\n## COMPILE")
    print("\n".join(compile_times(batch, entries)))
    print("\n## VERSUS_PREVIOUS")
    print("\n".join(versus_previous(entries, previous)))
    print("\n## DETAILS")
    for entry in entries:
        print("\n".join(detail(entry)))
        print()


if __name__ == "__main__":
    main()
